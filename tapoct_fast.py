# -*- coding: utf-8 -*-
"""
tapoct_fast.py — TAPOCT-FD (Fast Differential Contextual Teacher)
=================================================================
Замена модуля TAPOCT из ноутбука TAPOCT_TAPOCB-NN_main.ipynb
(ячейки 9 [TAPOCBAgent.step], 16 [run_TAPOCB], 18 [explore_EPOS_cached,
run_parallel_EPOS, get_EPOS_strategy, get_exp_couriers]).

Идея (вместо 150 итераций x (K+1) вызовов солверов на полигон):
  0) две проверки при ОДИНАКОВОЙ цене -> бесценовой дифференциал
     dS = S_a - S_b = beta*dx1 + gamma*dx2   (ценовой член alpha*(U-P)/U сокращается);
  1) n проходов по всем K алгоритмам -> триплеты (T_k, CT_k, S_k);
  2) оценка частных производных клиентской функции
        dS/dx1 = beta_hat, dS/dx2 = gamma_hat, intercept = alpha_hat*(U-P)/U
     (ограниченный ридж-МНК по допустимым наблюдениям);
  3) бесценовая логистическая полезность алгоритма по УСРЕДНЁННЫМ параметрам
        l_k = beta_hat*x1(T_bar_k) + gamma_hat*x2(CT_bar_k)   (= dE[S]/dp_k);
  4) политика — замкнутое решение KL-регуляризованного OMD-шага
        p* = argmax_p  p.l - tau*KL(p||p0) - lambda*(p.T_bar - B)
        p*_k ∝ p0_k * exp((l_k - lambda*T_bar_k)/tau),
     tau = max_gap/ln K  =>  max_k l_k - E_{p*}[l] <= max_gap (при равномерном p0).
Число вызовов солверов: n*K (по умолчанию 2*4 = 8) вместо <=150*(4+1) = 750.

СХЕМА ВЛОЖЕННЫХ ВЫЗОВОВ
-----------------------
get_exp_couriers(...)                                     [главный процесс]
 └─ run_parallel_EPOS(...)                                [главный процесс]
     ├─ _load_pickle / _save_pickle                       (кэш результатов, маршрутов)
     ├─ prepare_worker_payload(...)                       (лёгкий payload -> .pkl, один раз)
     │    └─ build_state_matrix(...)                      (матрица (n+1)x(n+1), как build_state)
     └─ joblib.Parallel(loky, n_jobs=6, generator)
          └─ _worker(polygon, ...)                        [процесс-воркер]
               ├─ _get_payload(path)                      (ленивая загрузка 1 раз на процесс)
               ├─ _get_solvers(factory)                   (импорт солверов 1 раз на процесс)
               │    └─ _warmup_solvers(...)               (прогрев CUDA/LKH, вне замеров CT)
               └─ explore_EPOS_cached(...)
                    └─ run_TAPOCT_fast(...)
                         ├─ ClientFeedbackOracle(...)     (скрытый клиент: отдаёт только S)
                         └─ TAPOCTFastAgent.fit(...)
                              ├─ _collect_triplets(...)
                              │    ├─ evaluate(...)  ×(n*K)
                              │    │    ├─ _run_solver(...)        (солвер + perf_counter)
                              │    │    │    └─ route_cost_open(...) (единый пересчёт стоимости)
                              │    │    ├─ offer_features(...)
                              │    │    └─ oracle(...)
                              │    └─ label_from_triplets(...)  (после каждого прохода >= n_min)
                              └─ label_from_triplets(triplets, ...)   [чистая математика]
                                   ├─ price_free_differential(...)
                                   ├─ estimate_preferences(...)
                                   ├─ logistic_scores(...)
                                   ├─ satisfaction_gradients(...)
                                   ├─ solve_kl_policy(...)
                                   ├─ prob_best(...)
                                   ├─ kl_divergence(...)
                                   └─ policy_to_counts(...)
get_EPOS_strategy(results)                                (совместим со старым форматом)
relabel_from_triplets(results, ...)                       (переразметка БЕЗ вызова солверов:
                                                           абляции, другие распределения предпочтений)
"""
from __future__ import annotations

import importlib
import math
import os
import pickle
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed

try:
    from scipy.optimize import lsq_linear
except Exception:  # pragma: no cover
    lsq_linear = None

ALGORITHMS: Tuple[str, ...] = ("MatNet", "NLNS", "lkh3", "Christofides")

# Сигнатура солвера как в OptimizerInterface: (points, prev_point, cost_matrix) -> (route, route_cost)
SolverFn = Callable[[List[Any], Any, np.ndarray], Tuple[List[Any], float]]


# =============================================================================
# 0. Конфигурации
# =============================================================================
@dataclass
class TAPOCTFastConfig:
    """
    Гиперпараметры TAPOCT-FD.

    algorithms              : tuple[str], длина K       — порядок = порядок классов MLP
    reuse_probes            : bool  — 2 пробы входят в первый проход (экономия 2 вызовов)
    n_probe                 : int   — число проверок при одинаковой цене (2)
    n_passes_min/max        : int   — мин/макс число проходов по всем K алгоритмам
    label_mode              : "omd" | "prob_best" — какая политика идёт в метки
    max_gap                 : float — допустимый разрыв max_k l_k - E_p[l] (задаёт tau)
    tau_min                 : float — нижняя граница температуры
    B                       : float — бюджет ожидаемого времени маршрута (как config.B)
    ridge                   : float — сила ридж-притяжения (alpha,beta,gamma) к pref_prior
    pref_prior              : (3,)  — априорные (alpha, beta, gamma)
    z_conf                  : float — z-порог разделимости лидера и второго места
    policy_tol              : float — порог L1-изменения политики между проходами
    recompute_cost          : bool  — пересчитывать стоимость маршрута по матрице единообразно
    n_mc                    : int   — число MC-выборок для prob_best
    counts_total            : int   — псевдосчётчики algorithm_wins (совместимость с NN)
    warmup                  : bool  — прогрев солверов в каждом воркере вне замеров CT
    seed                    : int
    """
    algorithms: Tuple[str, ...] = ALGORITHMS
    reuse_probes: bool = True
    n_probe: int = 2
    n_passes_min: int = 2
    n_passes_max: int = 3
    label_mode: str = "omd"
    max_gap: float = 0.01
    tau_min: float = 1e-4
    B: float = 43200.0
    ridge: float = 1e-3
    pref_prior: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    z_conf: float = 1.96
    policy_tol: float = 0.02
    recompute_cost: bool = True
    n_mc: int = 4000
    counts_total: int = 1000
    warmup: bool = True
    seed: int = 42


@dataclass
class TenderSpec:
    """
    Публичные параметры тендера (известны агенту).

    U                       : float — верхняя цена клиента
    price                   : float — цена предложения P (фиксирована в TAPOCT)
    CT_SA                   : float — лимит времени отклика, сек
    time_window             : (2,) float — (start, end), сек
    n_orders                : int   — число заказов (без депо)
    service_time_per_order  : float — e(A|i), сек на заказ ("additional"); 0 = как в eval_client_reward
    disqualify_ct           : bool  — S=0 при CT > CT_SA (как в DCMAB), False = как eval_client_reward
    clip                    : (2,) float — обрезка S (как np.clip(.., 0, 5) в тестах)
    """
    U: float
    price: float
    CT_SA: float
    time_window: Tuple[float, float]
    n_orders: int
    service_time_per_order: float = 0.0
    disqualify_ct: bool = False
    clip: Tuple[float, float] = (0.0, 5.0)

    @property
    def tw_len(self) -> float:
        """-> float: длина временного окна TW_len = end - start."""
        return float(self.time_window[1] - self.time_window[0])

    @property
    def additional(self) -> float:
        """-> float: суммарное сервисное время = n_orders * service_time_per_order."""
        return float(self.n_orders * self.service_time_per_order)

    @staticmethod
    def from_dynamic_config(dc: Dict[str, Any], n_orders: int, **kw) -> "TenderSpec":
        """
        dc        : dict {'U','price','CT_SA','time_window'} — как dynamic_config в ноутбуке
        n_orders  : int
        kw        : service_time_per_order / disqualify_ct / clip
        -> TenderSpec
        """
        return TenderSpec(U=float(dc["U"]), price=float(dc["price"]), CT_SA=float(dc["CT_SA"]),
                          time_window=tuple(dc["time_window"]), n_orders=int(n_orders), **kw)


@dataclass
class Triplet:
    """
    Одно наблюдение (одна оценка одного алгоритма).

    algorithm   : str
    alg_idx     : int   — индекс в config.algorithms
    route_cost  : float — T, стоимость (время) маршрута, сек
    ct          : float — CT, время вычисления маршрута, сек
    feedback    : float — S, скаляр удовлетворённости, который вернул клиент
    price       : float — P, цена, при которой получен S
    feasible    : bool  — маршрут уложился в окно (и в CT_SA, если disqualify_ct)
    valid_route : bool  — маршрут — перестановка всех точек с депо в начале
    phase       : str   — "probe" | "pass{j}"
    """
    algorithm: str
    alg_idx: int
    route_cost: float
    ct: float
    feedback: float
    price: float
    feasible: bool
    valid_route: bool
    phase: str


# =============================================================================
# 1. Клиент (оракул) и признаки предложения
# =============================================================================
def offer_features(price: float, route_cost: float, ct: float,
                   spec: TenderSpec) -> Tuple[np.ndarray, bool]:
    """
    Нормированные регрессоры клиентской функции (знает агент: U, TW, CT_SA публичны).

    price, route_cost, ct : float
    spec                  : TenderSpec
    -> x        : np.ndarray shape (3,) = [x0, x1, x2]
                  x0 = (U - P)/U,  x1 = (TW_len - T - additional)/TW_len,  x2 = (CT_SA - CT)/CT_SA
    -> feasible : bool — T + additional <= TW_len  (и CT <= CT_SA, если spec.disqualify_ct)
    """
    tw = spec.tw_len
    d_tw = tw - route_cost - spec.additional
    x = np.array([(spec.U - price) / spec.U, d_tw / tw, (spec.CT_SA - ct) / spec.CT_SA], dtype=float)
    feasible = d_tw >= 0.0 and (not spec.disqualify_ct or ct <= spec.CT_SA)
    return x, bool(feasible)


class ClientFeedbackOracle:
    """
    Скрытый клиент. Хранит истинные (alpha,beta,gamma); агент получает ТОЛЬКО скаляр S.
    Совпадает с eval_client_reward (ячейка 29) при service_time_per_order=0, disqualify_ct=False.

    __init__:
      prefs     : dict {'alpha','beta','gamma'} (float)
      spec      : TenderSpec
      noise_sd  : float — СКО аддитивного шума обратной связи (0 = детерминированно)
      quantize  : Optional[int] — если задано, S округляется на сетку из quantize уровней
                  (задел под задачу 4: оценка по 5-балльной шкале)
      rng       : np.random.Generator
    """

    def __init__(self, prefs: Dict[str, float], spec: TenderSpec, noise_sd: float = 0.0,
                 quantize: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        self._theta = np.array([prefs["alpha"], prefs["beta"], prefs["gamma"]], dtype=float)
        self.spec = spec
        self.noise_sd = float(noise_sd)
        self.quantize = quantize
        self.rng = rng or np.random.default_rng(0)

    def __call__(self, price: float, route_cost: float, ct: float, valid_route: bool = True) -> float:
        """
        price, route_cost, ct : float;  valid_route : bool
        -> S : float — 0, если маршрут невалиден/недопустим, иначе clip(theta·x [+шум])
        """
        x, feasible = offer_features(price, route_cost, ct, self.spec)
        if not (valid_route and feasible):
            return 0.0
        s = float(self._theta @ x)
        if self.noise_sd > 0:
            s += float(self.rng.normal(0.0, self.noise_sd))
        lo, hi = self.spec.clip
        if self.quantize:
            step = (hi - lo) / (self.quantize - 1)
            s = lo + step * round((s - lo) / step)
        return float(np.clip(s, lo, hi))


# =============================================================================
# 2. Чистая математика TAPOCT-FD
# =============================================================================
def price_free_differential(t_a: Triplet, t_b: Triplet, spec: TenderSpec) -> Dict[str, float]:
    """
    Две проверки при одинаковой цене: ценовой член alpha*x0 сокращается.

    t_a, t_b : Triplet (t_a.price == t_b.price)
    spec     : TenderSpec
    -> dict {'dS','dx1','dx2','same_price','both_feasible'} (float/bool)
       dS = S_a - S_b = beta*dx1 + gamma*dx2   (если обе оценки допустимы)
    """
    xa, fa = offer_features(t_a.price, t_a.route_cost, t_a.ct, spec)
    xb, fb = offer_features(t_b.price, t_b.route_cost, t_b.ct, spec)
    return {"dS": t_a.feedback - t_b.feedback, "dx1": float(xa[1] - xb[1]), "dx2": float(xa[2] - xb[2]),
            "same_price": bool(abs(t_a.price - t_b.price) < 1e-9),
            "both_feasible": bool(fa and fb and t_a.valid_route and t_b.valid_route)}


def estimate_preferences(triplets: Sequence[Triplet], spec: TenderSpec,
                         ridge: float = 1e-3,
                         prior: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)) -> Dict[str, Any]:
    """
    Оценка частных производных клиентской функции по допустимым наблюдениям.
    Модель: S_e = alpha*x0_e + beta*x1_e + gamma*x2_e,  0 <= alpha,beta,gamma <= 1.
    При одной цене x0 = const => alpha*x0 = c0 (свободный член), alpha_hat = c0/x0.

    triplets : Sequence[Triplet], длина m
    spec     : TenderSpec
    ridge    : float — вес притяжения к prior (стабилизация при коллинеарности)
    prior    : (3,) — (alpha0, beta0, gamma0)
    -> dict:
       'alpha','beta','gamma' : float — оценки (= -U*dS/dP, TW*dS/d(dTW), CT_SA*dS/d(dCT))
       'c0'                   : float — ценовой «шум» alpha*(U-P)/U
       'rmse'                 : float — СКО остатков на допустимых наблюдениях
       'n_used'               : int
       'identifiable'         : bool  — rank([x1,x2] центрированных) == 2
       'price_varies'         : bool
    """
    rows, ys = [], []
    for t in triplets:
        x, feas = offer_features(t.price, t.route_cost, t.ct, spec)
        if feas and t.valid_route and t.feedback > spec.clip[0]:
            rows.append(x)
            ys.append(t.feedback)
    a0, b0, g0 = prior
    if len(rows) == 0:
        x0 = (spec.U - spec.price) / spec.U
        return {"alpha": a0, "beta": b0, "gamma": g0, "c0": a0 * x0, "rmse": float("nan"),
                "n_used": 0, "identifiable": False, "price_varies": False}
    X = np.asarray(rows, float)            # (m, 3)
    y = np.asarray(ys, float)              # (m,)
    price_varies = bool(np.ptp(X[:, 0]) > 1e-9)
    Xc = X[:, 1:] - X[:, 1:].mean(0, keepdims=True)
    identifiable = bool(len(rows) >= 3 and np.linalg.matrix_rank(Xc, tol=1e-10) == 2)

    if price_varies:
        A, theta0 = X, np.array([a0, b0, g0])                         # theta = (alpha, beta, gamma)
        lb, ub = np.zeros(3), np.ones(3)
    else:
        x0 = float(X[0, 0])
        A = np.column_stack([np.ones(len(y)), X[:, 1], X[:, 2]])       # theta = (c0, beta, gamma)
        theta0 = np.array([a0 * x0, b0, g0])
        lb, ub = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    lam = math.sqrt(max(ridge, 0.0))
    A_aug = np.vstack([A, lam * np.eye(3)])
    y_aug = np.concatenate([y, lam * theta0])
    if lsq_linear is not None:
        theta = lsq_linear(A_aug, y_aug, bounds=(lb, ub), method="bvls").x
    else:  # запасной вариант: ридж + обрезка
        theta = np.clip(np.linalg.lstsq(A_aug, y_aug, rcond=None)[0], lb, ub)
    rmse = float(np.sqrt(np.mean((A @ theta - y) ** 2)))
    if price_varies:
        alpha, beta, gamma = theta
        c0 = float(alpha * (spec.U - spec.price) / spec.U)
    else:
        c0, beta, gamma = theta
        x0 = float(X[0, 0])
        alpha = float(np.clip(c0 / x0, 0.0, 1.0)) if abs(x0) > 1e-12 else a0
    return {"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma), "c0": float(c0),
            "rmse": rmse, "n_used": int(len(y)), "identifiable": identifiable,
            "price_varies": price_varies}


def logistic_scores(triplets: Sequence[Triplet], theta: Dict[str, Any], spec: TenderSpec,
                    K: int) -> Dict[str, np.ndarray]:
    """
    Бесценовая логистическая полезность алгоритмов по усреднённым параметрам.

    l_k = (1-q_k)*[beta*x1(T_bar_k) + gamma*x2(CT_bar_k)] + q_k*(-c0)
      q_k — доля недопустимых оценок алгоритма k (для них S=0 => логистическая часть = -c0).
    Если оценка theta неидентифицируема — эмпирический вариант l_e = S_e - c0.

    triplets : Sequence[Triplet];  theta : dict из estimate_preferences;  spec : TenderSpec;  K : int
    -> dict (все массивы shape (K,)):
       'ell'     — l_k (= dE[S]/dp_k без константы c0)
       'ell_emp' — эмпирическое среднее S_e - c0
       'se'      — стандартная ошибка l_k (inf, если 0 наблюдений; пул по алгоритмам при n_k = 1)
       'T_bar','CT_bar' — средние T и CT по допустимым оценкам (по всем, если допустимых нет)
       'n','feasible_rate'
    """
    beta, gamma, c0 = theta["beta"], theta["gamma"], theta["c0"]
    ell = np.full(K, np.nan); ell_emp = np.full(K, np.nan); var = np.full(K, np.nan)
    T_bar = np.full(K, np.nan); CT_bar = np.full(K, np.nan)
    n = np.zeros(K, int); feas_rate = np.zeros(K)
    for k in range(K):
        tk = [t for t in triplets if t.alg_idx == k]
        n[k] = len(tk)
        if n[k] == 0:
            continue
        feas = np.array([offer_features(t.price, t.route_cost, t.ct, spec)[1] and t.valid_route for t in tk])
        T = np.array([t.route_cost for t in tk], float)
        CT = np.array([t.ct for t in tk], float)
        S = np.array([t.feedback for t in tk], float)
        q = 1.0 - feas.mean()
        feas_rate[k] = 1.0 - q
        Tm = T[feas].mean() if feas.any() else T.mean()
        CTm = CT[feas].mean() if feas.any() else CT.mean()
        T_bar[k], CT_bar[k] = Tm, CTm
        x1 = (spec.tw_len - Tm - spec.additional) / spec.tw_len
        x2 = (spec.CT_SA - CTm) / spec.CT_SA
        ell[k] = (1 - q) * (beta * x1 + gamma * x2) + q * (-c0)
        per_e = np.where(feas, beta * (spec.tw_len - T - spec.additional) / spec.tw_len
                         + gamma * (spec.CT_SA - CT) / spec.CT_SA, -c0)
        ell_emp[k] = float(np.mean(S - c0))
        var[k] = np.var(per_e, ddof=1) if n[k] > 1 else np.nan
    if not theta.get("identifiable", False):
        ell = np.where(np.isnan(ell_emp), ell, ell_emp)
    pooled = np.nanmean(var) if np.any(~np.isnan(var)) else np.inf
    var = np.where(np.isnan(var), pooled, var)
    se = np.where(n > 0, np.sqrt(np.maximum(var, 0.0) / np.maximum(n, 1)), np.inf)
    return {"ell": ell, "ell_emp": ell_emp, "se": se, "T_bar": T_bar, "CT_bar": CT_bar,
            "n": n, "feasible_rate": feas_rate}


def satisfaction_gradients(theta: Dict[str, Any], spec: TenderSpec, ell: np.ndarray) -> Dict[str, Any]:
    """
    Частные производные ожидаемой удовлетворённости.

    theta : dict (alpha,beta,gamma); spec : TenderSpec; ell : (K,)
    -> dict:
       'dS_dP'   : float = -alpha/U          (ценовой канал — исключён из оптимизации политики)
       'dS_dT'   : float = -beta/TW_len      (стоимость маршрута)
       'dS_dCT'  : float = -gamma/CT_SA      (время вычисления)
       'grad_p'  : (K,)  = dE[S]/dp_k = l_k  (+ константа c0, не влияющая на симплекс)
       'grad_simplex' : (K,) = l - mean(l)   (проекция на касательное пространство симплекса)
    """
    g = np.nan_to_num(ell, nan=np.nanmin(ell) if np.any(~np.isnan(ell)) else 0.0)
    return {"dS_dP": -theta["alpha"] / spec.U, "dS_dT": -theta["beta"] / spec.tw_len,
            "dS_dCT": -theta["gamma"] / spec.CT_SA, "grad_p": g, "grad_simplex": g - g.mean()}


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """p, q : (K,) распределения -> float KL(p||q) (с eps-обрезкой)."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def solve_kl_policy(ell: np.ndarray, prior: np.ndarray, tau: float,
                    T_bar: Optional[np.ndarray] = None, B: float = np.inf,
                    iters: int = 60) -> Tuple[np.ndarray, float]:
    """
    Замкнутое решение KL-регуляризованного шага OMD с бюджетным ограничением:
        max_p  p·l - tau*KL(p||p0)   s.t.  p·T_bar <= B
        p*_k ∝ p0_k * exp((l_k - lambda*T_bar_k)/tau),  lambda >= 0 — бисекция.

    ell   : (K,)  — логистические полезности (NaN -> -inf: алгоритм не оценён)
    prior : (K,)  — p0 (тёплый старт; равномерный по умолчанию)
    tau   : float > 0
    T_bar : (K,) или None;  B : float
    -> p      : (K,) — политика на симплексе
    -> lambda : float — множитель Лагранжа бюджета (0, если ограничение неактивно)
    """
    l = np.where(np.isnan(ell), -np.inf, ell).astype(float)
    logp0 = np.log(np.clip(prior, 1e-12, 1.0))

    def _p(lmb: float) -> np.ndarray:
        z = logp0 + (l - (lmb * np.nan_to_num(T_bar, nan=0.0) if T_bar is not None else 0.0)) / tau
        z = z - np.max(z[np.isfinite(z)])
        w = np.where(np.isfinite(z), np.exp(z), 0.0)
        return w / w.sum()

    p = _p(0.0)
    if T_bar is None or not np.isfinite(B) or float(p @ np.nan_to_num(T_bar)) <= B:
        return p, 0.0
    lo, hi = 0.0, 1.0
    while float(_p(hi) @ np.nan_to_num(T_bar)) > B and hi < 1e12:
        hi *= 10.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(_p(mid) @ np.nan_to_num(T_bar)) > B:
            lo = mid
        else:
            hi = mid
    return _p(hi), float(hi)


def prob_best(ell: np.ndarray, se: np.ndarray, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """
    Вероятность оптимальности алгоритма (семантика старых меток algorithm_wins:
    частота argmax по повторным зашумлённым оценкам) — Монте-Карло по N(l_k, se_k^2).

    ell, se : (K,);  n_mc : int;  rng : Generator
    -> (K,) — распределение на симплексе
    """
    l = np.where(np.isnan(ell), -np.inf, ell)
    s = np.where(np.isfinite(se), se, 0.0)
    draws = l[None, :] + s[None, :] * rng.standard_normal((n_mc, len(l)))
    winners = np.argmax(draws, axis=1)
    return np.bincount(winners, minlength=len(l)) / n_mc


def policy_to_counts(p: np.ndarray, total: int = 1000) -> np.ndarray:
    """
    Псевдосчётчики для совместимости с get_EPOS_strategy/Supportive NN (метод наибольших остатков).
    p : (K,) -> (K,) int, сумма = total
    """
    raw = np.asarray(p, float) * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem > 0:
        base[np.argsort(-(raw - base))[:rem]] += 1
    return base


def label_from_triplets(triplets: Sequence[Triplet], spec: TenderSpec, cfg: TAPOCTFastConfig,
                        prior: Optional[np.ndarray] = None,
                        rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """
    Чистая (без солверов) разметка по набору триплетов.

    triplets : Sequence[Triplet], длина m;  spec : TenderSpec;  cfg : TAPOCTFastConfig
    prior    : (K,) или None (равномерный);  rng : Generator
    -> dict: 'policy' (K,), 'policy_omd' (K,), 'policy_prob_best' (K,), 'theta_hat', 'scores',
             'gradients', 'probe', 'tau', 'lambda_budget', 'expected_gap', 'loss',
             'kl_divergence', 'converged', 'chosen_idx'
    """
    K = len(cfg.algorithms)
    rng = rng or np.random.default_rng(cfg.seed)
    p0 = np.ones(K) / K if prior is None else np.asarray(prior, float) / np.sum(prior)
    probes = [t for t in triplets if t.phase == "probe"][:2]
    probe = price_free_differential(probes[0], probes[1], spec) if len(probes) == 2 else None
    theta = estimate_preferences(triplets, spec, cfg.ridge, cfg.pref_prior)
    sc = logistic_scores(triplets, theta, spec, K)
    grads = satisfaction_gradients(theta, spec, sc["ell"])
    tau = max(cfg.tau_min, cfg.max_gap / math.log(K))
    p_omd, lam = solve_kl_policy(sc["ell"], p0, tau, sc["T_bar"], cfg.B)
    p_pb = prob_best(sc["ell"], sc["se"], cfg.n_mc, rng)
    policy = p_omd if cfg.label_mode == "omd" else p_pb
    l_fin = np.where(np.isnan(sc["ell"]), np.nanmin(sc["ell"]) if np.any(~np.isnan(sc["ell"])) else 0, sc["ell"])
    order = np.argsort(-l_fin)
    gap12 = l_fin[order[0]] - l_fin[order[1]] if K > 1 else np.inf
    se12 = math.sqrt(sc["se"][order[0]] ** 2 + sc["se"][order[1]] ** 2) if K > 1 else 0.0
    separated = bool(np.isfinite(se12) and gap12 > cfg.z_conf * se12)
    over_budget = float(policy @ np.nan_to_num(sc["T_bar"])) > cfg.B
    loss = float(policy @ l_fin - tau * kl_divergence(policy, p0) - (1e3 if over_budget else 0.0))
    if probe is not None and probe["both_feasible"]:
        probe["model_residual"] = float(probe["dS"] - theta["beta"] * probe["dx1"] - theta["gamma"] * probe["dx2"])
    return {"policy": policy, "policy_omd": p_omd, "policy_prob_best": p_pb, "theta_hat": theta,
            "scores": sc, "gradients": grads, "probe": probe, "tau": tau, "lambda_budget": lam,
            "expected_gap": float(l_fin.max() - policy @ l_fin), "loss": loss,
            "kl_divergence": kl_divergence(policy, p0), "converged": separated,
            "chosen_idx": int(np.argmax(policy))}


# =============================================================================
# 3. Агент: сбор триплетов (единственное место, где вызываются солверы)
# =============================================================================
def route_cost_open(route: Sequence[Any], order_index: Dict[Any, int], matrix: np.ndarray) -> float:
    """
    Стоимость ОТКРЫТОГО пути (без возврата в депо) — единая для всех солверов.
    route : Sequence[id] длины n+1;  order_index : dict id->idx;  matrix : (n+1, n+1)
    -> float
    """
    idx = [order_index[r] for r in route]
    return float(sum(matrix[idx[i], idx[i + 1]] for i in range(len(idx) - 1)))


class TAPOCTFastAgent:
    """
    __init__:
      solvers : dict[str, SolverFn] — ключи = cfg.algorithms
      cfg     : TAPOCTFastConfig
      rng     : np.random.Generator
    """

    def __init__(self, solvers: Dict[str, SolverFn], cfg: TAPOCTFastConfig,
                 rng: Optional[np.random.Generator] = None):
        missing = [a for a in cfg.algorithms if a not in solvers]
        if missing:
            raise KeyError(f"нет солверов для: {missing}")
        self.solvers, self.cfg = solvers, cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.best_routes: Dict[int, Tuple[float, List[Any]]] = {}

    def _run_solver(self, alg: str, orders: List[Any], matrix: np.ndarray,
                    order_index: Dict[Any, int]) -> Tuple[List[Any], float, float, bool]:
        """
        alg : str;  orders : list[id] длины n+1 (orders[0] — депо);  matrix : (n+1, n+1)
        -> route : list[id];  cost : float;  ct : float (сек, perf_counter);  valid : bool
        """
        t0 = time.perf_counter()
        out = self.solvers[alg](orders, orders[0], matrix)
        ct = time.perf_counter() - t0
        route, cost = (out if out is not None else (None, None))
        valid = (route is not None and len(route) == len(orders)
                 and set(route) == set(orders) and route[0] == orders[0])
        if valid and self.cfg.recompute_cost:
            cost = route_cost_open(route, order_index, matrix)
        return (list(route) if route is not None else []), (float(cost) if cost is not None else float("inf")), ct, bool(valid)

    def evaluate(self, k: int, orders: List[Any], matrix: np.ndarray, order_index: Dict[Any, int],
                 oracle: ClientFeedbackOracle, price: float, phase: str) -> Triplet:
        """
        Одна оценка алгоритма k: солвер -> (T, CT) -> клиент -> S.
        k : int; orders : list[id] (n+1); matrix : (n+1,n+1); order_index : dict; oracle; price : float
        -> Triplet
        """
        alg = self.cfg.algorithms[k]
        route, cost, ct, valid = self._run_solver(alg, orders, matrix, order_index)
        S = oracle(price, cost, ct, valid)
        _, feas = offer_features(price, cost, ct, oracle.spec)
        if valid and (k not in self.best_routes or cost < self.best_routes[k][0]):
            self.best_routes[k] = (cost, route)
        return Triplet(alg, k, cost, ct, S, price, bool(feas and valid), valid, phase)

    def fit(self, orders: List[Any], matrix: np.ndarray, oracle: ClientFeedbackOracle,
            price: float, prior: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], List[Triplet]]:
        """
        Полный TAPOCT-FD для одного полигона.
        orders : list[id] (n+1); matrix : (n+1,n+1); oracle; price : float; prior : (K,) | None
        -> label : dict (см. label_from_triplets) + 'n_passes'
        -> triplets : list[Triplet], длина = n_passes*K (+2, если reuse_probes=False)
        """
        cfg, K = self.cfg, len(self.cfg.algorithms)
        order_index = {o: i for i, o in enumerate(orders)}
        p0 = np.ones(K) / K if prior is None else np.asarray(prior, float) / np.sum(prior)
        trip: List[Triplet] = []
        # --- фаза 0: две проверки при одинаковой цене (алгоритмы ~ p0 без возвращения) ---
        probe_idx = list(self.rng.choice(K, size=min(cfg.n_probe, K), replace=False, p=p0))
        for k in probe_idx:
            trip.append(self.evaluate(int(k), orders, matrix, order_index, oracle, price, "probe"))
        # --- фаза 1: n проходов по всем алгоритмам ---
        prev_policy, label, n_pass = None, None, 0
        for j in range(1, cfg.n_passes_max + 1):
            n_pass = j
            ks = [k for k in range(K) if not (j == 1 and cfg.reuse_probes and k in probe_idx)]
            for k in ks:
                trip.append(self.evaluate(k, orders, matrix, order_index, oracle, price, f"pass{j}"))
            if j >= cfg.n_passes_min:
                label = label_from_triplets(trip, oracle.spec, cfg, p0, self.rng)
                stable = prev_policy is not None and np.abs(label["policy"] - prev_policy).sum() < cfg.policy_tol
                if label["converged"] or stable:
                    label["converged"] = True
                    break
                prev_policy = label["policy"]
        if label is None:
            label = label_from_triplets(trip, oracle.spec, cfg, p0, self.rng)
        label["n_passes"] = n_pass
        return label, trip


# =============================================================================
# 4. Полигон: матрица, запуск, совместимый с прежним форматом результат
# =============================================================================
def build_state_matrix(ideal_matrix: Union[np.ndarray, List[List[float]]], depot_row: Sequence[float],
                       traffic_factor: float = 1.0) -> np.ndarray:
    """
    Аналог TAPOCBAgent.build_state + build_cost_matrix (open-ATSP трюк: столбец 0 = 0).
    c_ab = round(d_ab * (1 + max(0, x_SUMO))) — формула из статьи; traffic_factor = 1 + max(0, x).

    ideal_matrix : (n, n) — матрица полигона (AUG_global_map[p]['distance matrix'])
    depot_row    : (n,)   — расстояния депо(0) -> заказ j
    -> (n+1, n+1) int64
    """
    D = np.asarray(ideal_matrix, dtype=float)
    n = D.shape[0]
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[1:, 1:] = D
    M[0, 1:] = np.asarray(depot_row, dtype=float)
    return np.rint(M * float(traffic_factor)).astype(np.int64)


def run_TAPOCT_fast(polygon: str, matrix: np.ndarray, orders_ids: List[Any], client_prefs: Dict[str, float],
                    dynamic_config: Dict[str, Any], solvers: Dict[str, SolverFn],
                    cfg: Optional[TAPOCTFastConfig] = None, prior: Optional[np.ndarray] = None,
                    spec_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[Any], List[Dict]]:
    """
    Замена run_TAPOCB(...) для одного полигона.

    polygon        : str
    matrix         : (n+1, n+1) — из build_state_matrix
    orders_ids     : list[id] длины n (AUG_global_map[p]['full map']); депо 0 добавляется здесь
    client_prefs   : dict {'alpha','beta','gamma'} — ИСТИННЫЕ, видит только оракул
    dynamic_config : dict {'U','price','CT_SA','time_window'}
    solvers        : dict[str, SolverFn]
    cfg            : TAPOCTFastConfig | None;  prior : (K,) | None;  spec_kwargs : dict | None
    -> result   : dict — ключи старого step() + новые диагностические
    -> path     : list[id] — лучший маршрут (алгоритм argmax политики), а не всегда NLNS
    -> triplets : list[dict] — сырые наблюдения (для relabel_from_triplets)
    """
    cfg = cfg or TAPOCTFastConfig()
    seed = (cfg.seed * 1_000_003 + (abs(hash(polygon)) % 1_000_003)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    orders = [0] + list(orders_ids)
    spec = TenderSpec.from_dynamic_config(dynamic_config, len(orders_ids), **(spec_kwargs or {}))
    oracle = ClientFeedbackOracle(client_prefs, spec, rng=rng)
    agent = TAPOCTFastAgent(solvers, cfg, rng)
    t0 = time.perf_counter()
    label, trip = agent.fit(orders, matrix, oracle, spec.price, prior)
    k_best = label["chosen_idx"]
    path = agent.best_routes.get(k_best, (None, []))[1]
    sc = label["scores"]
    result = {
        # ---- ключи прежнего TAPOCBAgent.step() (совместимость с ячейками 18–24 и Supportive NN) ----
        "iteration": len(trip),
        "policy": label["policy"],
        "loss": label["loss"],
        "kl_divergence": label["kl_divergence"],
        "chosen_algorithm": cfg.algorithms[k_best],
        "chosen_reward": float(np.nan_to_num(sc["ell"][k_best])),
        "converged": bool(label["converged"]),
        "reward_spread": float(np.nanmax(sc["ell"]) - np.nanmin(sc["ell"])),
        "algorithm_wins": policy_to_counts(label["policy"], cfg.counts_total),
        "optimal_route": path,
        # ---- новое ----
        "policy_omd": label["policy_omd"],
        "policy_prob_best": label["policy_prob_best"],
        "ell_hat": sc["ell"], "ell_se": sc["se"], "T_bar": sc["T_bar"], "CT_bar": sc["CT_bar"],
        "feasible_rate": sc["feasible_rate"],
        "theta_hat": label["theta_hat"], "gradients": label["gradients"], "probe": label["probe"],
        "tau": label["tau"], "lambda_budget": label["lambda_budget"],
        "expected_gap": label["expected_gap"], "n_passes": label["n_passes"],
        "n_solver_calls": len(trip), "wall_time": time.perf_counter() - t0,
        "spec": asdict(spec), "algorithms": list(cfg.algorithms),
    }
    return result, path, [asdict(t) for t in trip]


# =============================================================================
# 5. Параллельная обработка (логика прежнего run_parallel_EPOS сохранена)
# =============================================================================
EPOS_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_FILE = "epos_cache_fd.pkl"
ROUTE_FILE = "Routes_cache_fd.pkl"
PAYLOAD_FILE = "tapoct_payload.pkl"
_PROC_CACHE: Dict[Any, Any] = {}       # кэш ВНУТРИ процесса-воркера (payload, солверы, прогрев)


def _save_pickle(path: str, obj: Any) -> None:
    """path : str; obj : Any -> None (атомарная запись через .tmp)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pickle(path: str, default: Any) -> Any:
    """path : str; default : Any -> содержимое файла или default."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def prepare_worker_payload(polygons: Sequence[str], aug_map: Dict[str, Dict[str, Any]],
                           prefs_map: Dict[str, Dict[str, float]],
                           depot_row_fn: Callable[[List[Any]], Sequence[float]],
                           path: str = PAYLOAD_FILE, traffic_factor: float = 1.0) -> str:
    """
    Готовит ЛЁГКИЙ payload для воркеров один раз в главном процессе, чтобы joblib не
    сериализовал в каждую задачу distance_array / AUG_global_map / модели (как сейчас через
    глобалы ноутбука).

    polygons     : Sequence[str], длина P
    aug_map      : AUG_global_map
    prefs_map    : GLOBAL_PREFERENCES_MAP
    depot_row_fn : f(full_map_ids: list[id]) -> (n,) расстояния депо->заказ
                   (в ноутбуке: [get_distance(distance_array, idx_dict, 0, get_basic_order_ID(o)) for o in ids])
    path         : str;  traffic_factor : float
    -> path : str  (файл: {polygon: {'matrix': (n+1,n+1) int64, 'orders': list[id], 'prefs': dict}})
    """
    payload = {}
    for p in polygons:
        ids = list(aug_map[p]["full map"])
        # Keep the depot row on the same road graph as the augmented OD matrix.
        # Originals retain the operational depot distances from depot_row_fn.
        depot_row = aug_map[p].get("depot row")
        if depot_row is None:
            depot_row = depot_row_fn(ids)
        payload[p] = {"matrix": build_state_matrix(aug_map[p]["distance matrix"], depot_row, traffic_factor),
                      "orders": ids, "prefs": dict(prefs_map[p])}
    _save_pickle(path, payload)
    return path


def _get_payload(path: str) -> Dict[str, Any]:
    """path : str -> dict payload (загружается 1 раз на процесс; перечитывается при изменении файла)."""
    key = ("payload", os.path.abspath(path), os.path.getmtime(path))
    if key not in _PROC_CACHE:
        for k in [k for k in _PROC_CACHE if isinstance(k, tuple) and k[0] == "payload"]:
            del _PROC_CACHE[k]
        _PROC_CACHE[key] = _load_pickle(path, {})
    return _PROC_CACHE[key]


def _warmup_solvers(solvers: Dict[str, SolverFn],
                    algorithms: Sequence[str], n: int = 6) -> None:
    """Прогреть все солверы; не скрывать ошибку инициализации."""
    rng = np.random.default_rng(0)
    matrix = rng.integers(10, 100, size=(n, n))
    np.fill_diagonal(matrix, 0)
    matrix[:, 0] = 0
    points = list(range(n))

    for name in algorithms:
        if name not in solvers or not callable(solvers[name]):
            raise KeyError(f"Solver {name!r} is missing from the factory")
        try:
            route, _ = solvers[name](points, points[0], matrix)
            if (route is None or len(route) != n
                    or route[0] != points[0]
                    or set(route) != set(points)):
                raise ValueError(f"Invalid warm-up route: {route!r}")
        except Exception as exc:
            raise RuntimeError(f"Warm-up failed for solver {name!r}") from exc


def _get_solvers(factory: Union[str, Dict[str, SolverFn]], algorithms: Sequence[str],
                 warmup: bool) -> Dict[str, SolverFn]:
    """
    factory : "module:function" (рекомендуется; импорт 1 раз на процесс) | dict[str, SolverFn]
    algorithms : Sequence[str]; warmup : bool
    -> dict[str, SolverFn]
    """
    if isinstance(factory, dict):
        solvers = factory
        key = ("solvers_dict", id(factory))
    else:
        key = ("solvers", factory)
        if key not in _PROC_CACHE:
            mod, fn = factory.split(":")
            _PROC_CACHE[key] = getattr(importlib.import_module(mod), fn)()
        solvers = _PROC_CACHE[key]
    wkey = ("warm", key)
    if warmup and wkey not in _PROC_CACHE:
        _warmup_solvers(solvers, algorithms)
        _PROC_CACHE[wkey] = True
    return solvers


def explore_EPOS_cached(agent_id: Any, polygon: str, client_config: Dict[str, Any],
                        payload: Dict[str, Any], solvers: Dict[str, SolverFn],
                        cfg: TAPOCTFastConfig, spec_kwargs: Optional[Dict[str, Any]] = None,
                        prior: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Кэшируемый запуск TAPOCT-FD. ВСЕГДА возвращает (result, path) (в старой версии при
    попадании в кэш возвращался один dict -> ошибка распаковки в _worker).

    agent_id      : Any (не используется в расчёте, сохранён для совместимости)
    polygon       : str
    client_config : dict {'U','price','CT_SA','time_window', ...}
    payload       : dict из _get_payload;  solvers : dict;  cfg : TAPOCTFastConfig
    -> result : dict (+ 'triplets');  path : list[id]
    """
    if polygon in EPOS_CACHE:
        r = EPOS_CACHE[polygon]
        return r, r.get("optimal_route", [])
    item = payload[polygon]
    result, path, trip = run_TAPOCT_fast(polygon, item["matrix"], item["orders"], item["prefs"],
                                         client_config, solvers, cfg, prior, spec_kwargs)
    result["triplets"] = trip
    EPOS_CACHE[polygon] = result
    return result, path


def _worker(polygon: str, courier_id: Any, dynamic_config: Dict[str, Any], payload_path: str,
            solver_factory: Union[str, Dict[str, SolverFn]], cfg_dict: Dict[str, Any],
            spec_kwargs: Optional[Dict[str, Any]]) -> Tuple[str, Optional[Dict], Optional[List], Optional[str]]:
    """
    Задача одного воркера (лёгкие аргументы: строки и маленькие dict).
    -> (polygon, result | None, path | None, error_traceback | None)
    """
    try:
        cfg = TAPOCTFastConfig(**cfg_dict)
        payload = _get_payload(payload_path)
        solvers = _get_solvers(solver_factory, cfg.algorithms, cfg.warmup)
        res, path = explore_EPOS_cached(courier_id, polygon, dynamic_config, payload, solvers, cfg, spec_kwargs)
        return polygon, res, path, None
    except Exception:
        return polygon, None, None, traceback.format_exc()


def run_parallel_EPOS(
    courier_id: Any,
    polygons: Sequence[str],
    dynamic_config: Dict[str, Any],
    num_workers: int = 6,
    *,
    aug_map: Dict[str, Any],
    prefs_map: Dict[str, Any],
    depot_row_fn: Callable[[List[Any]], Sequence[float]],
    solver_factory: Union[str, Dict[str, SolverFn]] = "routing_solvers:get_solvers",
    cfg: Optional[TAPOCTFastConfig] = None,
    spec_kwargs: Optional[Dict[str, Any]] = None,
    cache_file: str = CACHE_FILE,
    route_file: str = ROUTE_FILE,
    payload_file: str = PAYLOAD_FILE,
    checkpoint_every: int = 20,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Сохранить успешные полигоны и явно сообщить обо всех ошибках воркеров."""
    global EPOS_CACHE

    cfg = cfg or TAPOCTFastConfig()
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")

    polygons = list(dict.fromkeys(polygons))
    results = _load_pickle(cache_file, {})
    opt_routes = _load_pickle(route_file, {})

    def valid_result(item):
        if not isinstance(item, dict) or "policy" not in item:
            return False
        try:
            p = np.asarray(item["policy"], dtype=float)
            return (
                p.shape == (len(cfg.algorithms),)
                and np.isfinite(p).all()
                and np.all(p >= 0)
                and p.sum() > 0
            )
        except (TypeError, ValueError):
            return False

    todo = [p for p in polygons if not valid_result(results.get(p))]
    if verbose:
        print(f"Processing {len(todo)} polygons; {len(polygons) - len(todo)} cached.")

    errors = {}
    n_new = 0

    if todo:
        payload_path = prepare_worker_payload(
            todo, aug_map, prefs_map, depot_row_fn, payload_file
        )
        cfg_dict = asdict(cfg)
        processed = Parallel(
            n_jobs=num_workers, backend="loky", return_as="generator"
        )(
            delayed(_worker)(
                p, courier_id, dynamic_config, payload_path,
                solver_factory, cfg_dict, spec_kwargs
            )
            for p in todo
        )

        for polygon, result, path, err in processed:
            if not valid_result(result):
                errors[polygon] = err or "Worker returned an invalid policy"
                continue

            results[polygon] = result
            opt_routes[polygon] = path
            EPOS_CACHE[polygon] = result
            n_new += 1

            if n_new % checkpoint_every == 0:
                _save_pickle(cache_file, results)
                _save_pickle(route_file, opt_routes)
                if verbose:
                    print(f"Checkpoint: {n_new} new polygons saved.")

        _save_pickle(cache_file, results)
        _save_pickle(route_file, opt_routes)

    missing = [p for p in polygons if not valid_result(results.get(p))]
    if missing:
        examples = "\n\n".join(
            f"{p}:\n{errors.get(p, 'No valid policy in cache or worker result')}"
            for p in missing[:3]
        )
        raise RuntimeError(
            f"TAPOCT failed for {len(missing)} of {len(polygons)} polygons. "
            f"Successful results were checkpointed.\n{examples}"
        )

    return {p: results[p] for p in polygons}


# =============================================================================
# 6. Совместимость с пайплайном NN и переразметка без солверов
# =============================================================================
def get_EPOS_strategy(results: Dict[str, Dict[str, Any]],
                      algorithms: Sequence[str] = ALGORITHMS) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Исправленный get_EPOS_strategy: берёт 'policy' напрямую (иначе — 'algorithm_wins'),
    правильные имена алгоритмов, без «протекания» poly_strat между полигонами.

    results : {polygon: result}
    -> global_strategy : list[float] длины K — средняя политика
    -> poly_strategy   : {polygon: list[float] длины K}
    """
    K = len(algorithms)
    poly, acc, n = {}, np.zeros(K), 0
    for pid, item in results.items():
        if "policy" in item and len(item["policy"]) == K:
            p = np.asarray(item["policy"], float)
        elif "algorithm_wins" in item and len(item["algorithm_wins"]) == K:
            w = np.asarray(item["algorithm_wins"], float); p = w / max(w.sum(), 1.0)
        else:
            continue
        p = p / p.sum()
        poly[pid] = p.tolist(); acc += p; n += 1
    return (acc / max(n, 1)).tolist(), poly


def get_exp_couriers(
    final_solution: Dict[Any, Any],
    nearest_fn: Callable[[float, float, int], List[str]],
    max_couriers: int = 20,
    checkpoint_every: int = 2,
    dynamic_config: Optional[Dict[str, Any]] = None,
    k_polygons: int = 2000,
    base_latlon: Tuple[float, float] = (55.752, 37.6177),
    checkpoint_file: str = "courier_training_checkpoint_fd_seconds.pkl",
    **run_kwargs,
) -> Dict[Any, Dict[str, Any]]:
    """Записывать курьера только после получения всех его политик."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")

    dynamic_config = dynamic_config or {
        "U": 900, "price": 500, "CT_SA": 5,
        "time_window": (0.0, 43200.0),
    }
    trained = _load_pickle(checkpoint_file, {})
    cfg = run_kwargs.get("cfg") or TAPOCTFastConfig()
    n_saved = 0

    for courier in list(final_solution)[:max_couriers]:
        if final_solution[courier].get("New route") == [0]:
            continue

        lat, lon = base_latlon
        polys = list(dict.fromkeys(nearest_fn(lat, lon, k_polygons)))
        if not polys:
            raise ValueError(f"Courier {courier!r}: nearest_fn returned no polygons")

        old = trained.get(courier, {})
        old_local = old.get("EPOS_local_strat", {})
        old_global = np.asarray(old.get("EPOS_global_strat", []), dtype=float)
        complete = (
            old_global.shape == (len(cfg.algorithms),)
            and np.isfinite(old_global).all()
            and old_global.sum() > 0
            and all(p in old_local for p in polys)
        )
        if complete:
            continue

        # Существующая неполная запись курьера не считается завершённым обучением.
        results = run_parallel_EPOS(
            courier, polys, dynamic_config, **run_kwargs
        )
        global_policy, local_policy = get_EPOS_strategy(
            results, algorithms=cfg.algorithms
        )

        if len(local_policy) != len(polys):
            missing = [p for p in polys if p not in local_policy]
            raise RuntimeError(
                f"Courier {courier!r}: no local policy for "
                f"{len(missing)} polygons; examples: {missing[:5]}"
            )
        if not np.isclose(sum(global_policy), 1.0):
            raise RuntimeError(
                f"Courier {courier!r}: invalid global policy {global_policy}"
            )

        trained[courier] = {
            "base_lat": lat,
            "base_lon": lon,
            "polygon_list": polys,
            "EPOS_global_strat": global_policy,
            "EPOS_local_strat": local_policy,
        }
        n_saved += 1
        if n_saved % checkpoint_every == 0:
            _save_pickle(checkpoint_file, trained)

    _save_pickle(checkpoint_file, trained)
    return trained


def relabel_from_triplets(results: Dict[str, Dict[str, Any]], prefs_map: Dict[str, Dict[str, float]],
                          dynamic_config: Optional[Dict[str, Any]] = None,
                          cfg: Optional[TAPOCTFastConfig] = None,
                          pref_override: Optional[Dict[str, float]] = None,
                          spec_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Переразметка БЕЗ вызова солверов: сохранённые (T, CT) переоцениваются новым клиентом.
    Абляции Exp.2 (beta=1e-5 / gamma=1e-5), другие TW/CT_SA/цены, другие распределения
    предпочтений (задача 4) — за секунды вместо повторного 8–9-часового прогона.

    results       : {polygon: result с ключом 'triplets'}
    prefs_map     : {polygon: {'alpha','beta','gamma'}}
    dynamic_config: dict | None (None -> исходный spec из result)
    pref_override : dict | None, напр. {'beta': 1e-5}
    -> {polygon: {'policy','policy_omd','policy_prob_best','algorithm_wins','theta_hat','ell_hat','expected_gap'}}
    """
    cfg = cfg or TAPOCTFastConfig()
    out = {}
    for pid, r in results.items():
        if "triplets" not in r:
            continue
        sp = dict(r["spec"])
        if dynamic_config is not None:
            sp.update({"U": dynamic_config["U"], "price": dynamic_config["price"],
                       "CT_SA": dynamic_config["CT_SA"], "time_window": tuple(dynamic_config["time_window"])})
        if spec_kwargs:
            sp.update(spec_kwargs)
        spec = TenderSpec(**sp)
        prefs = dict(prefs_map[pid]); prefs.update(pref_override or {})
        oracle = ClientFeedbackOracle(prefs, spec)
        trip = []
        for t in r["triplets"]:
            t = dict(t); t["price"] = spec.price
            t["feedback"] = oracle(spec.price, t["route_cost"], t["ct"], t["valid_route"])
            _, feas = offer_features(spec.price, t["route_cost"], t["ct"], spec)
            t["feasible"] = bool(feas and t["valid_route"])
            trip.append(Triplet(**t))
        lab = label_from_triplets(trip, spec, cfg)
        out[pid] = {"policy": lab["policy"], "policy_omd": lab["policy_omd"],
                    "policy_prob_best": lab["policy_prob_best"],
                    "algorithm_wins": policy_to_counts(lab["policy"], cfg.counts_total),
                    "theta_hat": lab["theta_hat"], "ell_hat": lab["scores"]["ell"],
                    "expected_gap": lab["expected_gap"]}
    return out
