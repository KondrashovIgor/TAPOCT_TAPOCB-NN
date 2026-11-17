"""Sensitivity analysis using cached TAPOCT (route time, compute time) triplets.

Execute in the notebook after cell 21, or import these functions from this file.
This is fixed-triplet replay; no routing solver or OSRM request is made.
"""

import hashlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

import tapoct_fast as tf


AXES = ("alpha", "beta", "gamma")
SCENARIOS = ("gaussian", "beta", "dirichlet", "ordinal_posterior")
SPATIAL_MODES = ("independent", "spatial")
BASELINE = ("independent", "gaussian")


def _rng(seed, scenario, polygon):
    payload = f"{seed}|{scenario}|{polygon}".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def cluster_parameters(aug_map, properties, polygon_ids,
                       city_center=(55.752, 37.6177)):
    """Map arbitrary cluster IDs to unique, spatially ordered prior parameters."""
    lat0, lon0 = np.deg2rad(city_center)
    distances = {}
    for pid in polygon_ids:
        cluster = aug_map[pid]["cluster"]
        p = properties[pid]
        lat, lon = np.deg2rad((p["centroid_lat"], p["centroid_lon"]))
        a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
        km = 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        distances.setdefault(cluster, []).append(float(km))
    ordered = sorted(distances, key=lambda c: (np.median(distances[c]), str(c)))
    params = {}
    for rank, cluster in enumerate(ordered):
        r = rank / max(1, len(ordered) - 1)
        # Near center: callback (gamma); far: price (alpha) and delivery time (beta).
        mean = np.array([0.20 + 0.27*r, 0.25 + 0.13*r, 0.55 - 0.40*r])
        params[cluster] = {
            "mean": mean,
            "median_radius_km": float(np.median(distances[cluster])),
            "spatial_rank": rank,
            "gaussian_sd": 0.11 + 0.012*rank,
            "beta_concentration": 12.0 + 2.0*rank,
            "dirichlet_concentration": 10.0 + 2.0*rank,
            "ordinal_prior_concentration": 7.0 + 1.5*rank,
            "outlier_probability": 0.04 + 0.01*rank,
        }
    return params


def _simplex(x):
    x = np.maximum(np.asarray(x, dtype=float), 1e-8)
    return x / x.sum()


def _observed_full_offer_features(result, spec, max_offers=4):
    """One feasible offer per solver; all three offer attributes enter the rating."""
    seen, rows = set(), []
    for t in result["triplets"]:
        if t["algorithm"] in seen or not t["valid_route"]:
            continue
        x, feasible = tf.offer_features(spec.price, t["route_cost"], t["ct"], spec)
        if feasible and np.isfinite(x).all():
            rows.append(x)
            seen.add(t["algorithm"])
        if len(rows) >= max_offers:
            break
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _ordinal_posterior(rng, mu, par, result, n_particles=1024, rating_noise=0.08):
    """Simulated posterior predictive weights given 1–5 complete-offer ratings."""
    prior = np.maximum(par["ordinal_prior_concentration"] * mu, 1e-4)
    if rng.random() < par["outlier_probability"]:
        latent = rng.dirichlet(np.full(3, 0.40))
    else:
        latent = rng.dirichlet(prior)
    spec = tf.TenderSpec(**result["spec"])
    X = _observed_full_offer_features(result, spec)
    if len(X) == 0:
        return _simplex(rng.dirichlet(prior)), 0
    noisy = np.clip(X @ latent + rng.normal(0, rating_noise, len(X)), 0, 1)
    ratings = np.minimum(5, 1 + np.floor(5 * noisy).astype(int))
    particles = rng.dirichlet(prior, size=n_particles)
    predictions = np.clip(particles @ X.T, 0, 1)
    lower = np.where(ratings == 1, -np.inf, (ratings - 1) / 5)
    upper = np.where(ratings == 5, np.inf, ratings / 5)
    likelihood = ndtr((upper - predictions) / rating_noise) - ndtr((lower - predictions) / rating_noise)
    logp = np.log(np.maximum(likelihood, 1e-300)).sum(axis=1)
    weights = np.exp(logp - logp.max())
    weights /= weights.sum()
    posterior_sample = particles[rng.choice(n_particles, p=weights)]
    return _simplex(posterior_sample), len(ratings)


def preference_maps(replay_results, aug_map, properties, seed=42,
                    city_center=(55.752, 37.6177), gaussian_reference=None):
    """Return two stationary spatial modes × four distributions.

    Independent mode samples a persistent weight vector per polygon/client, with
    identical distribution parameters in every cluster. The overall mean and
    dispersion parameters match polygon-weighted spatial parameter averages.
    """
    polygon_ids = list(replay_results)
    params = cluster_parameters(aug_map, properties, polygon_ids, city_center)
    counts = {c: sum(aug_map[pid]["cluster"] == c for pid in polygon_ids) for c in params}
    independent = {}
    for name in ("mean", "gaussian_sd", "beta_concentration", "dirichlet_concentration",
                 "ordinal_prior_concentration", "outlier_probability"):
        independent[name] = np.average([params[c][name] for c in params],
                                       axis=0, weights=[counts[c] for c in params])
    maps = {mode: {scenario: {} for scenario in SCENARIOS} for mode in SPATIAL_MODES}
    diagnostic = {mode: {scenario: {} for scenario in SCENARIOS} for mode in SPATIAL_MODES}
    for pid, result in replay_results.items():
        for mode in SPATIAL_MODES:
            par = params[aug_map[pid]["cluster"]] if mode == "spatial" else independent
            mu = par["mean"]
            for scenario in SCENARIOS:
                rng = _rng(seed, mode + "|" + scenario, pid)
                if scenario == "gaussian":
                    weights = _simplex(rng.normal(mu, par["gaussian_sd"], 3))
                elif scenario == "beta":
                    k = par["beta_concentration"]
                    weights = _simplex(rng.beta(k*mu, k*(1-mu)))
                elif scenario == "dirichlet":
                    weights = rng.dirichlet(par["dirichlet_concentration"]*mu)
                else:
                    weights, n_ratings = _ordinal_posterior(rng, mu, par, result)
                    diagnostic[mode][scenario][pid] = {"synthetic_offer_ratings": n_ratings}
                maps[mode][scenario][pid] = dict(zip(AXES, map(float, weights)))
    if gaussian_reference is not None:
        missing = [pid for pid in polygon_ids if pid not in gaussian_reference]
        if missing:
            raise KeyError(f"Gaussian reference missing {len(missing)} polygons: {missing[:5]}")
        original = {}
        for pid in polygon_ids:
            values = np.array([gaussian_reference[pid][a] for a in AXES], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
                raise ValueError(f"Invalid reference preferences for {pid}")
            original[pid] = dict(zip(AXES, map(float, values)))
        maps["spatial"]["gaussian"] = original
        shuffled = np.random.default_rng(seed).permutation(polygon_ids)
        maps["independent"]["gaussian"] = {
            pid: dict(original[source]) for pid, source in zip(polygon_ids, shuffled)
        }
    return maps, {"spatial": params, "independent": independent}, diagnostic


def common_triplet_replay(raw_results, algorithms, calls_per_algorithm=2):
    """Select the same first two measured calls for each solver in every scenario."""
    replay = {}
    for pid, result in raw_results.items():
        if "triplets" not in result or "spec" not in result:
            continue
        if tuple(result.get("algorithms", ())) != tuple(algorithms):
            raise ValueError(f"{pid}: cached solver order differs from cfg.algorithms")
        used = {name: 0 for name in algorithms}
        triplets = []
        for t in result["triplets"]:
            name = t["algorithm"]
            if name in used and used[name] < calls_per_algorithm:
                triplets.append(t)
                used[name] += 1
        if any(count != calls_per_algorithm for count in used.values()):
            raise ValueError(f"{pid}: fewer than {calls_per_algorithm} calls per solver: {used}")
        replay[pid] = {**result, "triplets": triplets}
    if not replay:
        raise ValueError("No completed TAPOCT results with triplets and spec were found")
    return replay


def _relabel_five_point(replay, prefs, cfg):
    """Recalculate TAPOCT using 1–5 overall-offer ratings, scaled back to [0,1]."""
    output = {}
    for pid, result in replay.items():
        spec = tf.TenderSpec(**result["spec"])
        oracle = tf.ClientFeedbackOracle(prefs[pid], spec)
        triplets = []
        for old in result["triplets"]:
            t = dict(old)
            x, feasible = tf.offer_features(spec.price, t["route_cost"], t["ct"], spec)
            if t["valid_route"] and feasible:
                continuous = oracle(spec.price, t["route_cost"], t["ct"], True)
                rating = min(5, 1 + int(np.floor(5*np.clip(continuous, 0, 1))))
                t["feedback"] = (rating - 1) / 4
            else:
                t["feedback"] = 0.0
            t["price"] = spec.price
            t["feasible"] = bool(feasible and t["valid_route"])
            triplets.append(tf.Triplet(**t))
        label = tf.label_from_triplets(triplets, spec, cfg)
        output[pid] = {
            "policy": label["policy"], "policy_omd": label["policy_omd"],
            "policy_prob_best": label["policy_prob_best"],
            "theta_hat": label["theta_hat"],
            "ell_hat": label["scores"]["ell"],
            "expected_gap": label["expected_gap"],
        }
    return output


def _score_on_fixed_triplets(replay, preferences, policies, algorithms, scenario):
    """Expected simulated client score and regret on the common measured offers."""
    out = {}
    for pid, result in replay.items():
        spec = tf.TenderSpec(**result["spec"])
        oracle = tf.ClientFeedbackOracle(preferences[pid], spec)
        rewards = {a: [] for a in algorithms}
        for t in result["triplets"]:
            score = oracle(spec.price, t["route_cost"], t["ct"], t["valid_route"])
            if scenario == "ordinal_posterior" and t["valid_route"]:
                rating = min(5, 1 + int(np.floor(5*np.clip(score, 0, 1))))
                score = (rating - 1) / 4
            rewards[t["algorithm"]].append(score)
        means = np.array([np.mean(rewards[a]) for a in algorithms], dtype=float)
        policy = np.asarray(policies[pid]["policy"], dtype=float)
        expected = float(policy @ means)
        static = float(means[list(algorithms).index("lkh3")])
        out[pid] = {"expected_score": expected,
                    "oracle_regret": float(means.max() - expected),
                    "gain_vs_lkh3": expected - static}
    return out


def _summary(preferences, policies, rewards, aug_map, algorithms, mode, scenario, seed):
    rows = []
    clusters = sorted({aug_map[pid]["cluster"] for pid in preferences}, key=str)
    for cluster in clusters:
        ids = [p for p in preferences if aug_map[p]["cluster"] == cluster]
        w = np.array([[preferences[p][key] for key in AXES] for p in ids])
        pi = np.array([policies[p]["policy"] for p in ids], dtype=float)
        if pi.shape != (len(ids), len(algorithms)) or not np.isfinite(pi).all():
            raise ValueError(f"Invalid TAPOCT policy in {scenario}, cluster {cluster}")
        row = {"spatial_mode": mode, "distribution": scenario, "seed": seed, "cluster": cluster,
               "n_polygons": len(ids)}
        for j, key in enumerate(AXES):
            row[f"{key}_median"] = float(np.median(w[:, j]))
            row[f"{key}_variance"] = float(np.var(w[:, j], ddof=1)) if len(ids)>1 else np.nan
        for j, name in enumerate(algorithms):
            row[f"p_{name}"] = float(pi[:, j].mean())
        for metric in ("expected_score", "oracle_regret", "gain_vs_lkh3"):
            row[metric] = float(np.mean([rewards[p][metric] for p in ids]))
        rows.append(row)
    return rows


def run_distribution_robustness(raw_results, aug_map, properties, cfg,
                                seeds=(41, 42, 43),
                                city_center=(55.752, 37.6177),
                                gaussian_reference=None):
    """Replay eight stationary preference scenarios on identical cached solver calls."""
    replay = common_triplet_replay(raw_results, cfg.algorithms)
    missing = [p for p in replay if p not in aug_map or p not in properties or "cluster" not in aug_map[p]]
    if missing:
        raise KeyError(f"Missing geometry/properties/cluster for {len(missing)} polygons: {missing[:5]}")
    maps, labels, diagnostic, rows = {}, {}, {}, []
    for seed in seeds:
        maps[seed], cluster_par, diagnostic[seed] = preference_maps(
            replay, aug_map, properties, seed=seed, city_center=city_center,
            gaussian_reference=gaussian_reference)
        labels[seed] = {mode: {} for mode in SPATIAL_MODES}
        for mode in SPATIAL_MODES:
            for scenario in SCENARIOS:
                pref = maps[seed][mode][scenario]
                if scenario == "ordinal_posterior":
                    result = _relabel_five_point(replay, pref, cfg)
                else:
                    result = tf.relabel_from_triplets(replay, pref, cfg=cfg)
                if len(result) != len(replay):
                    raise RuntimeError(f"Incomplete replay for {mode}/{scenario}, seed={seed}")
                labels[seed][mode][scenario] = result
                rewards = _score_on_fixed_triplets(replay, pref, result, cfg.algorithms, scenario)
                rows.extend(_summary(pref, result, rewards, aug_map,
                                     cfg.algorithms, mode, scenario, seed))
    return maps, labels, pd.DataFrame(rows), cluster_par, diagnostic


def cluster_correlation_matrices(summary, algorithms, threshold=0.05):
    """Pearson r between cluster preference summaries and portfolio probabilities.

    Each scenario uses the same cluster × seed observations. Full table contains
    absolute r; highlights contain only Δr versus independent/Gaussian baseline.
    0.05 denotes an absolute change in r, not five percent of baseline r.
    Undefined correlations (constant columns) remain NaN; they are never imputed.
    """
    if not 0 <= threshold <= 2:
        raise ValueError("threshold must be in [0, 2] Pearson-r units")
    preference_columns = [f"{a}_{s}" for a in AXES for s in ("median", "variance")]
    policy_columns = [f"p_{name}" for name in algorithms]
    expected = {(m, d) for m in SPATIAL_MODES for d in SCENARIOS}
    observed = set(zip(summary.spatial_mode, summary.distribution))
    if observed != expected:
        raise ValueError(f"Missing or unexpected scenarios: {expected ^ observed}")
    # Pair corresponding cluster × seed rows to prevent unequal sample bases.
    reference_index = None
    absolute = {}
    for mode in SPATIAL_MODES:
        for scenario in SCENARIOS:
            part = summary[(summary.spatial_mode == mode) &
                           (summary.distribution == scenario)].set_index(["cluster", "seed"])
            if part.index.has_duplicates:
                raise ValueError(f"Duplicate cluster × seed rows in {mode}/{scenario}")
            if reference_index is None:
                reference_index = part.index
            if set(part.index) != set(reference_index):
                raise ValueError(f"Cluster × seed coverage differs in {mode}/{scenario}")
            part = part.reindex(reference_index)
            absolute[(mode, scenario)] = {
                f"{weight} : {policy}": (
                    part[weight].corr(part[policy], method="pearson")
                    if part[weight].dropna().nunique() > 1 and part[policy].dropna().nunique() > 1
                    else np.nan)
                for weight in preference_columns for policy in policy_columns
            }
    full = pd.DataFrame.from_dict(absolute, orient="index")
    full.index = pd.MultiIndex.from_tuples(full.index, names=["spatial_mode", "distribution"])
    base = full.loc[BASELINE]
    delta = full.subtract(base, axis="columns")
    delta.loc[BASELINE] = 0.0
    changed = (delta.abs() > threshold).any(axis=0)
    keep_rows = (delta.abs() > threshold).any(axis=1)
    keep_rows.loc[BASELINE] = True
    # Main-text heatmap: one column per algorithm, maximum |Δr| across six
    # preference statistics; retain the responsible statistic as an annotation.
    highlights = pd.DataFrame(index=delta.loc[keep_rows].index, columns=policy_columns,
                              dtype=float)
    feature = pd.DataFrame(index=highlights.index, columns=policy_columns, dtype=object)
    highlights.loc[BASELINE] = 0.0
    for idx in highlights.index:
        if idx == BASELINE:
            continue
        for policy in policy_columns:
            candidates = [f"{weight} : {policy}" for weight in preference_columns]
            valid = delta.loc[idx, candidates].dropna()
            if not valid.empty:
                strongest = valid.abs().idxmax()
                if abs(valid[strongest]) > threshold:
                    highlights.loc[idx, policy] = valid[strongest]
                    feature.loc[idx, policy] = strongest.split(" : ")[0]
    # A row can be retained due to a differing pair even if its probability
    # column is undefined in every scenario; keep only genuinely changed rows.
    highlights = highlights.loc[highlights.notna().any(axis=1)]
    feature = feature.reindex(highlights.index)
    full_delta = delta.loc[:, changed]
    return {"full": full, "full_delta": full_delta,
            "highlights": highlights, "highlight_feature": feature, "baseline": base,
            "threshold": threshold, "n_cluster_seed_rows": len(reference_index)}


def scenario_performance(summary):
    """Polygon-weighted means; repeated seeds contribute equally."""
    metrics = ("expected_score", "oracle_regret", "gain_vs_lkh3")
    rows = []
    for (mode, scenario), part in summary.groupby(["spatial_mode", "distribution"], sort=False):
        row = {"spatial_mode": mode, "distribution": scenario,
               "n_polygon_evaluations": int(part["n_polygons"].sum())}
        for metric in metrics:
            row[metric] = float(np.average(part[metric], weights=part["n_polygons"]))
        rows.append(row)
    return pd.DataFrame(rows).set_index(["spatial_mode", "distribution"])


def plot_correlation_matrices(matrices):
    """One compact ΔPearson heatmap; returns the Figure for manuscript export."""
    matrix = matrices["highlights"]
    labels = [f"{m} / {d}" + (" (baseline)" if (m, d) == BASELINE else "")
              for m, d in matrix.index]
    width = max(9, min(22, 0.52*len(matrix.columns) + 5))
    height = max(2.7, 0.55*len(matrix) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, vmin=-2, vmax=2, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=58, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"Change in Pearson r vs independent/Gaussian (|Δr| > {matrices['threshold']:.2f})")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                feature = matrices["highlight_feature"].iloc[i, j]
                caption = f"{values[i, j]:+.2f}"
                if isinstance(feature, str):
                    caption += f"\n{feature.replace('_median', ' med').replace('_variance', ' var')}"
                ax.text(j, i, caption, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Δr (absolute Pearson-r units)")
    fig.tight_layout()
    plt.show()
    return fig


def plot_cluster_policies(summary, algorithms):
    """Compare cluster policy allocations across all eight scenarios."""
    fig, axs = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for ax, (mode, scenario) in zip(axs.flat,
                                    ((m, d) for m in SPATIAL_MODES for d in SCENARIOS)):
        frame = summary[(summary.distribution == scenario) &
                        (summary.spatial_mode == mode)].groupby("cluster", sort=True)[
            [f"p_{a}" for a in algorithms]].mean()
        bottom = np.zeros(len(frame))
        for name in algorithms:
            values = frame[f"p_{name}"].to_numpy()
            ax.bar(frame.index.astype(str), values, bottom=bottom, label=name)
            bottom += values
        ax.set(title=f"{mode} / {scenario}", xlabel="cluster",
               ylabel="mean selection probability", ylim=(0, 1))
    axs.flat[0].legend(loc="upper left", bbox_to_anchor=(0, 1.30), ncol=4)
    plt.show()
    return fig
