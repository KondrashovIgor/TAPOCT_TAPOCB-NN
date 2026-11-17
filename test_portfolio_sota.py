"""Small, runnable checks for pairing, group isolation and outcome accounting."""
import sys
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio_sota as ps


class DummyPolicy:
    scenario = ("spatial", "gaussian")

    def __init__(self, path):
        pass

    def predict(self, x):
        return np.array([0, 0, 1, 0], dtype=float)


class DummySelector:
    def __init__(self, k):
        self.k = k
        self.observed = []

    def select(self, x):
        return self.k

    def update(self, x, k, score):
        self.observed.append((k, score))


class UncopyableNeural(DummySelector):
    """Simulate NeuralSelector's unpickleable torch module reference."""
    def __init__(self, k):
        super().__init__(k)
        self.torch = math

    def export(self):
        return {'k': self.k, 'observed': list(self.observed)}

    @classmethod
    def restore(cls, state):
        obj = cls(state['k'])
        obj.observed = list(state['observed'])
        return obj


class TestPortfolioSota(unittest.TestCase):
    def setUp(self):
        self.map = {}
        self.prop = {}
        self.epos = {}
        self.prefs = {}
        for k in range(20):
            pid = f"AUG--G-{k}-2-3"
            self.map[pid] = {"augmented": False, "cluster": k % 4,
                             "distance matrix": np.array([[0, 10], [11, 0]]),
                             "depot row": [2, 3], "full map": [100, 101],
                             "distance provenance": {"matrix_unit": "s"}}
            self.prop[pid] = {"area_km2": 1, "centroid_lat": 55,
                              "centroid_lon": 37, "bounding_box": {"width_km": 1, "height_km": 1},
                              "relative_position": {"normalized_lat": .2,
                                                    "normalized_lon": .3,
                                                    "distance_from_center_km": k},
                              "num_points": 3}
            self.epos[pid] = {"spec": {"U": 900, "price": 500,
                                       "CT_SA": 5, "time_window": (0, 7200),
                                       "n_orders": 2},
                              "algorithms": list(ps.ALGORITHMS),
                              "triplets": [{"algorithm": name, "route_cost": 13,
                                            "ct": .01, "valid_route": True}
                                           for name in ps.ALGORITHMS for _ in range(2)]}
            self.prefs[pid] = {"alpha": .2, "beta": .4, "gamma": .4}

    def test_group_split_and_repetition(self):
        orig = next(iter(self.map))
        aug = "AUG--AUG-G-0-2-3-1"
        self.map[aug] = {**self.map[orig], "augmented": True}
        self.prop[aug] = self.prop[orig]
        ids = list(self.prop) + [aug]
        train, val, test = ps.grouped_partition(ids, self.map)
        sets = [{ps.original_polygon_id(pid, self.map) for pid in part}
                for part in (train, val, test)]
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))
        seq = ps.sequence_with_repetitions(test, self.prop, max_repeats=50)
        self.assertTrue(all(1 <= seq.count(pid) <= 50 for pid in set(test)))

    def test_logged_feedback_and_route(self):
        ids = list(self.prop)[:10]
        records = ps.make_logged_feedback(ids, self.epos, self.prefs)
        self.assertEqual(len(records), len(ids) * 8)
        self.assertTrue(all(0 <= reward <= 1 for _, _, reward in records))
        solvers = {name: (lambda orders, depot, matrix: (orders, 0))
                   for name in ps.ALGORITHMS}
        cost, seconds = ps.measure_route(ids[0], 2, aug_map=self.map, solvers=solvers)
        self.assertEqual(cost, 12)  # open route: depot->100 (2), 100->101 (10)
        self.assertGreaterEqual(seconds, 0)

    def test_missing_depot_row_uses_notebook_lookup(self):
        pid = next(iter(self.map))
        del self.map[pid]['depot row']
        requested = []
        def depot_lookup(ids):
            requested.append(tuple(ids))
            return [2, 3]
        rows = ps.precompute_depot_rows([pid, pid], self.map, depot_lookup)
        self.assertEqual(requested, [(100, 101)])
        solvers = {name: (lambda orders, depot, matrix: (orders, 0))
                   for name in ps.ALGORITHMS}
        ps.warmup_solver_bundle(solvers, pid, self.map, depot_rows=rows)
        cost, _ = ps.measure_route(pid, 0, aug_map=self.map,
                                   solvers=solvers, depot_rows=rows)
        self.assertEqual(cost, 12)
        with self.assertRaisesRegex(ValueError, 'missing OD'):
            ps.precompute_depot_rows([pid], self.map, lambda ids: [10**9, 3])

    def test_paired_run_and_aggregation(self):
        ids = list(self.prop)
        scaler = ps.build_context_scaler(ids[:10], self.prop, self.map)
        baselines = {'Regression': DummySelector(1),
                     **{name: UncopyableNeural(i % 4)
                        for i, name in enumerate(ps.COMPETITORS[3:])}}
        solvers = {name: (lambda orders, depot, matrix: (orders, 0))
                   for name in ps.ALGORITHMS}
        with patch.object(ps, "CourierPolicyNetwork", DummyPolicy), patch.object(
                ps, "NeuralSelector", UncopyableNeural):
            kwargs = dict(aug_map=self.map, properties=self.prop,
                          epos_results=self.epos, preference_map=self.prefs,
                          tapocb_checkpoint="unused", warmstart_selectors=baselines,
                          context_scaler=scaler, solvers=solvers, reuse_routes=True)
            raw = ps.run_scenario(ids[10:12] * 2, ("spatial", "gaussian"), **kwargs)
            again = ps.run_scenario(ids[10:12] * 2, ("spatial", "gaussian"), **kwargs)
        self.assertEqual(raw['arm'].tolist(), again['arm'].tolist())
        self.assertEqual(raw['route_cost_s'].tolist(), again['route_cost_s'].tolist())
        # Selection wall time differs between runs and legitimately enters CT.
        self.assertTrue(np.allclose(raw['client_reward'], again['client_reward'], atol=.01))
        self.assertTrue(all(not learner.observed for learner in baselines.values()))
        self.assertEqual(len(raw), 4 * 7)
        cluster, scenario = ps.summarize_sensitivity(raw)
        self.assertEqual(len(scenario), 1)
        self.assertIn("avg_delta_TAPOCB-NN_from_top_1_units", scenario)
        self.assertIn("delta_TAPOCB_from_best_share", cluster)
        self.assertAlmostEqual(scenario["TAPOCB-NN_win_rate"].iloc[0], .5, delta=.5)

    def test_rho_gamma_dirichlet_gate(self):
        class Inject:
            def random(self):
                return 0.0  # gate x < rho

            def dirichlet(self, alpha):
                self.alpha = alpha
                return np.array([.1, .2, .3, .4])

        rng = Inject()
        memory = ps.TAPOCBMemory(rho=.15)
        prior = np.full(4, .25)
        self.assertTrue(np.allclose(memory.get_policy("p", prior, rng), prior))
        memory.record_outcome("p", 2, 4., True)
        observed = memory.get_policy("p", prior, rng)
        state = memory.entries["p"]
        self.assertAlmostEqual(state["last_gamma"], .15 / 2)
        self.assertAlmostEqual(state["last_injection_weight"], .15 / 2)
        self.assertTrue(np.allclose(observed, .925 * state["policy_hat"] +
                                    .075 * np.array([.1, .2, .3, .4])))
        memory.record_outcome("p", 2, 1., False)
        observed = memory.get_policy("p", prior, rng)
        self.assertAlmostEqual(state["last_gamma"], .15 / 3)
        self.assertAlmostEqual(state["last_injection_weight"], .15 - .15 / 3)
        self.assertTrue(np.allclose(observed, .9 * state["policy_hat"] +
                                    .1 * np.array([.1, .2, .3, .4])))
        self.assertEqual(state["processed"], 2)

    def test_no_injection_and_masked_online_history(self):
        prior = np.full(4, .25)
        no_noise = ps.TAPOCBMemory(injection_enabled=False)
        rng = np.random.default_rng(42)
        no_noise.get_policy("p", prior, rng)
        no_noise.record_outcome("p", 1, 4., True)
        observed = no_noise.get_policy("p", prior, rng)
        self.assertTrue(np.allclose(observed, no_noise.entries["p"]["policy_hat"]))
        self.assertEqual(no_noise.entries["p"]["last_injection_weight"], 0.)
        for disabled in range(4):
            allowed = np.ones(4, dtype=bool)
            allowed[disabled] = False
            memory = ps.TAPOCBMemory(allowed_arms=allowed)
            rng = np.random.default_rng(123)
            for t in range(100):
                policy = memory.get_policy("p", prior, rng)
                self.assertEqual(policy[disabled], 0.)
                self.assertAlmostEqual(policy.sum(), 1.)
                arm = int(rng.choice(4, p=policy))
                self.assertNotEqual(arm, disabled)
                memory.record_outcome("p", arm, 4., t % 2 == 0)


if __name__ == "__main__":
    unittest.main()
