"""Small deterministic checks for the seven-agent market accounting."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import seven_agent_market as market


class FixedBidder:
    def __init__(self, action):
        self.action = action
        self.observed = []

    def act(self, state):
        return self.action

    def observe(self, state, action, reward):
        self.observed.append(reward)

    def learn(self):
        pass


def routes(n=5, seconds=1000.):
    rows = []
    for step in range(n):
        for method in market.METHODS:
            rows.append({"scenario": "spatial/gaussian", "step": step,
                         "polygon_id": "p", "method": method, "arm": "lkh3",
                         "route_cost_s": seconds, "total_ct_s": .1})
    return market.paired_route_trace(pd.DataFrame(rows), n_tenders=n)


class TestMarket(unittest.TestCase):
    def setUp(self):
        self.spec = {"p": {"spec": {"U": 900., "price": 500., "CT_SA": 5.,
                                   "time_window": (0., 7200.), "n_orders": 2}}}
        self.prefs = {"p": {"alpha": 1., "beta": 0., "gamma": 0.}}

    def test_guaranteed_base_payment_and_welfare_accounting(self):
        agents = {name: FixedBidder(0. if i == 0 else .5)
                  for i, name in enumerate(market.METHODS)}
        summary, total, curve, detail = market.simulate_market(
            routes(), self.prefs, self.spec, agents, base_reward_rate=.013)
        self.assertEqual(len(summary), 7)
        self.assertEqual(len(curve), 5)
        self.assertEqual(len(detail), 35)
        self.assertAlmostEqual(summary.win_pct_all_tenders.sum(), 100.)
        self.assertAlmostEqual(summary.loc[0, "mean_bid"], 450.)
        self.assertAlmostEqual(summary.loc[0, "mean_income_per_tender"],
                               450. * .013 + 450. - 10.)
        self.assertAlmostEqual(summary.loc[1, "mean_income_per_tender"], 675. * .013)
        self.assertAlmostEqual(summary.mean_social_welfare_contribution.sum(),
                               total.mean_social_welfare.iloc[0])
        self.assertAlmostEqual(total.mean_social_welfare.iloc[0], 450.)
        self.assertAlmostEqual(total.mean_client_satisfaction.iloc[0], .5)
        self.assertAlmostEqual(summary.mean_client_satisfaction_contribution.sum(),
                               total.mean_client_satisfaction.iloc[0])
        self.assertTrue(np.allclose(curve.iloc[-1, 1:].to_numpy(float),
                                    summary.mean_income_per_tender))
        self.assertTrue(all(len(agent.observed) == 5 for agent in agents.values()))

    def test_all_infeasible_means_no_allocation(self):
        agents = {name: FixedBidder(.5) for name in market.METHODS}
        summary, total, _, _ = market.simulate_market(
            routes(2, seconds=8000.), self.prefs, self.spec, agents)
        self.assertEqual(summary.win_pct_all_tenders.sum(), 0.)
        self.assertEqual(total.mean_social_welfare.iloc[0], 0.)
        self.assertEqual(total.mean_client_satisfaction.iloc[0], 0.)
        self.assertTrue((summary.mean_income_per_tender > 0).all())

    def test_reject_incomplete_paired_trace(self):
        r = routes(2).reset_index()
        with self.assertRaisesRegex(ValueError, "exactly one offer"):
            market.paired_route_trace(r.iloc[:-1], n_tenders=2)

    def test_eight_scenario_aggregation_and_log_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        scenario_names = [f'spatial/{i}' for i in range(4)] + [f'independent/{i}' for i in range(4)]
        summaries = pd.DataFrame([{
            'scenario': scenario, 'method': method, 'mean_bid': 500. + k,
            'win_pct_all_tenders': 100. / 7, 'mean_income_per_tender': 10. + k,
            'mean_base_payment_per_tender': 6.5,
            'mean_client_satisfaction_contribution': .5 / 7,
            'mean_social_welfare_contribution': 400. / 7,
        } for k, scenario in enumerate(scenario_names) for method in market.METHODS])
        totals = pd.DataFrame({'scenario': scenario_names})
        running = pd.DataFrame({'iteration': np.arange(1, 10_001),
                                **{method: np.full(10_000, 11.) for method in market.METHODS}})
        with TemporaryDirectory() as temp:
            means, fig = market.save_eight_scenario_report(summaries, totals, running, temp)
            self.assertEqual(len(means), 7)
            self.assertEqual(len(fig.axes[0].lines), 7)
            self.assertEqual(fig.axes[0].get_title(), '')
            self.assertAlmostEqual(fig.axes[0].lines[0].get_ydata()[0], np.log(11.))
            self.assertAlmostEqual(means.loc[0, 'mean_income_per_tender'], 13.5)
            self.assertEqual(len(pd.read_csv(Path(temp) / 'seven_agents_eight_scenarios.csv')), 56)
            self.assertEqual(len(pd.read_csv(Path(temp) / 'seven_agents_scenario_means.csv')), 7)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
