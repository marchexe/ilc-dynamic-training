import unittest

from tests.test_pbt_artifacts import fixed_curve_metrics, synthetic_manifest
from training.pbt.reporting.constants import TOTAL_SCORE_COLUMN
from training.pbt.reporting.research_plots import (
    build_generation_decision_rows,
    build_member_metric_rows,
    generation_winner_member,
    shared_lr_center_series,
    validate_metric_rows,
)
from training.runtime import combine_group_scores


def _anchor_copy_manifest():
    """A small, fully hand-specified anchor_copy_lr_recenter manifest: 3
    generations, 2 members, one occurrence of each decision outcome, and
    known LR/metric values throughout -- so numeric-verification tests can
    assert exact expected results rather than just "did not crash". Shared
    with tests/test_pbt_report_plots.py."""
    manifest = synthetic_manifest()
    manifest["config"]["pbt"]["strategy"] = "anchor_copy_lr_recenter"
    manifest["config"]["pbt"]["min_lr"] = 3.0e-6
    manifest["config"]["pbt"]["max_lr"] = 1.4e-5
    manifest["members"] = {
        "m_a": {"name": "m_a", "lr": 1.0e-5, "parent": None},
        "m_b": {"name": "m_b", "lr": 8.0e-6, "parent": None},
    }
    manifest["generations"] = [
        {
            "index": 0,
            "epoch": 20,
            "status": "completed",
            "workers": {
                "m_a": {"status": "completed", "lr": 1.0e-5, "metrics": fixed_curve_metrics(0.9, 0.9, 0.97, 0.3)},
                "m_b": {"status": "completed", "lr": 8.0e-6, "metrics": fixed_curve_metrics(1.1, 0.88, 0.96, 0.32)},
            },
            "ranking": ["m_a", "m_b"],
            "exploit": [
                {"source": "anchor_copy_lr_recenter", "recipient": "m_a", "donor": "m_a", "recipient_lr": 1.0e-5, "new_lr": 1.0e-5, "applied": True},
                {"source": "anchor_copy_lr_recenter", "recipient": "m_b", "donor": "m_a", "recipient_lr": 8.0e-6, "new_lr": 9.0e-6, "applied": True},
            ],
            "anchor_copy_lr_recenter": {
                "decision": "accepted_new_anchor",
                "winner": "m_a",
                "winner_lr": 1.0e-5,
                "previous_lr_center": 1.0e-5,
                "new_lr_center": 1.0e-5,
                "assigned_lrs": {"m_a": 1.0e-5, "m_b": 9.0e-6},
                "spread_collapsed": False,
            },
        },
        {
            "index": 1,
            "epoch": 21,
            "status": "completed",
            "workers": {
                "m_a": {"status": "completed", "lr": 1.0e-5, "metrics": fixed_curve_metrics(0.95, 0.9, 0.97, 0.3)},
                "m_b": {"status": "completed", "lr": 9.0e-6, "metrics": fixed_curve_metrics(0.85, 0.9, 0.98, 0.29)},
            },
            "ranking": ["m_b", "m_a"],
            "exploit": [
                {"source": "anchor_copy_lr_recenter", "recipient": "m_a", "donor": "m_b", "recipient_lr": 1.0e-5, "new_lr": 3.0e-6, "applied": True},
                {"source": "anchor_copy_lr_recenter", "recipient": "m_b", "donor": "m_b", "recipient_lr": 9.0e-6, "new_lr": 1.4e-5, "applied": True},
            ],
            "anchor_copy_lr_recenter": {
                "decision": "reused_previous_anchor",
                "winner": "m_b",
                "winner_lr": 9.0e-6,
                "previous_lr_center": 1.0e-5,
                "new_lr_center": 9.0e-6,
                "assigned_lrs": {"m_a": 3.0e-6, "m_b": 1.4e-5},  # deliberately at the min/max bounds
                "spread_collapsed": True,
            },
        },
        {
            "index": 2,
            "epoch": 22,
            "status": "completed",
            "workers": {
                "m_a": {"status": "completed", "lr": 3.0e-6, "metrics": fixed_curve_metrics(1.3, 0.8, 0.9, 0.4)},
                "m_b": {"status": "completed", "lr": 1.4e-5, "metrics": fixed_curve_metrics(1.4, 0.79, 0.89, 0.41)},
            },
            "ranking": ["m_a", "m_b"],
            "exploit": [
                {"source": "anchor_copy_lr_recenter", "recipient": "m_a", "donor": "m_b", "recipient_lr": 3.0e-6, "new_lr": 8.1e-6, "applied": True},
                {"source": "anchor_copy_lr_recenter", "recipient": "m_b", "donor": "m_b", "recipient_lr": 1.4e-5, "new_lr": 9.9e-6, "applied": True},
            ],
            "anchor_copy_lr_recenter": {
                "decision": "rewound_to_previous_anchor",
                "winner": "m_a",
                "winner_lr": 3.0e-6,
                "previous_lr_center": 9.0e-6,
                "new_lr_center": 9.0e-6,
                "assigned_lrs": {"m_a": 8.1e-6, "m_b": 9.9e-6},
                "spread_collapsed": False,
            },
        },
    ]
    return manifest


def _confidence_aware_manifest():
    """A single-generation manifest where generation["ranking"] deliberately
    disagrees with a plain sort by raw optimization_metric_value --
    simulating confidence-aware incumbent persistence (m_a keeps winning
    despite m_b's numerically better raw value this generation)."""
    return {
        "config": {
            "shared": {"samples_per_epoch": 100, "epochs_per_generation": 1},
            "pbt": {"metric": "validation_working_point_mistag_percent", "mode": "min"},
        },
        "initial_evaluation": {"status": "skipped", "metrics": {}},
        "members": {
            "m_a": {"name": "m_a", "lr": 1.0e-4, "parent": None},
            "m_b": {"name": "m_b", "lr": 2.0e-4, "parent": None},
        },
        "generations": [
            {
                "index": 0,
                "status": "completed",
                "workers": {
                    "m_a": {"status": "completed", "lr": 1.0e-4, "metrics": {"validation_working_point_mistag_percent": 1.0}},
                    "m_b": {"status": "completed", "lr": 2.0e-4, "metrics": {"validation_working_point_mistag_percent": 0.5}},
                },
                "ranking": ["m_a", "m_b"],
            }
        ],
    }


def _old_max_mode_manifest():
    """A run whose configured metric is a max-mode HEP score, not
    total_mistag_score -- and where the true winner (by that real metric,
    via the recorded ranking) has a numerically WORSE total_mistag_score
    than the runner-up. A winner-selection implementation that (incorrectly)
    minimized total_mistag_score would pick the wrong member here."""
    strong_rejections = {
        "bc": [800.0, 500.0, 40.0], "bd": [900.0, 700.0, 120.0],
        "cb": [250.0, 80.0, 20.0], "cd": [900.0, 120.0, 12.0],
    }
    weak_rejections = {
        "bc": [20.0, 15.0, 10.0], "bd": [25.0, 18.0, 12.0],
        "cb": [15.0, 10.0, 8.0], "cd": [20.0, 14.0, 9.0],
    }

    def metrics_for(rejections, bkg_rejection_score):
        return {
            "validation_bkg_rejection_score": bkg_rejection_score,
            "validation_bkg_rejection_at_eff": {"efficiencies": [0.5, 0.8, 0.9], "pairs": rejections},
        }

    return {
        "config": {
            "shared": {"samples_per_epoch": 100, "epochs_per_generation": 1},
            "pbt": {"metric": "validation_bkg_rejection_score", "mode": "max"},
        },
        "initial_evaluation": {"status": "skipped", "metrics": {}},
        "members": {
            "trial_a": {"name": "trial_a", "lr": 1.0e-4, "parent": None},
            "trial_b": {"name": "trial_b", "lr": 2.0e-4, "parent": None},
        },
        "generations": [
            {
                "index": 0,
                "status": "completed",
                "workers": {
                    "trial_a": {"status": "completed", "lr": 1.0e-4, "metrics": metrics_for(weak_rejections, 9.0)},
                    "trial_b": {"status": "completed", "lr": 2.0e-4, "metrics": metrics_for(strong_rejections, 5.0)},
                },
                "ranking": ["trial_a", "trial_b"],
            }
        ],
    }


class DataLayerTest(unittest.TestCase):
    def test_member_rows_flag_the_actual_configured_metric_winner(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        generation_0 = [row for row in rows if row["generation"] == 0]
        winners = [row for row in generation_0 if row["is_winner"]]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["trial"], "m_a")  # 0.9 < 1.1, mode=min

    def test_member_rows_winner_matches_confidence_aware_ranking_not_raw_value(self):
        manifest = _confidence_aware_manifest()
        rows = build_member_metric_rows(manifest)
        winners = [row for row in rows if row["is_winner"]]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["trial"], "m_a")
        raw_winner = min(rows, key=lambda row: row["optimization_metric_value"])
        self.assertEqual(raw_winner["trial"], "m_b")

    def test_old_max_mode_manifest_winner_is_not_total_mistag_score_derived(self):
        manifest = _old_max_mode_manifest()
        rows = build_member_metric_rows(manifest)
        winners = [row for row in rows if row["is_winner"]]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["trial"], "trial_a")
        total_score_winner = min(rows, key=lambda row: row[TOTAL_SCORE_COLUMN])
        self.assertEqual(total_score_winner["trial"], "trial_b")

    def test_decision_rows_empty_for_non_anchor_copy_strategy(self):
        manifest = synthetic_manifest()  # strategy=anchored_lr_sweep
        rows = build_member_metric_rows(manifest)
        self.assertEqual(build_generation_decision_rows(manifest, rows), [])

    def test_decision_rows_carry_the_three_decision_types(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        self.assertEqual([d["decision"] for d in decisions], ["accepted_new_anchor", "reused_previous_anchor", "rewound_to_previous_anchor"])

    def test_anchor_row_carries_forward_unchanged_across_reuse_and_rewind(self):
        # gen0 accepts m_a as anchor; gen1/gen2 never accept again, so the
        # anchor's *own* row must stay gen0's m_a row throughout, even
        # though a different member wins gen1/gen2.
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        anchor_rows = [d["anchor_row"] for d in decisions]
        self.assertTrue(all(row is anchor_rows[0] for row in anchor_rows))
        self.assertEqual(anchor_rows[0]["trial"], "m_a")
        self.assertEqual(anchor_rows[0]["generation"], 0)

    def test_anchor_score_before_decision_is_none_for_the_first_ever_decision(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        self.assertIsNone(decisions[0]["anchor_total_score_before_decision"])
        # gen1/gen2 compare against gen0's now-established anchor score.
        self.assertIsNotNone(decisions[1]["anchor_total_score_before_decision"])
        self.assertAlmostEqual(decisions[1]["anchor_total_score_before_decision"], decisions[0]["anchor_row"][TOTAL_SCORE_COLUMN])

    def test_validate_metric_rows_keeps_a_genuine_zero(self):
        rows = [{"trial": "m", "generation": 0, "x": 0.0}]
        valid, warnings = validate_metric_rows(rows, ["x"])
        self.assertEqual(valid, rows)
        self.assertEqual(warnings, [])

    def test_validate_metric_rows_rejects_negative_missing_and_non_finite(self):
        rows = [
            {"trial": "a", "generation": 0, "x": -1.0},
            {"trial": "b", "generation": 0, "x": None},
            {"trial": "c", "generation": 0, "x": float("nan")},
            {"trial": "d", "generation": 0, "x": float("inf")},
            {"trial": "e", "generation": 0, "x": 5.0},
        ]
        valid, warnings = validate_metric_rows(rows, ["x"])
        self.assertEqual([row["trial"] for row in valid], ["e"])
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("member=a" in warning and "negative" in warning for warning in warnings))
        self.assertTrue(any("member=b" in warning and "missing" in warning for warning in warnings))
        self.assertTrue(any("member=c" in warning and "non-finite" in warning for warning in warnings))
        self.assertTrue(any("member=d" in warning and "non-finite" in warning for warning in warnings))

    def test_total_score_equals_sqrt_ctag_times_btag_for_every_row(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        from training.pbt.reporting.constants import BTAG_SCORE_COLUMN, CTAG_SCORE_COLUMN

        for row in rows:
            self.assertAlmostEqual(row[TOTAL_SCORE_COLUMN], combine_group_scores(row[CTAG_SCORE_COLUMN], row[BTAG_SCORE_COLUMN]))


class GenerationWinnerMemberTest(unittest.TestCase):
    def test_returns_ranking_zero(self):
        manifest = _confidence_aware_manifest()
        self.assertEqual(generation_winner_member(manifest, 0), "m_a")

    def test_falls_back_to_sorted_metric_when_no_ranking(self):
        manifest = _confidence_aware_manifest()
        del manifest["generations"][0]["ranking"]
        # mode=min -> lower raw value wins: m_b (0.5) beats m_a (1.0).
        self.assertEqual(generation_winner_member(manifest, 0), "m_b")

    def test_none_for_missing_generation(self):
        manifest = _confidence_aware_manifest()
        self.assertIsNone(generation_winner_member(manifest, 5))


class SharedLrCenterSeriesTest(unittest.TestCase):
    def test_empty_for_non_center_strategies(self):
        manifest = _anchor_copy_manifest()
        manifest["config"]["pbt"]["strategy"] = "exploit_mutate"
        self.assertEqual(shared_lr_center_series(manifest), [])

    def test_reads_anchor_copy_lr_recenter_field(self):
        manifest = _anchor_copy_manifest()
        expected = [(generation["index"], generation["anchor_copy_lr_recenter"]["new_lr_center"]) for generation in manifest["generations"]]
        self.assertEqual(shared_lr_center_series(manifest), expected)

    def test_reads_anchored_lr_sweep_field(self):
        manifest = synthetic_manifest()  # strategy=anchored_lr_sweep by default
        manifest["generations"][0]["exploit"] = [
            {"source": "anchored_lr_sweep", "recipient": "trial_a", "donor": "trial_a", "new_lr": 1.0e-4, "lr_center": 1.2e-4},
            {"source": "anchored_lr_sweep", "recipient": "trial_b", "donor": "trial_a", "new_lr": 1.1e-4, "lr_center": 1.2e-4},
        ]
        manifest["generations"][1]["exploit"] = [
            {"source": "anchored_lr_sweep", "recipient": "trial_a", "donor": "trial_b", "new_lr": 9.0e-5, "lr_center": 1.0e-4},
            {"source": "anchored_lr_sweep", "recipient": "trial_b", "donor": "trial_b", "new_lr": 1.0e-4, "lr_center": 1.0e-4},
        ]
        self.assertEqual(shared_lr_center_series(manifest), [(0, 1.2e-4), (1, 1.0e-4)])


if __name__ == "__main__":
    unittest.main()
