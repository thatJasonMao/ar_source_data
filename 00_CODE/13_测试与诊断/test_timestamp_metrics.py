"""Known-answer tests for the formal timestamp-local metric extractor."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import extract_timestamp_metrics as m


class TimestampMetricTests(unittest.TestCase):
    def test_exact_and_bracketed_sync_no_extrapolation_or_long_gap(self):
        pns = np.array([0, 20_000_000, 500_000_000], dtype=np.int64)
        points = np.array([[0., 0, 0], [2., 0, 0], [50., 0, 0]])
        eye = np.array([-1, 0, 10_000_000, 20_000_000, 200_000_000, 500_000_001])
        result, mode = m.align_positions(eye, pns, points)
        np.testing.assert_array_equal(mode, [0, 1, 2, 1, 0, 0])
        np.testing.assert_allclose(result[2], [1, 0, 0])
        self.assertTrue(np.isnan(result[[0, 4, 5]]).all())

    def test_angular_wrap_and_invalid_interval(self):
        v = Rotation.from_euler("y", [359, 1, 3, 5], degrees=True).apply(np.tile([0., 0, 1], (4, 1)))
        a = m.interval_angles(v, np.array([True, True, False, True]), np.array([np.nan, .02, .02, .02]))
        self.assertAlmostEqual(a[1], 2.)
        self.assertTrue(np.isnan(a[[0, 2, 3]]).all())

    def test_smoothing_does_not_bridge_long_gap(self):
        v = np.array([[0., 0, 1], [0., 0, 1], [1., 0, 0], [1., 0, 0]])
        out = m.smooth_vectors(v, np.ones(4, bool), np.array([0, .02, 1., 1.02]))
        np.testing.assert_allclose(out, v)

    def test_reference_detector_known_step(self):
        yaw = np.radians(np.r_[np.full(6, 5.), np.full(6, 45.)])
        elevation = np.radians(10.)
        positions = 10*np.column_stack([
            np.sin(yaw)*np.cos(elevation),
            np.full(len(yaw), np.sin(elevation)),
            np.cos(yaw)*np.cos(elevation),
        ])
        events, metadata = m.detect_reference_saccades(positions)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(metadata["velocity_threshold_deg_s"], 300.)
        # Preserve the reference chord/mean-radius formula rather than
        # substituting the 40-degree yaw difference.
        self.assertAlmostEqual(events[0]["amplitude_deg"], 39.36699615882736)

    def synthetic_trial(self, root, event=False, missing_gaze=False):
        scenario, participant = "StaticSign-LowSmoke", "SYNTHETIC"
        folder = root / "RAW_DATA" / scenario / participant
        folder.mkdir(parents=True)
        dates = pd.date_range("2026-01-01", periods=21, freq="20ms")
        stamps = dates.strftime(m.TIME_FORMAT).str[:-3]
        filename = "A5-1_Subject_Simulation_Info_test.csv"
        pos_x = (np.r_[np.zeros(6), np.full(15, 2.0)]
                 if event else np.linspace(0, 5, 21))
        pos = np.column_stack([pos_x, np.zeros(21), np.zeros(21)])
        head_yaw = np.zeros(21) if event else np.linspace(0, 36, 21)
        local_yaw = (np.r_[np.full(6, 5.), np.full(15, 45.)]
                     if event else np.full(21, 5.))
        yaw = np.radians(local_yaw)
        elevation = np.radians(10.)
        local_direction = np.column_stack([
            np.sin(yaw)*np.cos(elevation),
            np.full(21, np.sin(elevation)),
            np.cos(yaw)*np.cos(elevation),
        ])
        world = Rotation.from_euler("y", head_yaw, degrees=True).apply(local_direction)
        hit = pos+10*world
        if missing_gaze:
            world[10] = 0
        sdf = pd.DataFrame(dict(TimeStamp=stamps, PosX=pos[:, 0], PosY=pos[:, 1], PosZ=pos[:, 2]))
        edf = pd.DataFrame(dict(TimeStamp=stamps, HeadRotX=np.zeros(21), HeadRotY=head_yaw,
                               HeadRotZ=np.zeros(21), GazeDirX=world[:, 0], GazeDirY=world[:, 1],
                               GazeDirZ=world[:, 2], Hit1_X=hit[:, 0], Hit1_Y=hit[:, 1], Hit1_Z=hit[:, 2]))
        # Exercise the real trailing-delimiter parser, too.
        for frame, name in [(sdf, filename), (edf, filename.replace("_Subject_", "_EyeGaze_"))]:
            lines = frame.to_csv(index=False).splitlines()
            (folder / name).write_text(lines[0]+"\n"+"\n".join(line+"," for line in lines[1:])+"\n")
        trial = SimpleNamespace(Scenario=scenario, ParticipantID=participant, SubjectFolder=participant,
                                FileName=filename, StartTime=stamps[0], EndTime=stamps[-1],
                                Guidance="StaticSign", Smoke="LowSmoke")
        classifier = SimpleNamespace(assign=lambda a: np.zeros(len(a), dtype=int),
                                     zone_list=[SimpleNamespace(zone_type="Platform", zone_id="Platform_Main")])
        with patch.object(m, "ROOT", root):
            return m.process_trial(trial, classifier)

    def test_known_rotation_depth_and_independent_head_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells, events, audit, _ = self.synthetic_trial(Path(tmp), missing_gaze=True)
        self.assertAlmostEqual(cells.HeadAngle_deg.sum(), 36., places=7)
        self.assertAlmostEqual(cells.HeadTime_s.sum(), .4)
        self.assertAlmostEqual(cells.EyeTime_s.sum(), .36)
        self.assertAlmostEqual(cells.DepthIntegral_m_s.sum()/cells.DepthTime_s.sum(), 10.)
        self.assertLess(cells.EyeAngle_deg.sum(), 1e-4)
        self.assertEqual(cells.SaccadeCount.sum(), 0)
        self.assertEqual(audit["ExactPositionSamples"], 21)

    def test_cross_cell_saccade_detected_once_and_assigned_to_onset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells, events, audit, _ = self.synthetic_trial(Path(tmp), event=True)
        self.assertEqual(audit["SaccadeDetectedWholeTrial"], 1)
        self.assertEqual(audit["SaccadeUnassignedOutsideZones"], 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(cells.SaccadeCount.sum(), 1)
        self.assertEqual(int(events.iloc[0].GridCellX), 0)
        self.assertEqual(
            int(cells.loc[cells.GridCellX.eq(0), "SaccadeCount"].sum()), 1
        )
        self.assertEqual(
            int(cells.loc[cells.GridCellX.eq(1), "SaccadeCount"].sum()), 0
        )

    def test_saccade_metric_outputs_events_per_minute(self):
        totals = pd.DataFrame({
            "Scenario": ["StaticSign-LowSmoke"], "Guidance": ["StaticSign"],
            "Smoke": ["LowSmoke"], "ParticipantID": ["SYNTHETIC"],
            "Layer": ["platform"], "GridCellX": [0], "GridCellZ": [0],
            "PosX": [0.], "PosY": [0.], "PosZ": [0.],
            "HeadAngle_deg": [20.], "HeadTime_s": [2.],
            "EyeAngle_deg": [30.], "EyeTime_s": [2.],
            "DepthIntegral_m_s": [10.], "DepthTime_s": [2.],
            "PositionTime_s": [2.], "SaccadeCount": [3],
            "SaccadeValidHitSamples": [90], "SaccadeExposure_s": [2.],
        })
        values = m.metric_values(totals, minimum=1.)
        rate = values.loc[values.Metric.eq("saccade_rate")].iloc[0]
        self.assertEqual(rate.Numerator, 180.)
        self.assertEqual(rate.Exposure_s, 2.)
        self.assertEqual(rate.Value, 90.)
        self.assertEqual(rate.Unit, "events/min")
        self.assertTrue(rate.Supported)


if __name__ == "__main__":
    unittest.main(verbosity=2)
