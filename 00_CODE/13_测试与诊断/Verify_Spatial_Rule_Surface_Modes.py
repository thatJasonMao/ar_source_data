"""Check closed columns, continuous shared edges and raw-value preservation; no training."""
from pathlib import Path
import argparse
import importlib.util
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLOT = None


def fixture():
    # Large jumps, negative/zero/positive values, an interior hole, an isolated cell.
    positions = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)] + [(6, 0)]
    values = [-100., 0., 80., -30., 5., 100., -2., 50., -7.]
    return pd.DataFrame([dict(x=(i+.5)*.4, z=(j+.5)*.4, relative_information_utility=v)
                         for (i, j), v in zip(positions, values)])


class SurfaceGeometry(unittest.TestCase):
    def test_columns_closed_exact_signed_top_and_full_cell_width(self):
        part = fixture()
        base = -116.
        faces, values = PLOT.bar_faces(part, base)
        self.assertEqual(faces.shape, (len(part)*6, 4, 3))
        np.testing.assert_array_equal(values, np.repeat(part[PLOT.FIELD].to_numpy(), 6))
        for index, row in enumerate(part.itertuples(index=False)):
            group = faces[index*6:(index+1)*6]
            self.assertEqual(set(group[:, :, 2].ravel()), {base, row.relative_information_utility})
            np.testing.assert_allclose(np.ptp(group[:, :, :2].reshape(-1, 2), axis=0), [.4, .4])
            np.testing.assert_allclose(group[:, :, :2].reshape(-1, 2).mean(0), [-row.x, -row.z])
            self.assertEqual(sum(np.all(face[:, 2] == row.relative_information_utility) for face in group), 1)
            # A closed cuboid has 12 geometric edges; every edge belongs to two faces.
            edges = {}
            for face in group:
                for a, b in zip(face, np.roll(face, -1, axis=0)):
                    key = tuple(sorted((tuple(np.round(a, 10)), tuple(np.round(b, 10)))))
                    edges[key] = edges.get(key, 0)+1
            self.assertEqual(len(edges), 12)
            self.assertEqual(set(edges.values()), {2})

    def test_continuous_exact_centers_shared_edges_and_holes(self):
        part = fixture()
        faces, values = PLOT.continuous_faces(part)
        self.assertEqual(faces.shape, (len(part)*4, 3, 3))
        np.testing.assert_array_equal(values, np.repeat(part[PLOT.FIELD].to_numpy(), 4))
        vertex_heights = {}
        for index, row in enumerate(part.itertuples(index=False)):
            group = faces[index*4:(index+1)*4]
            center = np.array([-row.x, -row.z, row.relative_information_utility])
            for face in group:
                self.assertTrue(np.any(np.all(np.isclose(face, center, rtol=0, atol=1e-12), axis=1)))
            # All projected faces stay within their original valid cell, so none fill the hole.
            self.assertTrue(np.all(np.abs(group[:, :, :2]-center[:2]) <= .2+1e-12))
            delta_a = group[:, 1, :2]-group[:, 0, :2]
            delta_b = group[:, 2, :2]-group[:, 0, :2]
            areas = np.abs(delta_a[:, 0]*delta_b[:, 1]-delta_a[:, 1]*delta_b[:, 0])/2
            self.assertTrue(np.all(areas > 0))
            self.assertAlmostEqual(areas.sum(), .16, places=12)
            for vertex in group.reshape(-1, 3):
                key = tuple(np.round(vertex[:2], 10))
                if key in vertex_heights:
                    self.assertAlmostEqual(vertex_heights[key], vertex[2], places=12)
                vertex_heights[key] = vertex[2]
        self.assertGreaterEqual(faces[:, :, 2].min(), part[PLOT.FIELD].min())
        self.assertLessEqual(faces[:, :, 2].max(), part[PLOT.FIELD].max())
        # An isolated cell stays flat, with no extrapolation towards remote cells.
        np.testing.assert_array_equal(faces[-4:, :, 2], np.full((4, 3), -7.))

    def test_continuous_corner_height_is_local_and_original_values_are_not_mutated(self):
        part = pd.DataFrame({'x': [.2, .6], 'z': [.2, .2], PLOT.FIELD: [-100., 100.]})
        before = part.copy(deep=True)
        faces, colors = PLOT.continuous_faces(part)
        seam = faces.reshape(-1, 3)
        seam = seam[np.isclose(seam[:, 0], -.4)]
        np.testing.assert_array_equal(seam[:, 2], np.zeros(len(seam)))
        pd.testing.assert_frame_equal(part, before)
        np.testing.assert_array_equal(colors, [-100.]*4+[100.]*4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path, default=ROOT/'0_RL_FRAMEWORK/Plot_Spatial_Rule_3D.py')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('surface_renderer_under_test', args.renderer)
    PLOT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PLOT)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SurfaceGeometry)
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
