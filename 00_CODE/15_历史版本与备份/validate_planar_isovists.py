"""Analytic geometry, independent ray checks, and Timeline compatibility tests."""
import json
from pathlib import Path
import time
import unittest
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box
from spatial_isovist_core import PlanarIsovist, polygon_metrics, HERE
from UnifiedZoneQuery import VectorizedZoneClassifier


def synthetic(poly, rays=720):
    e = PlanarIsovist(rays=rays)
    e.domains['Hall'] = poly
    e.exclusions['Hall'] = Polygon()
    c = np.asarray(poly.exterior.coords)
    e.segments['Hall'] = c[:-1], c[1:]
    return e


class Validation(unittest.TestCase):
    def test_unobstructed_circle(self):
        e = synthetic(box(-50, -50, 50, 50))
        for radius in (7.5, 12.5):
            r = e.query(0, None, 0, radius, floor='Hall')
            self.assertEqual(r.status, 'ok')
            self.assertLess(abs(r.horizontal_size/(np.pi*radius**2)-1), 1.3e-5)
            self.assertAlmostEqual(r.concavity, 0, places=12)
            self.assertAlmostEqual(r.distance_variation, 0, places=12)
            self.assertAlmostEqual(r.elongation, 1, places=12)

    def test_rectangle_area_and_uniform_area_elongation(self):
        e = synthetic(box(-4, -1, 4, 1))
        r = e.query(0, None, 0, 12.5, floor='Hall')
        self.assertAlmostEqual(r.horizontal_size, 16, places=8)
        self.assertAlmostEqual(r.concavity, 0, places=10)
        self.assertAlmostEqual(r.elongation, 4, places=8)
        # Extra boundary vertices cannot change an area-weighted moment.
        p = np.array([[-4,-1], [0,-1], [4,-1], [4,1], [-4,1]])
        self.assertAlmostEqual(polygon_metrics(p, np.ones(10))[3], 4, places=12)

    def test_L_shape_concavity(self):
        poly = Polygon([(0,0),(8,0),(8,2),(2,2),(2,8),(0,8)])
        r = synthetic(poly).query(1, None, 1, 12.5, floor='Hall')
        self.assertAlmostEqual(r.horizontal_size, 28, places=6)
        self.assertAlmostEqual(r.concavity, 1-28/46, places=7)

    def test_boundary_origin(self):
        e = synthetic(box(0,-50,50,50))
        r = e.query(0,None,0,7.5,floor='Hall')
        self.assertEqual(r.status, 'ok')
        self.assertLess(abs(r.horizontal_size/(np.pi*7.5**2/2)-1), 1.3e-5)
        self.assertAlmostEqual(r.concavity, 0, places=10)

    def test_status_and_stairs(self):
        e=PlanarIsovist()
        self.assertEqual(e.query(1000,-10,1000,7.5).status, 'outside_floor')
        self.assertEqual(e.query(0,float('nan'),0,7.5).status, 'invalid_position')
        self.assertEqual(e.query(0,50,0,7.5).status, 'outside_height_envelope')
        self.assertEqual(e.query(0,50,0,7.5,floor='Hall').status, 'ok')
        for floor,g in e.exclusions.items():
            p=g.representative_point()
            self.assertEqual(e.query(p.x,None,p.y,7.5,floor=floor).status, 'excluded_stair')
        with self.assertRaises(ValueError):e.query(0,-10,0,-1)

    def test_timeline_floor_equivalence(self):
        e=PlanarIsovist(); q=e.zone_query
        ys=np.concatenate((np.linspace(-16,-7,1001),[-11.850001,-11.85,-11.849999,-11.800001,-11.8,-11.799999,np.nan]))
        a=VectorizedZoneClassifier(q).floor_of(ys)
        b=np.array([{'Platform':0,'Hall':1,None:-1}[q.classify_floor(y)] for y in ys])
        np.testing.assert_array_equal(a,b)
        self.assertEqual(q.classify_floor(-11.849), 'Hall')
        self.assertEqual(q.classify_floor(-11.851), 'Platform')

    def test_independent_shapely_rays(self):
        e=PlanarIsovist(); rng=np.random.default_rng(127)
        for floor,g in e.domains.items():
            lo=np.array(g.bounds[:2]); hi=np.array(g.bounds[2:])
            for _ in range(12):
                while True:
                    p=rng.uniform(lo,hi)
                    if g.contains(Point(*p)):break
                angles=rng.uniform(0,2*np.pi,20)
                directions=np.column_stack((np.cos(angles),np.sin(angles)))
                actual=e._radial(p,floor,12.5,directions,*e.segments[floor])
                for u,r in zip(directions,actual):
                    cut=g.intersection(LineString([p,p+u*12.5]))
                    parts=list(getattr(cut,'geoms',[cut]))
                    first=[v for v in parts if v.distance(Point(*p))<1e-8]
                    self.assertEqual(len(first),1)
                    self.assertAlmostEqual(r,first[0].length,places=7)

    def test_public_modules(self):
        import Analyze_HorizontalSize as a
        import Analyze_Concavity as b
        import Analyze_DistanceVariation as c
        import Analyze_Elongation as d
        for m in (a,b,c,d):
            self.assertTrue(np.isfinite(m.query_high_smoke(0,-10,0)))
            self.assertTrue(np.isfinite(m.query_low_smoke(0,None,0,floor='Hall')))
            self.assertTrue(np.isnan(m.query_high_smoke(1000,-10,1000)))


def convergence_audit():
    e=PlanarIsovist(); fine=PlanarIsovist(rays=2880)
    rng=np.random.default_rng(9126); errors=[]; count=0
    start=time.perf_counter()
    for floor,g in e.domains.items():
        for _ in range(40):
            while True:
                p=rng.uniform(g.bounds[:2],g.bounds[2:])
                if e.locate(p[0],None,p[1],floor=floor)[1]=='ok':break
            for radius in (7.5,12.5):
                a=e.query(p[0],None,p[1],radius,floor=floor)
                b=fine.query(p[0],None,p[1],radius,floor=floor)
                errors.append([abs(a.horizontal_size/b.horizontal_size-1),abs(a.concavity-b.concavity),abs(a.distance_variation-b.distance_variation),abs(a.elongation/b.elongation-1)])
                count+=1
            small=e.query(p[0],None,p[1],7.5,floor=floor)
            large=e.query(p[0],None,p[1],12.5,floor=floor)
            assert small.horizontal_size <= large.horizontal_size+1e-8
    # Warm real-position query timing (all four metrics at once).
    t=time.perf_counter()
    for i in range(1000):e.query(0.001*i,-10,0,12.5)
    ms=(time.perf_counter()-t)*1000/1000
    return {'comparison':'720 vs 2880 uniform rays, independent random origins',
            'evaluations':count, 'max_errors_area_relative_concavity_absolute_CV_absolute_elongation_relative':np.max(errors,axis=0).tolist(),
            'mean_ms_per_query_all_four_metrics':ms,'seconds':time.perf_counter()-start}

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Validation)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    report={'tests_run':result.testsRun,'passed':True,**convergence_audit()}
    (HERE/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
