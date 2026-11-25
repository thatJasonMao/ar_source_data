"""Stage coverage and independent analytic distance checks for metric five."""
import json
import time
import unittest
import numpy as np
from Analyze_NextDecisionDistance import NextDecisionDistance, query_high_smoke, query_low_smoke, HERE

class Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.e=NextDecisionDistance()

    def test_every_nonstair_zone(self):
        for (floor,name),stage in self.e.rules.items():
            p=self.e.zone_shapes[name].representative_point()
            r=self.e.query(p.x,None,p.y,floor=floor)
            self.assertEqual(r.status,'ok',name)
            self.assertEqual(r.zone_id,name)
            self.assertEqual(r.stage,stage)
            self.assertTrue(np.isfinite(r.next_decision_distance))

    def test_platform_known_distance(self):
        r=self.e.query(0,-14,0)
        self.assertEqual(r.target_zone,'Stair_EFGH')
        self.assertAlmostEqual(r.next_decision_distance,7.145946,places=9)
        self.assertAlmostEqual(r.target_z,7.145946,places=9)

    def test_stages_and_progression(self):
        samples=[((0,-14,0),'to_stair'),((5,-10,0),'to_gate'),((-9,-10,0),'to_exit_corridor'),
                 ((20,-10,34),'to_branch_corridor'),((38,-10,40),'to_exit'),((38,-10,56),'arrived_exit')]
        for xyz,stage in samples:self.assertEqual(self.e.query(*xyz).stage,stage)

    def test_gate_and_exit_zero(self):
        for n,g in self.e.zone_shapes.items():
            if n.startswith('Gate_') or n.startswith('Exit_'):
                p=g.representative_point();r=self.e.query(p.x,None,p.y,floor='Hall')
                self.assertEqual(r.next_decision_distance,0.)
                self.assertEqual(r.target_zone,n)

    def test_smoke_independence_and_no_clipping(self):
        a=self.e.query_high_smoke(-9,-10,0);b=self.e.query_low_smoke(-9,-10,0)
        self.assertEqual(a,b)
        self.assertGreater(a.next_decision_distance,12.5)
        self.assertEqual(query_high_smoke(-9,-10,0),query_low_smoke(-9,-10,0))

    def test_exclusions_and_floor(self):
        self.assertEqual(self.e.query(0,-11.84,0).floor,'Hall')
        self.assertEqual(self.e.query(0,-11.86,0).floor,'Platform')
        self.assertEqual(self.e.query(0,100,0).status,'outside_height_envelope')
        self.assertEqual(self.e.query(0,100,0,floor='Hall').status,'ok')
        self.assertEqual(self.e.query(1000,-10,0).status,'outside_floor')
        self.assertEqual(self.e.query(0,float('nan'),0).status,'invalid_position')
        for floor,g in self.e.exclusions.items():
            p=g.representative_point()
            self.assertEqual(self.e.query(p.x,None,p.y,floor=floor).status,'excluded_stair')

    def test_boundary_and_equal_targets(self):
        # Midway between the two corridor mouths: true tie is reported.
        r=self.e.query(0,None,-57,floor='Hall')
        self.assertEqual(set(r.tied_targets),{'ExitCorridor_AB','ExitCorridor_CD'})
        self.assertAlmostEqual(r.next_decision_distance,10.75)
        # Corridor -> branch shared boundary is owned by higher-priority DecisionZone.
        r=self.e.query(38,None,37.104219,floor='Hall')
        self.assertEqual(r.stage,'to_branch_corridor')
        self.assertEqual(r.next_decision_distance,0.)
        self.assertEqual(self.e.query(38,None,37.10422,floor='Hall').stage,'to_exit')

    def test_random_distances_against_rectangle_formula(self):
        rng=np.random.default_rng(927)
        zones={z.zone_id:z for z in self.e.zone_query.zones}
        checked=0
        for floor,g in self.e.domains.items():
            while checked < (250 if floor=='Hall' else 500):
                x,z=rng.uniform(g.bounds[:2],g.bounds[2:])
                r=self.e.query(x,None,z,floor=floor)
                if r.status!='ok':continue
                if r.stage=='arrived_exit':expected=0.
                else:
                    candidates=[]
                    for n in self.e.groups[r.stage]:
                        for p in zones[n].collider_parts:
                            dx=max(p.xmin-x,0,x-p.xmax);dz=max(p.zmin-z,0,z-p.zmax)
                            candidates.append(np.hypot(dx,dz))
                    expected=min(candidates)
                self.assertAlmostEqual(r.next_decision_distance,expected,places=9)
                self.assertAlmostEqual(np.hypot(x-r.target_x,z-r.target_z),expected,places=9)
                checked+=1

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Validation))
    if not result.wasSuccessful():raise SystemExit(1)
    e=Validation.e;t=time.perf_counter()
    for i in range(1000):e.query(0.001*i,-10,0)
    report={'passed':True,'tests_run':result.testsRun,'analytic_random_checks':500,
            'nonstair_zone_rules':len(e.rules),'mean_ms_per_query':(time.perf_counter()-t)*1000/1000}
    (HERE/'validation_next_decision_distance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
