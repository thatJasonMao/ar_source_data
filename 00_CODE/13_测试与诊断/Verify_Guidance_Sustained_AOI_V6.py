"""Boundary tests for condition grouping, event unions and matched denominators."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'0_RL_SOURCE'))
from Query_GuidanceGaze import event_flags, sample_weights_ns
from Guidance_Channel_Contract import FIELDS, condition_gaze_masks
from Guidance_Sustained_AOI_V6 import duration_record


def frame_for(n, **hits):
    data={}
    for channel,(valid,hit) in FIELDS.items():
        data[valid]=np.ones(n,dtype=int)
        data[hit]=np.asarray(hits.get(channel,np.zeros(n)),dtype=int)
    return pd.DataFrame(data)


class SustainedAOITests(unittest.TestCase):
    def test_threshold_terminal_and_full_duration(self):
        ns=np.arange(6)*20_000_000
        flags,duration,first,last=event_flags(ns,np.array([1,1,1,1,1,0]))
        np.testing.assert_array_equal(flags,[1,1,1,1,1,0])
        np.testing.assert_array_equal(duration[:5],[100]*5)
        self.assertEqual(first[0],0); self.assertEqual(last[0],4)
        ns=np.array([0,10,40,90])*1_000_000
        np.testing.assert_array_equal(sample_weights_ns(ns),np.array([10,30,50,30])*1_000_000)
        np.testing.assert_array_equal(event_flags(ns,np.ones(4,bool))[0],np.ones(4,bool))

    def test_singleton_and_gap_boundary(self):
        ns=np.array([0,100,201,302])*1_000_000
        # The first two samples qualify; the 101 ms gaps split singletons.
        np.testing.assert_array_equal(event_flags(ns,np.ones(4,bool))[0],[1,1,0,0])
        self.assertFalse(event_flags(np.array([0,200])*1_000_000,np.array([1,0]))[0].any())
        np.testing.assert_array_equal(sample_weights_ns(np.array([0,200])*1_000_000),[100_000_000]*2)
        self.assertEqual(sample_weights_ns(np.array([0]))[0],0)

    def test_miss_and_invalid_interrupt_same_channel(self):
        ns=np.arange(11)*20_000_000
        frame=frame_for(11,ArrowRibbon=np.ones(11))
        frame.loc[5,FIELDS['ArrowRibbon'][0]]=0
        masks=condition_gaze_masks(frame,'ArrowRibbon',ns,event_flags)
        np.testing.assert_array_equal(masks['sustained'],[1]*5+[0]+[1]*5)
        self.assertEqual(masks['channel_event_counts']['ArrowRibbon'],2)
        frame.loc[2,FIELDS['ArrowRibbon'][1]]=0
        self.assertFalse(condition_gaze_masks(frame,'ArrowRibbon',ns,event_flags)['sustained'][:5].any())

    def test_filter_before_union_prevents_switching_event(self):
        ns=np.arange(6)*20_000_000
        frame=frame_for(6,ArrowRibbon=[1,1,1,0,0,0],MiniMap=[0,0,0,1,1,1],StaticSign=np.ones(6))
        masks=condition_gaze_masks(frame,'Full',ns,event_flags)
        self.assertTrue(masks['raw'].all())
        self.assertFalse(masks['sustained'].any())
        self.assertTrue(event_flags(ns,masks['raw'])[0].all()) # incorrect union-first alternative
        self.assertNotIn('StaticSign',masks['channels'])
        self.assertEqual(duration_record(ns,masks)['SustainedGuidanceGazeProportion'],0.)

    def test_overlap_counted_once_and_valid_union(self):
        ns=np.arange(6)*20_000_000
        frame=frame_for(6,ArrowRibbon=np.ones(6),MiniMap=np.ones(6))
        masks=condition_gaze_masks(frame,'Full',ns,event_flags)
        result=duration_record(ns,masks)
        self.assertEqual(result['ValidDuration_ns'],120_000_000)
        self.assertEqual(result['SustainedGazeDuration_ns'],120_000_000)
        self.assertEqual(result['SustainedGuidanceGazeProportion'],1.)
        frame[FIELDS['ArrowRibbon'][0]]=0
        frame[FIELDS['TextBar'][0]]=0
        self.assertTrue(condition_gaze_masks(frame,'Full',ns,event_flags)['valid'].all())

    def test_own_condition_channel_and_denominator(self):
        ns=np.arange(10)*20_000_000
        frame=frame_for(10,StaticSign=np.ones(10),ArrowRibbon=[1,1,1,1,0,0,0,0,0,0],MiniMap=np.ones(10))
        masks=condition_gaze_masks(frame,'ArrowRibbon',ns,event_flags)
        result=duration_record(ns,masks)
        self.assertEqual(masks['channels'],('ArrowRibbon',))
        self.assertEqual(result['ValidDuration_ns'],200_000_000)
        self.assertEqual(result['RawGuidanceGazeProportion'],.4)
        self.assertEqual(result['SustainedGuidanceGazeProportion'],0.)
        self.assertEqual(result['ObservationStatus'],'valid_no_event')
        frame[FIELDS['ArrowRibbon'][0]]=0
        missing=duration_record(ns,condition_gaze_masks(frame,'ArrowRibbon',ns,event_flags))
        self.assertTrue(np.isnan(missing['SustainedGuidanceGazeProportion']))
        self.assertEqual(missing['ObservationStatus'],'no_valid_data')

    def test_bad_timestamps_rejected(self):
        for ns in [np.array([0,0]),np.array([1,0])]:
            with self.assertRaises(ValueError): sample_weights_ns(ns)
            with self.assertRaises(ValueError): event_flags(ns,np.ones(2,bool))


if __name__=='__main__':
    unittest.main(verbosity=2)
