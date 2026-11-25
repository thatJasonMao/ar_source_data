"""Check exported Full composition, common-denominator fields and assets."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from Guidance_Channel_Contract import digest
from Visualize_Multimodal_Guidance_Components_V6 import exclusive_states


def verify(out):
    meta=json.loads((out/'Metadata.json').read_text(encoding='utf8'))
    trial=pd.read_csv(out/'Trial_Exclusive_Composition.csv')
    summary=pd.read_csv(out/'Composition_Summary.csv')
    channel=pd.read_csv(out/'Trial_Channel_Proportions.csv')
    labels=pd.read_csv(out/'Composition_Plot_Labels.csv')
    keys=['Scenario','ParticipantID','FileName']
    states=['ArrowOnly','MiniMapOnly','TextBarOnly','Overlap']
    assert len(trial)==90 and trial.ParticipantID.nunique()==45
    assert not trial.duplicated(keys).any()
    assert trial.Scenario.str.startswith('Full-').all()
    assert trial.groupby('Smoke').size().eq(45).all()
    assert channel.groupby(keys).size().eq(3).all()
    np.testing.assert_array_equal(trial[[s+'Duration_ns' for s in states]].sum(axis=1),trial.SustainedUnionDuration_ns)
    for state in states:
        expected=trial[state+'Duration_ns']/trial.SustainedUnionDuration_ns.replace(0,np.nan)
        np.testing.assert_allclose(trial[state+'Share'],expected,atol=1e-14,rtol=0,equal_nan=True)
    np.testing.assert_allclose(trial[[s+'Share' for s in states]].sum(axis=1),1,atol=1e-14,rtol=0)
    for row in summary.itertuples():
        values=trial.loc[trial.Smoke.eq(row.Smoke),row.State+'Share'].dropna()
        np.testing.assert_allclose([row.mean,row.median,row.sd],
            [values.mean(),values.median(),values.std(ddof=1)],atol=1e-14,rtol=0)
        label=labels.loc[labels.Smoke.eq(row.Smoke)&labels.State.eq(row.State)].iloc[0]
        assert label.label==f'{values.mean():.1%}'
    np.testing.assert_allclose(channel.ProportionOfUnionValidTime,
        channel.SustainedDuration_ns/channel.UnionValidDuration_ns,atol=1e-14,rtol=0)
    source=trial[keys+['ValidDuration_ns']].merge(channel,on=keys,validate='one_to_many')
    np.testing.assert_array_equal(source.ValidDuration_ns,source.UnionValidDuration_ns)
    # All eight possible retained-channel states, including triple overlap.
    flags=np.array([[bool(code&(1<<i)) for code in range(8)] for i in range(3)])
    split=exclusive_states(flags)
    np.testing.assert_array_equal(split.sum(axis=0),[0,1,1,1,1,1,1,1])
    np.testing.assert_array_equal(split[3],[0,0,0,1,0,1,1,1])
    surface_n=0
    for smoke in ['LowSmoke','HighSmoke']:
        for floor in ['Hall','Platform']:
            with np.load(out/'surfaces'/f'Union_Reference_{smoke}_{floor}.npz') as union:
                for mode in ['Arrow','MiniMap','TextBar']:
                    with np.load(out/'surfaces'/f'{mode}_{smoke}_{floor}.npz') as data:
                        np.testing.assert_array_equal(data['support_n'],union['support_n'])
                        np.testing.assert_array_equal(np.isfinite(data['field']),np.isfinite(union['field']))
                        finite=np.isfinite(data['field'])
                        assert np.all(data['field'][finite]<=union['field'][finite]+1e-12)
                        assert np.all(data['field'][finite]>=0)
                        surface_n+=1
    for relative,checksum in meta['source_sha256'].items():
        assert digest(ROOT/relative)==checksum,relative
    for relative,checksum in meta['output_sha256'].items():
        assert digest(out/relative)==checksum,relative
    pair_sizes=set()
    for info in meta['rendering']['images']:
        with Image.open(out/info['file']) as image:
            image.verify()
        with Image.open(out/info['file']) as image:
            assert image.mode=='RGBA' and image.getextrema()[3][0]==0
            assert image.size==(info['width_px'],info['height_px'])
            if info['file'].endswith('Hall_Platform.png'): pair_sizes.add(image.size)
    assert len(pair_sizes)==1
    for svg in out.glob('*.svg'):
        assert '<text' in svg.read_text(encoding='utf8'),'SVG text is not editable: '+str(svg)
    print(json.dumps(dict(status='passed',participants=45,trials=90,trial_channel_records=270,
        exclusive_shares_sum_to_one=True,common_valid_denominator=True,channel_surfaces=surface_n,
        absolute_scale=[0,1],aligned_map_pairs=6,transparent_pngs=len(meta['rendering']['images']),
        source_and_output_checksums=True,editable_svg_text=True),indent=2))


if __name__=='__main__':
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'Results_Guidance_Sustained_AOI_V6'/'Multimodal_Components')
