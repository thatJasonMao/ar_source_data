"""Independently reconcile exported data, maps and labels after generation."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from Guidance_Channel_Contract import digest
import Visualize_Guidance_Channels_V6 as maps


def verify(output):
    metadata=json.loads((output/'Metadata.json').read_text(encoding='utf8'))
    trial=pd.read_csv(output/'Trial_Proportions_Raw_vs_Sustained.csv')
    summary=pd.read_csv(output/'Summary_Raw_vs_Sustained.csv')
    labels=pd.read_csv(output/'Violin_Mean_Labels.csv',dtype={'label':str})
    floors=pd.read_csv(output/'Trial_Floor_Time_Audit.csv')
    channel=pd.read_csv(output/'Trial_Channel_Event_Audit.csv')
    keys=['Scenario','ParticipantID','FileName']
    assert len(trial)==270 and trial.ParticipantID.nunique()==45
    assert not trial.duplicated(keys).any()
    for row in trial.itertuples():
        channels=metadata['condition_channels'][row.SourceGuidance]
        assert row.Channels=='|'.join(channels)
        assert row.Guidance==metadata['condition_labels'][row.SourceGuidance]
    assert (trial.SustainedGazeDuration_ns<=trial.RawGazeDuration_ns).all()
    assert (trial.RawGazeDuration_ns<=trial.ValidDuration_ns).all()
    for name,prefix in [('RawGuidanceGazeProportion','Raw'),('SustainedGuidanceGazeProportion','Sustained')]:
        expected=trial[prefix+'GazeDuration_ns']/trial.ValidDuration_ns.replace(0,np.nan)
        np.testing.assert_allclose(trial[name],expected,atol=1e-14,rtol=0,equal_nan=True)
    for row in summary.itertuples():
        part=trial.loc[trial.Guidance.eq(row.Guidance)]
        if row.Smoke!='AllSmoke': part=part.loc[part.Smoke.eq(row.Smoke)]
        assert row.n_trials==len(part) and row.n_valid==part.ValidDuration_ns.gt(0).sum()
        assert row.n_participants==part.ParticipantID.nunique()
        for prefix,metric in [('raw','RawGuidanceGazeProportion'),('sustained','SustainedGuidanceGazeProportion')]:
            values=part[metric]
            np.testing.assert_allclose([getattr(row,prefix+'_'+stat) for stat in ['mean','median','sd']],
                                       [values.mean(),values.median(),values.std(ddof=1)],atol=1e-14,rtol=0)
    for row in labels.itertuples():
        values=trial.loc[trial.Guidance.eq(row.Guidance)&trial.Smoke.eq(row.Smoke),row.Metric].dropna()
        assert row.label==f'{values.mean():.2f}' and row.n==len(values)
    durations=['ValidDuration_ns','RawGazeDuration_ns','SustainedGazeDuration_ns']
    totals=floors.groupby(keys)[durations].sum().reindex(pd.MultiIndex.from_frame(trial[keys]))
    np.testing.assert_array_equal(totals.to_numpy(),trial[durations].to_numpy())
    channel_totals=channel.groupby(keys)[['RawDuration_ns','SustainedDuration_ns']].sum()
    channel_totals=channel_totals.reindex(pd.MultiIndex.from_frame(trial[keys]))
    np.testing.assert_array_equal(channel_totals.to_numpy(),trial[['RawChannelSumDuration_ns','SustainedChannelSumDuration_ns']].to_numpy())
    np.testing.assert_array_equal(trial.RawChannelSumDuration_ns-trial.RawGazeDuration_ns,trial.RawOverlapRemoved_ns)
    np.testing.assert_array_equal(trial.SustainedChannelSumDuration_ns-trial.SustainedGazeDuration_ns,trial.SustainedOverlapRemoved_ns)
    for mode in metadata['condition_labels'].values():
        for smoke in maps.legacy.SMOKE_ORDER:
            positive=[]
            for floor in maps.legacy.FLOOR_ORDER:
                name=f'{mode}_{smoke}_{floor}.npz'
                with np.load(output/'surfaces_sustained'/name) as sustained,np.load(output/'sensitivity_raw'/'surfaces'/name) as raw:
                    field=sustained['field']
                    np.testing.assert_array_equal(sustained['support_n'],raw['support_n'])
                    np.testing.assert_array_equal(np.isfinite(field),np.isfinite(raw['field']))
                    assert np.all(field[np.isfinite(field)]<=raw['field'][np.isfinite(field)]+1e-12)
                    assert np.all((field[np.isfinite(field)]>=0)&(field[np.isfinite(field)]<=1))
                    positive.append(field[np.isfinite(field)&(field>0)])
            combined=np.concatenate(positive)
            scale=max(float(np.quantile(combined,.98)),1e-6) if len(combined) else 1.
            np.testing.assert_allclose(scale,metadata['spatial']['scale'][mode+'/'+smoke],atol=1e-14,rtol=0)
    # Check actual renderer masking with known zero/missing/positive cells.
    zones=maps.legacy.load_zones()
    surface=dict(field=np.array([[0.,np.nan],[.2,.5]]),support_n=np.ones((2,2)),
                 participant_n=1,extent=maps.legacy.geometry_extent(zones,'Hall'))
    fig,ax=plt.subplots()
    maps.legacy.draw_panel(ax,zones,surface,.5,'Hall',None,zero_colored=True)
    artist=ax.images[1]
    np.testing.assert_array_equal(artist.get_alpha(),[[.92,0],[.92,.92]])
    assert artist.get_array()[0,0]==0 and np.ma.getmaskarray(artist.get_array())[0,1]
    np.testing.assert_allclose(artist.cmap(0.),maps.legacy.HEAT_CMAP(0.))
    assert artist.cmap(0.)[2]>artist.cmap(0.)[0] # supported zero is blue
    plt.close(fig)
    for relative,checksum in metadata['source_sha256'].items():
        assert digest(ROOT/relative)==checksum,'Changed analysis dependency: '+relative
    for info in metadata['rendering']['images']:
        with Image.open(output/info['file']) as im:
            assert im.size==(info['width_px'],info['height_px']); im.verify()
    print(json.dumps(dict(status='passed',participants=45,trials=270,summary_rows=len(summary),
        mean_labels=len(labels),surfaces=40,png_files=len(metadata['rendering']['images']),
        exported_tables_reconciled=True,renderer_supported_zero_blue=True,
        renderer_missing_transparent=True,source_dependencies_unchanged=True),indent=2))


if __name__=='__main__':
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'Results_Guidance_Sustained_AOI_V6')
