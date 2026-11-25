"""Query the saved global performance rule with spatial features only.

Python point API: PerformanceIRLQuery(run).query_position(x,y,z,guidance=...,smoke=...)
CSV API: --model-dir <run> --input <spatial features CSV> --output <new CSV>
"""
from pathlib import Path
import argparse
import pandas as pd
from irl_performance import PerformanceIRLQuery


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose a new path')
    frame=pd.read_csv(args.input)
    if frame.empty or 'guidance' not in frame or frame.guidance.isna().any():p.error('Input must contain guidance and nonempty spatial rows')
    model=PerformanceIRLQuery(args.model_dir);pieces=[]
    for guidance,index in frame.groupby('guidance',sort=False).indices.items():
        result=model.query_features(frame.iloc[index],guidance)
        result.index=index;pieces.append(result)
    result=pd.concat(pieces).sort_index()
    pd.concat([frame.reset_index(drop=True),result],axis=1).to_csv(args.output,index=False)


if __name__=='__main__':main()
