"""Last bounded V3 resolution check: four predefined 96-cluster candidates."""
from pathlib import Path
from datetime import datetime
import argparse,json,time,itertools
from irl_search_v4 import write_json
from Train_IRL_V3_Optimize import launch_batch

HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--jobs',type=int,default=2);p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=16);p.add_argument('--budget-hours',type=float,default=.5)
    args=p.parse_args();out=HERE/'Results_IRL'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_V3Resolution96');out.mkdir()
    for n in ['Tasks','Trials','Caches','Frame_Caches']:(out/n).mkdir()
    records={r['id']:r for r in json.loads((args.search/'search_records.json').read_text(encoding='utf-8'))}
    base=dict(lr=.02,l2=.001,deviation_ratio=1.,clusters=96,run_cap_s=10.,prior=.5,seed=20260926,horizon_seconds=2.,gamma=.98,basis='rbf32')
    configs=[dict(base,l2=l,horizon_seconds=h) for l,h in itertools.product([.0001,.001],[1.,2.])]
    write_json(out/'plan.json',dict(configs=configs,parent_search=str(args.search),test_used=False,scope='Last four candidates; do not auto-expand'))
    launch_batch(configs,args,out,time.time()+args.budget_hours*3600-60,records,{}, {})
    write_json(out/'completion.json',dict(status='complete',new_candidates=len(configs),total_records=len(records),test_used=False))
    print('OUTPUT='+str(out))
if __name__=='__main__':main()
