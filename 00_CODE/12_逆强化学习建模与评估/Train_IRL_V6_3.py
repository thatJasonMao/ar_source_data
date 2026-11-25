"""V6_3: four independent own-condition IRL models, pooling high/low smoke.

Uses the independent V6_2 implementation, with a distinct immutable profile:
  * StaticSign <- baseline StaticSign trials ONLY.
  * ArrowRibbon/MiniMap/TextBar <- each channel's own-condition trials ONLY.
  * Full excluded; one high + one low trial per retained person/condition.
  * Recompute ALL four isovist metrics at 12.5 m on source-valid positions.
  * Remove high_smoke from the state encoder AND the spatial reward basis.
  * Pool smoke for MDP contexts; keep every original trial/segment boundary.
  * Fit one reward per channel, then query one geometry grid at 12.5 m.
  * Export exactly four global fields (one per channel), with no high/low
    duplicates and no averaging of two separately learned smoke fields.

Original labels, valid masks, person splits and NextDecisionDistance remain
those of the V6 dataset. Smoke is retained only as evaluation provenance.
Fixed-radius geometry is cached under the new result directory; source
datasets and older models are never modified. This entrypoint and V6_2 must
remain together; no third implementation or data-engineering script is needed.

Train (from project root):
  python -B 0_RL_FRAMEWORK/Train_IRL_V6_3.py \
      --dataset 0_RL_SOURCE/Results_IRL_Dataset_V6/20261003_010150_548829 \
      --train --device cuda --budget-hours 4 --jobs 2 --epochs 300

Without --train, print a manifest-only plan. Import never runs the pipeline.
For one point: --run <Best_Model> --query X Y Z --guidance MiniMap
There is deliberately no --smoke option for this version.
"""
from pathlib import Path

from Train_IRL_V6_2 import main as _main


def main(argv=None):
    return _main(argv, variant='V6_3', entrypoint=Path(__file__).resolve())


if __name__ == '__main__':
    main()
