# EXPLORATORY (results/evolution/DIAGNOSIS.md). Usage: python scripts/diagnose_labelswap.py results/evolution/diagnosis/labelswap.json
"""(b) label-allocation test: serve the STATIC population (base models only) with the ledger fitted on
the labelled set that the evolved / noselect run actually acquired (from its blackboard triage events)."""
import json, sys, glob, dataclasses
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/azureuser/fittest')
from fittest.cache import load_frames, load_models
from fittest.data import assign_leakfree_splits
from fittest.evolution import EvalScorer
from fittest.evolution_c import NicheEvolution, EvoCConfig, static_c
R = Path('/home/azureuser/fittest/results')
frames, _ = assign_leakfree_splits(load_frames('/home/azureuser/fittest/data/frames.pkl'), 0)
base = load_models('/home/azureuser/fittest/data/preds', ['mace_mpa0', 'aimnet2', 'orb'])
dirs = sorted(set(p.parent for p in list(R.glob('evolution_c/**/generations.json')) + list(R.glob('evolution_d/**/generations.json'))
                  if 'partial' not in str(p)))
out = []
for d in dirs:
    J = json.loads((d / 'generations.json').read_text())
    cfgd = {k: (tuple(v) if isinstance(v, list) else v) for k, v in J['config'].items()}
    cfg0 = EvoCConfig(**cfgd)
    labs = {}
    for l in open(d / 'blackboard.jsonl'):
        e = json.loads(l)
        if e['kind'] == 'triage':
            labs.setdefault(e['author'], []).extend(e['payload']['new_labels'])
    G = cfg0.generations
    seeds = sorted({int(k.rsplit('_g', 1)[0].rsplit('_s', 1)[1]) for k in J['states']})
    for s in seeds:
        evo = NicheEvolution(frames, base, static_c(dataclasses.replace(cfg0, seed=s)), None, None, 'x')
        sc = EvalScorer(1000, 0)
        def serve(uids, name):
            evo.oracle.labelled = [evo.store.index[u] for u in uids]
            st = evo.serve_eval()
            return sc.record(name, evo, st), st['served']
        v0, _ = serve(labs[f'static_s{s}'], 'static')
        ref = J['states'][f'static_s{s}_g{G-1}']['force_rmse']['value']
        assert abs(v0['force_rmse']['value'] - ref) < 1e-9, (d, s, v0['force_rmse']['value'], ref)
        for arm in ('evolved', 'noselect'):
            a = f'{arm}_s{s}'
            if a not in labs or f'{a}_g{G-1}' not in J['states']:
                continue
            v1, served = serve(labs[a], a)
            p = sc.paired(a, 'static')['force_rmse']
            ov = len(set(labs[a]) & set(labs[f'static_s{s}'])) / len(labs[a])
            real = J['comparisons'][a]['final_vs_static']['force_rmse']
            out.append({'dir': str(d.relative_to(R)), 'run': a, 'label_overlap_with_static': ov,
                        'swap_dF': p['diff'], 'swap_ci': p['ci'], 'swap_sig': p['significant'],
                        'real_dF': real['diff'], 'real_sig': real['significant'], 'swap_served': served})
            print(f"{out[-1]['dir']:42s} {a:12s} overlap {ov:.2f} swap dF {p['diff']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] | real {real['diff']:+.4f}{'*' if real['significant'] else ''}", flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
