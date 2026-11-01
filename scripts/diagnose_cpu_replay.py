# EXPLORATORY (results/evolution/DIAGNOSIS.md). Needs the local CPU child cache (gitignored .npz) under /home/azureuser/fittest_evoC/results/*/cache.
# Usage: cd /home/azureuser/fittest_evoC && python /home/azureuser/fittest/scripts/diagnose_cpu_replay.py <out.json>
"""CPU replays (cached AIMNet2 children) -> routing and specialist-quality diagnostics. cwd = /home/azureuser/fittest_evoC."""
import json, sys, dataclasses
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/azureuser/fittest'); sys.path.insert(0, '/home/azureuser/fittest/scripts')
from fittest.cache import load_frames, load_models
from fittest.data import assign_leakfree_splits, EVAL
from fittest.evolution import EvalScorer, Swarm, fit_ledger, recommend, clean_matrix, SELECT
from fittest.evolution_c import NicheEvolution, EvoCConfig, static_c, noselect_c, FLAGGED, TRUSTED
import evolve_c

class CacheOnly(evolve_c.GPUFactory):
    def __call__(self, requests):
        for r in requests:
            s = self.spec(r)
            if not (Path(s['result']).exists() and Path(s['out']).exists()):
                raise RuntimeError(f'cache miss {s["id"]}')
        return super().__call__(requests)

frames, _ = assign_leakfree_splits(load_frames('/home/azureuser/fittest/data/frames.pkl'), 0)
RUNS = [('/home/azureuser/fittest/results/evolution_c/cpu_aimnet2_ext_seeds2-5/b800', 'results/evolution_c_ext', [2, 3, 4, 5]),
        ('/home/azureuser/fittest/results/evolution_d/cpu/b800', 'results/evolution_d_cpu', [6, 7, 8]),
        ('/home/azureuser/fittest/results/evolution_d/cpu/b800_s9', 'results/evolution_d_cpu', [9, 10, 11]),
        ('/home/azureuser/fittest/results/evolution_c/expl_cpu_protect_gatetol/b800', 'results/evolution_c_protect_gatetol', [0, 1, 2, 3, 4, 5])]
routing, children = [], []
for ref_dir, out_rel, seeds in RUNS:
    J = json.loads(Path(ref_dir, 'generations.json').read_text())
    cfg0 = EvoCConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in J['config'].items()})
    base = load_models('/home/azureuser/fittest/data/preds', list(cfg0.base_models))
    fac = CacheOnly(Path(out_rel), 'data/frames.pkl', sys.executable, sys.executable, 'cpu', 1, 1, 1)
    G = cfg0.generations
    for s in seeds:
        cfg = dataclasses.replace(cfg0, seed=s)
        for arm in ('static', 'evolved', 'noselect'):
            name = f'{arm}_s{s}'
            if f'{name}_g{G-1}' not in J['states']:
                continue
            acfg = {'static': static_c(cfg), 'evolved': cfg, 'noselect': noselect_c(cfg)}[arm]
            evo = NicheEvolution(frames, base, acfg, None if arm == 'static' else fac, None, name)
            for g in range(G):
                evo.triage(g); evo.select(g)
                if g < G - 1:
                    evo.breed(g)
            st = evo.serve_eval()
            assert abs(EvalScorer(10, 0).record('x', evo, st)['force_rmse']['value']
                       - J['states'][f'{name}_g{G-1}']['force_rmse']['value']) < 1e-6
            ev = st['eval_idx']; pop = st['population']
            sw = Swarm(evo.store, pop)
            fw, fe = evo.labels.candidate_errors(pop, sw.W, ev)
            b = sw.bucket[ev]; served = fw[np.arange(len(ev)), st['rec']]
            # hindsight best candidate per bucket on eval (ceiling of the available set)
            best = np.array([fw[b == k].mean(0).argmin() if (b == k).any() else 0 for k in range(sw.buckets.n)])
            hind = fw[np.arange(len(ev)), best[b]].mean()
            # force-objective ledger: per bucket, lowest mean force RMSE on clean labelled frames
            idx = np.array(evo.oracle.labelled)
            lf, _ = evo.labels.candidate_errors(pop, sw.W, idx)
            clean = clean_matrix(sw.subsets, evo.members, evo.component, idx)
            lb = sw.bucket[idx]
            led_rec = st['rec']
            alt = np.zeros(sw.buckets.n, int)
            for k in range(sw.buckets.n):
                m = lb == k
                mu = np.array([lf[m & clean[:, c], c].mean() if (m & clean[:, c]).sum() >= 3 else np.inf for c in range(len(sw.subsets))])
                alt[k] = mu.argmin() if np.isfinite(mu).any() else 0
            forceobj = fw[np.arange(len(ev)), alt[b]].mean()
            # with base aimnet2 restored (if pruned)
            niche = evo.niche[ev]
            names = sw.candidate_names()
            routing.append({'dir': ref_dir.split('results/')[1], 'run': name, 'population': pop,
                            'served_force_rmse': float(served.mean()), 'hindsight_best_per_bucket': float(hind),
                            'force_objective_ledger': float(forceobj),
                            'served_flagged': float(served[niche == FLAGGED].mean()), 'served_trusted': float(served[niche == TRUSTED].mean()),
                            'n_clean_labels_per_candidate': {names[c]: int(clean[:, c].sum()) for c in range(len(names))},
                            'top_served': sorted(st['served'].items(), key=lambda x: -x[1])[:3]})
            print(json.dumps({k: v for k, v in routing[-1].items() if k != 'n_clean_labels_per_candidate'}), flush=True)
            if arm == 'static':
                continue
            for mid, mem in evo.members.items():
                if mem.kind != 'offspring' or mid not in evo.store.F or not mem.gate or 'checks' not in mem.gate:
                    continue
                fc, ec = evo.labels.model_errors(mid, ev); fp, ep = evo.labels.model_errors(mem.parent, ev)
                fb, eb = evo.labels.model_errors('aimnet2', ev)
                ck = {c['name']: c for c in mem.gate['checks']}
                gF = ck['holdout_force_rmse_not_worse']; gE = ck['holdout_rel_energy_mae_not_worse']
                children.append({'dir': ref_dir.split('results/')[1], 'run': name, 'id': mid, 'parent': mem.parent, 'status': mem.status,
                                 'gate_dF_select': gF['value'] - gF['limit'], 'gate_dE_select': gE['value'] - gE['limit'] / (1 + cfg0.gate_rel_e_tol),
                                 'gate_passed': mem.gate['passed'],
                                 'eval_dF_flagged_vs_parent': float(fc[niche == FLAGGED].mean() - fp[niche == FLAGGED].mean()),
                                 'eval_dF_trusted_vs_parent': float(fc[niche == TRUSTED].mean() - fp[niche == TRUSTED].mean()),
                                 'eval_dE_flagged_vs_parent': float(ec[niche == FLAGGED].mean() - ep[niche == FLAGGED].mean()),
                                 'eval_dE_trusted_vs_parent': float(ec[niche == TRUSTED].mean() - ep[niche == TRUSTED].mean()),
                                 'eval_dF_trusted_vs_base_aimnet2': float(fc[niche == TRUSTED].mean() - fb[niche == TRUSTED].mean())})
json.dump({'routing': routing, 'children': children}, open(sys.argv[1], 'w'), indent=1)
