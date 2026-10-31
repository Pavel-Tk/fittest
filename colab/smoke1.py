import subprocess, sys, os, time
os.chdir("/content/trustswarm"); sys.path.insert(0, "/content/trustswarm")
import importlib, numpy as np, h5py
for m in [k for k in list(sys.modules) if k.startswith("trustswarm")]: del sys.modules[m]
from trustswarm import t1x
P="/content/data/Transition1x.h5"
with h5py.File(P) as f:
    print(list(f.keys()), {k: len(f[k]) for k in f.keys()})
    fm = next(iter(f["test"])); rx = next(iter(f["test"][fm])); g=f["test"][fm][rx]
    print(fm, rx, list(g.keys()), g["positions"].shape)
    print("n test rxns", sum(len(f["test"][k]) for k in f["test"]), "val", sum(len(f["val"][k]) for k in f["val"]))
fr = t1x.sample_split(P, "test", 3, 4, seed=0)
print(len(fr), [(x.uid, x.role, len(x.Z), round(x.E,2)) for x in fr[:8]])
from trustswarm.model_server import spawn
cl = [spawn("orb", 6003, python="/content/envs/orb/bin/python", log_dir="/content/logs"), spawn("aimnet2", 6002, python="/usr/bin/python3", log_dir="/content/logs")]
Z=np.concatenate([x.Z for x in fr]); R=np.concatenate([x.R for x in fr]); b=np.concatenate([np.full(len(x.Z),i) for i,x in enumerate(fr)])
Fd=np.concatenate([x.F for x in fr])
for c in cl:
    try:
        c.connect(timeout=400); print(c.info())
        o=c.predict(Z,R,b); o=c.predict(Z,R,b)
        rm=np.sqrt(((o["forces"]-Fd)**2).mean()); print(c.model, "E", o["energy"][:3], "Frmse_vs_dft", rm, "sec", o["seconds"])
    except Exception as e:
        print("FAIL", c.model, e); print(open(f"/content/logs/server_{c.model}.log").read()[-3000:])
    finally:
        try: c.close()
        except Exception as e: print(e)
