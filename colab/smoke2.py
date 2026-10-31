import sys, os, numpy as np
os.chdir("/content/fittest"); sys.path.insert(0, "/content/fittest")
for m in [k for k in list(sys.modules) if k.startswith("fittest")]: del sys.modules[m]
from fittest import t1x
from fittest.model_server import spawn
fr = t1x.sample_split("/content/data/Transition1x.h5", "test", 3, 4, seed=0)
c = spawn("orb", 6003, python="/content/envs/orb/bin/python", log_dir="/content/logs")
Z=np.concatenate([x.Z for x in fr]); R=np.concatenate([x.R for x in fr]); b=np.concatenate([np.full(len(x.Z),i) for i,x in enumerate(fr)]); Fd=np.concatenate([x.F for x in fr])
try:
    c.connect(timeout=400); o=c.predict(Z,R,b); o=c.predict(Z,R,b)
    print("orb E", o["energy"][:3], "Frmse", np.sqrt(((o["forces"]-Fd)**2).mean()), "sec", o["seconds"])
except Exception as e:
    print("FAIL", e); print(open("/content/logs/server_orb.log").read()[-2500:])
finally:
    c.close()
