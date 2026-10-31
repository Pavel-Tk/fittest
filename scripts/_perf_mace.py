import sys, time; sys.path.insert(0, "/content/trustswarm")
import numpy as np, torch
from trustswarm import t1x
from trustswarm.models.mace_batched import BatchedMACE
from trustswarm.models.toolkit_io import batch_from_arrays
fr = t1x.sample_split("/content/data/Transition1x.h5", "test", 40, 4, seed=1)
Z=np.concatenate([f.Z for f in fr]); R=np.concatenate([f.R for f in fr]); b=np.concatenate([np.full(len(f.Z),i) for i,f in enumerate(fr)])
m = BatchedMACE(); print(m.info, len(fr), len(Z))
m.predict(Z,R,b)
t=time.perf_counter(); bb=batch_from_arrays(Z,R,b,device="cuda"); torch.cuda.synchronize(); t1=time.perf_counter()-t
t=time.perf_counter(); m._forward(bb); torch.cuda.synchronize(); t2=time.perf_counter()-t
t=time.perf_counter(); m.predict(Z,R,b); t3=time.perf_counter()-t
print(f"build {t1:.3f}s forward {t2:.3f}s predict {t3:.3f}s atoms/s {len(Z)/t3:.0f}")
