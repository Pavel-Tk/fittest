import sys, os, numpy as np, json
os.chdir("/content/fittest"); sys.path.insert(0, "/content/fittest")
for m in [k for k in list(sys.modules) if k.startswith("fittest")]: del sys.modules[m]
from fittest.cache import load_frames, load_predictions
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
fr = load_frames("data/frames.pkl")
P = {m: load_predictions(f"data/preds/{m}.npz") for m in ["aimnet2","aimnet2_m1","aimnet2_m2","aimnet2_m3","orb"]}
def rmse(m, f): return float(np.sqrt(((P[m].forces[f.uid]-f.F)**2).mean()))
err = {m: np.array([rmse(m,f) for f in fr]) for m in P}
for m in P: print(m, "per-frame force RMSE quantiles 10/50/90:", np.round(np.quantile(err[m],[.1,.5,.9]),3))
roles = np.array([f.role for f in fr])
for r in ["reactant","product","ts","path"]: print(r, {m: round(float(np.median(err[m][roles==r])),3) for m in ["aimnet2","orb"]})
def sig(models):
    out=[]
    for f in fr:
        F=np.stack([P[m].forces[f.uid] for m in models]); Fb=F.mean(0)
        out.append(np.sqrt(((F-Fb)**2).sum(-1).mean(0)).mean())
    return np.array(out)
s_het = sig(["aimnet2","orb"]); s_seed = sig(["aimnet2","aimnet2_m1","aimnet2_m2","aimnet2_m3"])
y = err["aimnet2"]
print("spearman sigma_het vs aimnet2 err", spearmanr(s_het,y)[0], " sigma_seed", spearmanr(s_seed,y)[0])
for t in [0.1,0.15,0.25,0.4]:
    w = y>t; print("thr",t,"frac wrong",round(w.mean(),3),"AUROC het",round(roc_auc_score(w,s_het),3),"seed",round(roc_auc_score(w,s_seed),3))
print("sigma quantiles het", np.round(np.quantile(s_het,[.1,.5,.9]),3), "seed", np.round(np.quantile(s_seed,[.1,.5,.9]),3))
