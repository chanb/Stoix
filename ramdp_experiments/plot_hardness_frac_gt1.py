import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(9,3.2))
for ax,n,t in zip(axes,["lightsout","slidingpuzzle"],["Lightsout 5x4 (seed 4)","Sliding puzzle 3x3 (seed 2)"]):
    df=pd.read_pickle(f"rollout_cache-{n}-cnn16x4-hent0-n50000-seed0.pkl")
    df["gt1"]=df.compute_times.apply(lambda c:c[0]>1)
    g=df.groupby("shortest_path_length")["gt1"]; p=g.mean(); k=g.size()
    se=np.sqrt(p*(1-p)/k)
    ax.errorbar(p.index,100*p,yerr=196*se,marker="o",ms=3)
    ax.set_title(t); ax.set_xlabel("Shortest path length"); ax.grid(alpha=.3)
axes[0].set_ylabel("% first steps with >1 compute step")
fig.tight_layout(); fig.savefig("analysis-hardness_frac_gt1.png",dpi=150)
