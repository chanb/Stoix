### Parity
```
uv run python thought_mdp_parity_q.py
```

### Frozen Lake
```
# Frozen Lake policy training
uv run python train_reinforce.py --seeds 10 --updates 1000 --state-encoding scalar

# Estimator analysis
uv run python variance_analysis.py --rho geometric
```
