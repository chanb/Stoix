# SEM round 2: hyperparameters

Launched 2026-11-07 06:05 (wandb project `stoix-sem`). This replaced two earlier round-2 launches the
same morning that were stopped partway: one used the default orthogonal `block_init`, and MinAtar also
used max_grad_norm 0.5. These values are taken from the generated
`ff_ppo.py` commands (`ff_ppo_reinforce`, implicit-CoT transformer). Launch flags:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/settings.sh`. Launcher:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch.sh`.

"same" means the value is the same as Lightsout. "n/a" means the setting does not apply.

## Runs

| | Lightsout | Sliding puzzle | Sokoban | MinAtar |
|---|---|---|---|---|
| Variants | fixed budget 1, 5; variable (max 5) | same | same | same |
| Seeds | 5 (0–4) | same | same | same |
| Runs | 15 | 15 | 15 | 75 (5 games × 15) |
| total_timesteps | 1e8 | 1e8 | 3e8 | 1e8 |
| total_num_envs | 1024 | same | same | same |
| Runs per GPU | 2 (XLA memory fraction 0.475) | same | same | same |

## Environment

| | Lightsout | Sliding puzzle | Sokoban | MinAtar |
|---|---|---|---|---|
| Env config | `lightsout/lightsout_3x3`, overridden to 5x4 | `jumanji/slidingtile_bank_grid` | `jumanji/sokoban_grid` | `gymnax/<game>` |
| Task | Lights Out 5x4 | 15-puzzle (4x4) from level banks | Boxoban | asterix, breakout, freeway, seaquest, space_invaders |
| Train instances | goals of optimal length 1–9 (`difficulty_threshold` 0.5) | train bank: optimal length ≤ 60, length drawn uniformly | `unfiltered-train` | the game |
| Eval instances | goals of optimal length 10–19 | test bank: optimal length > 60 | `unfiltered-valid` | the game |
| Train episode length | 10 | 60 | jumanji default (120) | game default |
| Eval episode length | 20 | 75 | jumanji default (120) | game default |
| Observation | flattened grid | grid (H, W, 1) | grid | (10, 10, C) grid |
| Train-distribution eval (`evaluator/train/*`) | yes | yes | yes | no |

## PPO / optimization

| | Lightsout | Sliding puzzle | Sokoban | MinAtar |
|---|---|---|---|---|
| gamma | 0.999 | 0.999 | 0.999 | 0.99999 |
| max_grad_norm | 1.0 | 1.0 | 5.0 | 1.0 |
| actor_lr | 1e-4 | same | same | same |
| critic_lr | 1e-4 | same | same | same |
| halting_lr | 1e-4 (variable budget) | same | same | same |
| actor / critic / halting weight decay | 0 / 0 / 0 | same | same | same |
| decay_learning_rates | False | same | same | same |
| rollout_length | 64 | same | same | same |
| epochs | 2 | same | same | same |
| num_minibatches | 64 | same | same | same |
| Batch per update / minibatch size | 65,536 / 1,024 | same | same | same |
| clip_eps | 0.3 | same | same | same |
| clip_value_loss | False | same | same | same |
| Value loss | MSE (use_expectile_value_loss False) | same | same | same |
| vf_coef | 1.0 | same | same | same |
| ent_coef | 0.01 | same | same | same |
| gae_lambda | 0.95 | same | same | same |
| standardize_advantages | True | same | same | same |
| recompute_advantages | False | same | same | same |
| critic_before_actor | False | same | same | same |
| use_dpo_loss | False | same | same | same |
| latent_kl_coef | 0 | same | same | same |

## Halting (compute budget)

| | Lightsout | Sliding puzzle | Sokoban | MinAtar |
|---|---|---|---|---|
| Fixed budget | min_steps = max_steps = budget (1 or 5) | same | same | same |
| Variable budget min / max steps | 1 / 5 | same | same | same |
| halting_ent_coef | 0.01 | same | same | same |
| halting_temperature | 1.0 | same | same | same |
| halting_hidden_dims | [] (linear readout) | same | same | same |
| clip_halting_head | True | same | same | same |
| stop_gradient_halting_input | False | same | same | same |

## Actor network

| | Lightsout | Sliding puzzle | Sokoban | MinAtar |
|---|---|---|---|---|
| Network config | `transformer_compute` | `cnn_transformer_compute` | same as sliding | same as sliding |
| Input encoder | one Dense layer → 8 state tokens (`num_state_tokens` 8) | CNN, channels [32, 8], kernels [3, 3], strides [1, 1], GELU; one token per last-layer channel → 8 tokens of H·W features (`channel_tokens`, `input_is_tokens`); its MLP `hidden_sizes` [128] is not used in token mode | same as sliding | same as sliding |
| Transformer | `TransformerChainOfThoughtTorso` | same | same | same |
| hidden_dim | 128 | same | same | same |
| mlp_dim | 512 | same | same | same |
| qkv_dim | = hidden_dim (default) | same | same | same |
| num_layers | 2 | same | same | same |
| num_heads | 4 | same | same | same |
| activation | GELU | same | same | same |
| use_input_layer_norm | True | same | same | same |
| use_rmsnorm | True | same | same | same |
| use_sandwich_norm | False | same | same | same |
| SEM | sem_group_size 8, sem_temperature 1.0 | same | same | same |
| latent_rms_norm | True | same | same | same |
| input_injection | True | same | same | same |
| residual_scaling | True | same | same | same |
| block_init (kernel init of every `TransformerBlock` projection in the CoT loop) | `trunc_normal` (`variance_scaling`, fan_in, truncated normal) | same | same | same |
| Action head | categorical | same | same | same |

## Critic network

| | Lightsout | Sliding puzzle | Sokoban | MinAtar |
|---|---|---|---|---|
| Input encoder | none (flattened observation) | CNN, channels [32, 64], kernels [3, 3], strides [1, 1], GELU, flattened (no hidden MLP) | same as sliding | same as sliding |
| MLP | [256, 256, 256], GELU after every layer, no layer norm | same | same | same |
| Head | scalar value | same | same | same |

## Evaluation

| | Lightsout | Sliding puzzle | Sokoban | MinAtar |
|---|---|---|---|---|
| num_evaluation | 50 | same | same | same |
| num_eval_episodes | 256 | same | same | same |
| evaluation_greedy | False (sampled actions) | same | same | same |

## Lightsout CNN encoder + halting-loss comparison

Launched 2026-11-07 ~18:25 (wandb project `stoix-sem`). Launcher:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch_lightsout_cnn_haltloss.sh`. Output:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/lightsout_cnn_haltloss/`.

All settings are the Lightsout variable-budget column above (max 5 steps, 1e8 steps, 2 runs per GPU),
with two changes:

1. The actor encoder is the sliding-puzzle CNN, not one Dense layer: `--architectures cnn+transformer
   --cnn-tokens true`. The actor CNN [32, 8] gives 8 state tokens (one per last-layer channel). The
   critic CNN [32, 64] is flattened into the 3×256 GELU MLP. All use GELU.
2. `system.halting_loss_type` is swept over `pg`, `pondernet` and `act`, with 5 seeds (0–4) each, so 15 runs.

### Halting-loss settings

| | pg | pondernet | act |
|---|---|---|---|
| How the halting head is trained | per-step PPO clip on the sampled halt decisions | gradient through p_n in the mixture policy Σ p_n π_n(a\|s), PPO clip on the mixture ratio | gradient through the p_n-weighted mean state and the ponder cost |
| Rollout halting | sampled per step | sampled per step (a sample from the mixture) | deterministic: first N with Σ λ ≥ 1 − ε |
| Extra halting loss term | none | `pondernet_kl_coef` · KL(p ‖ truncated Geometric(`pondernet_prior_lambda`)) | `act_ponder_cost_coef` · (N + R) |
| pondernet_kl_coef (β) | n/a | 0.01 | n/a |
| pondernet_prior_lambda (λ_p) | n/a | 0.2 | n/a |
| act_ponder_cost_coef (τ) | n/a | n/a | 0.01 |
| act_eps (ε) | n/a | n/a | 0.01 |
| clip_eps on the halting decision | 0.3 | n/a (halting is inside the mixture ratio, which uses 0.3) | n/a (inside the action ratio, which uses 0.3) |
| Action entropy bonus (ent_coef 0.01) on | π at the halt step | Σ p_n H(π_n), p held fixed | π of the mean state |
| halting_ent_coef | 0.01 | 0.01 | 0.01 |
| halting_lr / clip_halting_head | 1e-4 / True | same | same |
| halting_temperature / halting_hidden_dims | 1.0 / [] | same | same |
| min_steps / max_steps | 1 / 5 | same | same |

For pondernet and act, the per-step halting entropy comes from a replay to max_steps. It is masked to
the steps the rollout took. All pondernet/act values are the `ff_ppo.yaml` defaults, not tuned.

## Halting-loss tuning log

Goal: higher train episode discounted return (`ACTOR` episode discounted return, mean of the last 5
logs) for each halting loss. Settings not listed are the ones in the section above. Analysis script:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/analysis/haltloss_curves.py`. It cuts every run to the
shortest budget in the comparison, so round 0 is compared at 5e7 steps when it is compared with round 1.
"lr" means actor_lr = critic_lr = halting_lr.

### Round 0 (2026-11-07, 1e8 steps, 5 seeds) — defaults

| Config | Halting loss | Changes from defaults | Train disc. return @1e8 | Train compute time |
|---|---|---|---|---|
| R0-pg | pg | none | 0.704 ± 0.038 | 1.64 |
| R0-pn | pondernet | none (β 0.01, λ_p 0.2) | 0.760 ± 0.023 | 2.36 |
| R0-act | act | none (τ 0.01, ε 0.01) | 0.760 ± 0.025 | 2.00 (stuck) |

All curves are still rising at 1e8. ACT is stuck at N = 2: λ₂ gets no gradient (R = 1 − λ₁), so N
never changes. Fix tried in round 1: `halting_bias_init` (new torso option, initial halting-logit
offset, default 0).

### Round 1 (2026-11-08, 5e7 steps, 2 seeds, one GPU per config)

Launcher: `/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch_haltloss_tune_r1.sh`. Output:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/haltloss_tune_r1/`.

| Config | GPU | Halting loss | Changes from defaults | Question |
|---|---|---|---|---|
| R1-pg-lr3e-4 | 0 | pg | lr 3e-4 | Is lr 1e-4 too slow (curves still rising)? |
| R1-pn-lr3e-4 | 1 | pondernet | lr 3e-4 | same |
| R1-act-lr3e-4 | 2 | act | lr 3e-4 | same |
| R1-act-bias-3 | 3 | act | halting_bias_init −3 | Does ACT stop sticking at N = 2? |
| R1-act-bias-3-tau1e-3 | 4 | act | halting_bias_init −3, τ 0.001 | Same, with a lower ponder cost |
| R1-pn-beta1e-3 | 5 | pondernet | β (pondernet_kl_coef) 0.001 | Is the prior too strong? |
| R1-pn-prior0.5 | 6 | pondernet | λ_p (pondernet_prior_lambda) 0.5 | Does a short-pondering prior help? |
| R1-pg-bias-2 | 7 | pg | halting_bias_init −2 | Does pg do better if it starts pondering longer? |

Results at 5e7 steps (train discounted return, mean of the last 5 logs ± s.e. over seeds; round 0
cut to 5e7 for the baselines). Plot: `/home/bryanpu1/scratch/logs/ramdp/sem-round2/analysis/round1.png`.

| Config | Train disc. return @5e7 | Train compute time | vs. same-loss baseline |
|---|---|---|---|
| R1-act-lr3e-4 | **0.766 ± 0.006** | 2.00 (stuck) | +0.132 |
| R1-pn-lr3e-4 | **0.758 ± 0.053** | 2.33 | +0.134 |
| R1-pg-lr3e-4 | **0.738 ± 0.011** | 1.63 | +0.156 |
| R1-pn-beta1e-3 | 0.669 ± 0.024 | 2.08 | +0.045 |
| R1-pn-prior0.5 | 0.636 ± 0.023 | 1.88 | +0.012 (noise) |
| R0-act (baseline) | 0.634 ± 0.018 | 2.00 | — |
| R0-pn (baseline) | 0.624 ± 0.014 | 2.36 | — |
| R1-act-bias-3 | 0.615 ± 0.029 | 2.00 → varies (2.0–2.6) | −0.019 |
| R1-act-bias-3-tau1e-3 | 0.593 ± 0.000 | 2.12 | −0.041 |
| R0-pg (baseline) | 0.582 ± 0.017 | 1.72 | — |
| R1-pg-bias-2 | 0.559 ± 0.000 | 1.71 | −0.023 |

Findings: lr 3e-4 is the largest gain for every loss. A weaker pondernet prior (β 0.001) helps. A
negative halting bias unsticks ACT's N but lowers return, and it slows early learning for pg.

### Round 2 (2026-11-08, 5e7 steps, 2 seeds, one GPU per config) — base lr 3e-4

Launcher: `/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch_haltloss_tune_r2.sh`. Output:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/haltloss_tune_r2/`. Baselines: the round-1 lr 3e-4 configs.

| Config | GPU | Halting loss | Changes from defaults | Question |
|---|---|---|---|---|
| R2-pg-lr1e-3 | 0 | pg | lr 1e-3 | Does lr keep helping above 3e-4? |
| R2-pn-lr1e-3 | 1 | pondernet | lr 1e-3 | same |
| R2-act-lr1e-3 | 2 | act | lr 1e-3 | same |
| R2-pn-beta1e-3 | 3 | pondernet | lr 3e-4, β 0.001 | Does the β gain hold at lr 3e-4? |
| R2-pn-beta0 | 4 | pondernet | lr 3e-4, β 0 | Is the geometric prior needed at all? |
| R2-act-tau1e-3 | 5 | act | lr 3e-4, τ 0.001 | Does a lower ponder cost help ACT (no bias)? |
| R2-pg-hent0 | 6 | pg | lr 3e-4, halting_ent_coef 0 | Does the halting entropy bonus hurt pg? |
| R2-pg-hlr1e-3 | 7 | pg | lr 3e-4, halting_lr 1e-3 | Does a faster halting head help pg? |

Results at 5e7 steps (baselines: round-1 lr 3e-4 configs). Plot:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/analysis/round2.png`.

| Config | Train disc. return @5e7 | Train compute time | vs. same-loss lr 3e-4 baseline |
|---|---|---|---|
| R2-pn-beta1e-3 | **0.782 ± 0.006** | 2.01 | +0.024 |
| R1-act-lr3e-4 (baseline) | 0.766 ± 0.006 | 2.00 (stuck) | — |
| R2-act-tau1e-3 | 0.761 ± 0.000 | 2.02 (stuck) | −0.005 (noise) |
| R1-pn-lr3e-4 (baseline) | 0.758 ± 0.053 | 2.33 | — |
| R2-pn-beta0 | 0.749 ± 0.015 | 1.92 | −0.009 (noise) |
| R1-pg-lr3e-4 (baseline) | 0.738 ± 0.011 | 1.63 | — |
| R2-pg-hlr1e-3 | 0.658 ± 0.037 | 1.65 | −0.080 |
| R2-pg-lr1e-3 | 0.637 ± 0.017 | 1.61 | −0.101 |
| R2-pg-hent0 | 0.623 ± 0.028 | 2.45 (rises to 4.5, then collapses) | −0.115 |
| R2-act-lr1e-3 | 0.604 ± 0.009 | 2.00 | −0.162 |
| R2-pn-lr1e-3 | 0.597 ± 0.009 | 2.32 | −0.161 |

Findings: lr 3e-4 is the best lr. lr 1e-3 learns fastest early but stalls near 0.6 for all losses.
pondernet β 0.001 is the best config so far; the β trend is 0 < 0.01 < 0.001, so β = 0.003 is worth
a check. ACT stays at N = 2 whatever τ is. pg needs its halting entropy bonus (without it, compute
time first grows to 4.5 and then collapses), and a faster halting head hurts it.

### Round 3 (2026-11-08, 5e7 steps, 2 seeds, GPUs 5–7 next to the fixed-budget runs) — base lr 3e-4

Launcher: `/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch_haltloss_tune_r3.sh`. Output:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/haltloss_tune_r3/`.

| Config | GPU | Halting loss | Changes from defaults | Question |
|---|---|---|---|---|
| R3-pn-beta3e-3 | 5 | pondernet | lr 3e-4, β 0.003 | Is the best β between 0.001 and 0.01? |
| R3-pg-hent0.03 | 6 | pg | lr 3e-4, halting_ent_coef 0.03 | Removing the halting entropy bonus hurt; does a larger one help? |
| R3-act-bias-1 | 7 | act | lr 3e-4, halting_bias_init −1 | Does a mild negative bias unstick ACT without the return loss seen at −3? |

Results: pending.

### Fixed-budget reference (queued 2026-11-08, starts after round 2)

Launcher: `/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch_lightsout_cnn_fixed.sh`. Output:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/lightsout_cnn_fixed/`. Same CNN-encoder setup as the
halting-loss runs, run with `lightsout_sweep.py --budgets` (min_steps = max_steps = budget, so every
step is forced and the halting head gets no loss).

| Config | Budget | lr | Seeds | Steps |
|---|---|---|---|---|
| FB-1, FB-5 | 1, 5 (2–4 were stopped partway, 2026-11-08, to free GPUs) | 3e-4 | 2 (0–1) | 5e7 |

lr 3e-4 is the best lr for all three halting losses in round 1. 2 seeds × 5e7 steps matches the
tuning rounds, so the fixed budgets compare directly with every tuning config.

Results: pending.

Halting-loss search paused 2026-11-08 (user). Round 3 was stopped before it produced results.

## Architecture screen: does train/episode_return grow with fixed budget?

Metric: `evaluator/train/episode_return` (sampled actions, training distribution), mean of the last 5
logs; gap = budget 5 − budget 1. Launcher:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch_arch_budget_screen.sh`. Output:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/arch_budget_screen/`.

Evidence so far: no variant uses extra budget. Dense encoder, round 2, 1e8 steps, 5 seeds: budget 1
0.604 ± 0.018, budget 5 0.598 ± 0.005 (variable max 5: 0.600 ± 0.007). With the CNN encoder at budget 5,
the latent converges at about step 2.7 (`first_convergence_step`), and about 2 of the 4 later steps
are "close" to the step before. So later steps mostly repeat the same state. The stabilizers
(residual_scaling makes the per-step map contractive, input_injection, SEM, latent RMSNorm) are the
first suspects. num_layers stays at 2 (user, 2026-11-08), so smaller per-step networks are not tested. V7 and V8
run from `launch_arch_budget_screen_c.sh`.

Every variant: CNN-token encoder, lr 3e-4, budgets 1 and 5, 2 seeds, 5e7 steps; other settings as in
round-2 Lightsout. V0 is `lightsout_cnn_fixed` (budgets 1 and 5; 2–4 stopped). V2, V3 and V6 run
from `launch_arch_budget_screen_b.sh` on the GPUs freed by stopping budgets 2–4.

| Variant | GPUs | Changes from V0 | Question |
|---|---|---|---|
| V0 | — | none (SEM g8, latent RMSNorm, input injection, residual scaling, 2 layers, hd 128) | baseline |
| V1-noresscale | 5,6 | residual_scaling off | Is the contractive per-step map the cause? |
| V2-noinject | 1,2 | input_injection off | Does re-injecting the observation pin the state? |
| V3-nosem | 3 | sem_group_size 0 (no SEM) | Does the SEM readout limit what later steps add? |
| V4-plain | 7 | all four stabilizers off | Plain CoT transformer |
| ~~V5-1layer~~ | — | num_layers 1 | dropped before it started: keep num_layers = 2 (user) |
| ~~V6-hd64-1layer~~ | — | hidden_dim 64, num_layers 1 | stopped partway, same reason |
| V7-nolatentrms | 0,4 | latent RMSNorm off | The fourth stabilizer, by itself |
| V8-noresscale-noinject | 1,5 | residual_scaling and input_injection off (SEM kept) | Both fixed-point mechanisms off |

Results (train/episode_return @5e7, mean of the last 5 logs ± s.e. over 2 seeds). Plot for V0:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/analysis/v0.png`.

| Variant | Budget 1 | Budget 5 | Gap (5 − 1) | Eval (harder) gap |
|---|---|---|---|---|
| V0 | 0.735 ± 0.002 | 0.777 ± 0.036 | +0.042 (inside seed noise; the curves separate only after ~3.5e7) | about +0.08 (0.68 vs 0.60) |
| V1-noresscale | 0.634 ± 0.066 | 0.787 ± 0.004 | **+0.153** | |
| V2-noinject | 0.803 ± 0.015 | 0.646 ± 0.001 | −0.157 | |
| V3-nosem | 0.788 ± 0.006 | 0.805 ± 0.003 | +0.018 | |
| V4-plain | 0.766 ± 0.005 | 0.772 ± 0.002 | +0.007 | |
| V7-nolatentrms | 0.597 ± 0.020 | 0.605 ± 0.036 | +0.009 | |
| V8-noresscale-noinject | 0.708 ± 0.108 | 0.641 ± 0.061 | −0.066 | |

Plot: `/home/bryanpu1/scratch/logs/ramdp/sem-round2/analysis/screen_gap.png`. V1 has the largest gap, but
mostly because its budget-1 return is lower than V0's; its budget-5 return is about the same as V0's.
Removing input injection (V2, V8) makes budget 5 worse than budget 1. At budget 5 near the end of
training, V1 converges more than V0 (num close steps 1.39 vs 0.23 of 4), so the "fixed point" idea
above is not supported: V0's state keeps changing between steps, but this does not help.

### Residual scaling off, fixed budgets 1–5 (queued 2026-11-08, starts when the screen ends)

Setting chosen by the user from the screen (= V1): residual_scaling False; input_injection True,
latent_rms_norm True, SEM g8, num_layers 2, hd 128; CNN-token encoder; lr 3e-4. Launcher:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/launch_noresscale_fixed.sh`. Output:
`/home/bryanpu1/scratch/logs/ramdp/sem-round2/lightsout_cnn_noresscale_fixed/`.

| Budgets | Seeds | Steps | Runs |
|---|---|---|---|
| 1, 2, 3, 4, 5 | 5 (0–4) | 1e8 | 25 (2 waves on 16 slots) |

Results: pending.
