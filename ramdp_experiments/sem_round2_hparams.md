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
