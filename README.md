# SPALF

SPALF trains sparse autoencoders with constrained optimization.

## Objective

SPALF minimizes discretization-corrected sparsity under three constraints:
- Faithfulness: whitened reconstruction error
- Drift: anchored decoder deviation from vocabulary
- Orthogonality: co-activation-weighted decoder overlap

Training uses an augmented Lagrangian with dual ascent and monotone penalty updates.

## Install

```bash
pip install -e .
```

Requirements: Python 3.10+, CUDA, PyTorch 2.4+.

## Train

```bash
spalf-train configs/pythia_1b.yaml
```

## Evaluate

```bash
spalf-eval configs/pythia_1b.yaml
```

`spalf-eval` requires `checkpoint` in the config.

## Config

Core knobs:
- `F`: dictionary width (`0` => auto `32*d_model`)
- `L0_target`: target active features (`null` => auto `ceil(F/400)`)
- `R2_target`: faithfulness target
- `lr`: Adam learning rate

Operational fields:
- `model_name`, `hook_point`, `dataset`, `batch_size`, `seq_len`, `total_tokens`
- `output_dir`, `checkpoint_interval`, `resume_from_checkpoint`
- `eval_suites`, `checkpoint`

See `configs/pythia_1b.yaml` and `configs/llama3_8b.yaml`.

## Runtime Logs

All runtime output is structured JSON via `print(json.dumps(...))`.

Typical events:
- `train_start`, `calibration_start`, `calibration_ready`
- `train_loop_start`, `train_step`, `kl_onset`, `checkpoint_saved`, `train_loop_complete`
- `checkpoint_loaded`, `eval_suite_start`, `eval_suite_summary`

## Checkpoints

Each checkpoint directory contains:
- `model.safetensors` (SAE weights)
- `optimizer.bin` and Accelerate state files
- `calibration.safetensors` (whitener + `W_vocab`)
- `metadata.json` (step + config + calibration metadata)
