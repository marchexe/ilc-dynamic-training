# Online training controller

Weaver can optionally run a small hyperparameter controller after optimizer
steps.  The controller is separate from the model, optimizer, data loader, and
validation code.  It receives model-independent batch observations and may
change only explicitly supported optimizer settings.

The first implementation is `linucb_lr`, a contextual bandit that chooses a
multiplicative learning-rate action.  Its context contains training progress,
exponentially-smoothed loss, loss slope, gradient norm, and the current learning
rate.  The next window's relative loss improvement is used as its online
reward.

This is an experimental foundation for proxy-validation rewards, not evidence
that training-loss reward is already superior to a conventional scheduler.
Use `observe_only: true` first: decisions and proposed learning rates are logged
without changing training. Rewards are not attributed to hypothetical actions,
so the bandit starts learning only after `observe_only` is disabled.

## Usage

Start from the example configuration:

```bash
weaver ... \
  --seed 12345 \
  --lr-scheduler none \
  --training-controller examples/training_control/linucb_lr.yaml
```

For experiments with different epoch sizes, controller timing can be expressed
as fractions of one epoch:

```yaml
warmup_fraction: 0.10
interval_fraction: 0.05
```

These values are resolved from `--steps-per-epoch`.  Use either the fractional
form or `warmup_steps` / `interval_steps`, not both.  `--seed` also seeds the
model RNGs and explicit train/validation data-loader generators so paired runs
receive reproducible initialization and worker seeds.

Events are written as JSON Lines next to the model checkpoint by default.  Each
epoch checkpoint also stores the controller state, so `--load-epoch` resumes
the learned bandit state together with model and optimizer states.

## Boundaries

- The controller and a Weaver learning-rate scheduler cannot be enabled at the
  same time.
- Only learning rate is mutable in the initial implementation.
- `min_lr` and `max_lr` are hard safety bounds.
- A fixed proxy-validation metric should replace training-loss reward before
  physics-performance claims are made.
