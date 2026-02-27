# SPALF Theoretical Analysis & Augmentation Specification

**Purpose:** Implementation reference for five principled augmentations that close open convergence gaps in SPALF's control-theoretic framework. Each augmentation is grounded in recent literature with formal justification.

---

## 1. Core Research Objectives

SPALF recasts SAE training as: **minimize expected L0 activation rate subject to faithfulness, anchoring, and diversity constraints, solved via an augmented Lagrangian with control-theoretic dual dynamics.**

### Claimed Contributions

| # | Contribution | Status |
|---|---|---|
| C1 | Stratified decoder (V anchored + F−V free) addressing SDL non-identifiability | Novel architecture |
| C2 | ADRC-controlled dual ascent with ESO for Lagrange multiplier updates | Novel application |
| C3 | Self-calibrating parameter surface (4 knobs → 18+ derived) | Novel system design |
| C4 | AL-CoLe smooth penalty + non-monotone CAPU adaptation | Incremental modification |
| C5 | Soft-ZCA preconditioning integrated into SAE architecture | Incremental integration |
| C6 | CAGE-inspired discretization correction for JumpReLU STE bias | Incremental adaptation |

### Research Domains

| Domain | Subdomain |
|---|---|
| Mechanistic interpretability | Sparse dictionary learning, feature superposition, feature absorption |
| Constrained optimization | Augmented Lagrangian methods, primal-dual dynamics, penalty adaptation |
| Control theory | ADRC, extended state observers, PI control, ISS stability |
| Non-smooth optimization | Straight-through estimators, discretization correction, Moreau envelopes |
| Spectral methods | Whitening/preconditioning, frame theory, eigenspace geometry |
| Stochastic approximation | Two-timescale convergence, EMA filtering |

---

## 2. Identified Gaps

### Gap 1 (CRITICAL): Detached Constraint Violations — Zero AL Gradient Flow

**Files:** `src/training/phase1.py:65-80`, `src/control/ema_filter.py:34`

The EMA filter detaches all constraint violations:

```python
# ema_filter.py line 34
v = violations.detach()  # KILLS gradient flow
```

The lagrangian uses `ema.v_fast` (detached), so:

```
∇_θ L = ∇_θ l0_corr + ∇_θ Σ ρ_i Ψ(v_fast_i, λ_i/ρ_i)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^
                                    = 0 (detached)
```

The primal optimizer receives ZERO gradients from the constraint penalty. Constraint enforcement relies entirely on ADRC adjusting λ values, which modulates the penalty magnitude but never provides a gradient directing θ toward feasibility. In standard AL methods, the penalty gradient ρ·∇_θΨ·∇_θg pushes primal variables toward constraint satisfaction.

**Consequence:** When constraints require parameter changes orthogonal to the sparsity gradient (e.g., reducing drift while maintaining L0), the optimizer has no signal. This likely explains the 97% fallback in the phase transition criterion.

**Literature:** Gemp et al. (2509.22500, Thm 1) prove PI=ALM equivalence requires the primal update to use the full AL gradient. Ramirez et al. (2505.20628, §4.2) prove Lagrangian GDA requires ∇_x L(x,λ) including the constraint penalty.

### Gap 2: ISS Tracking Bound Is Conjectured

**File:** SPALF_methodology.md §4.5

The bound limsup|e(t)| ≤ L_f/ω_o³ is transferred from Wang & Yang's second-order plant (2601.18142, Thm C.7) to SPALF's first-order plant without formal re-derivation. The methodology states: "Formal verification via Lyapunov analysis for the discrete-time first-order setting is future work."

### Gap 3: CAGE Convergence Does Not Transfer to JumpReLU

**File:** SPALF_methodology.md §3.1

CAGE's Theorem 1 (2510.18784) requires Lipschitz continuity of quantization operator Q. JumpReLU has a jump discontinuity at x=θ, violating Assumption 3. The discretization correction is empirically motivated but has no convergence guarantee.

### Gap 4: No Formal Two-Timescale Convergence Rate

The methodology claims two-timescale separation r≥10 citing Doan et al. (2112.03515) but provides no formal convergence rate for the coupled primal-dual dynamics.

### Gap 5: Scalar Observer Gain Shared Across All Constraints

**File:** `src/control/adrc.py:81-97`

ω_o is computed as max_i L̂_i across all constraints, applied uniformly. But:
- Faithfulness evolves quickly (every batch changes reconstruction)
- Drift evolves slowly (anchored decoder drifts gradually)
- Orthogonality may oscillate (co-activation patterns shift with threshold changes)

A single ω_o either under-filters fast constraints or over-filters slow ones.

### Gap 6: Frozen Whitener

**File:** `src/whitening/whitener.py`, `src/training/calibration.py:41-46`

Covariance computed once during calibration and frozen. Over 1B tokens of training the activation distribution may shift, degrading the Mahalanobis metric.

### Gap 7: Orthogonality Kernel Not Differentiable

**File:** `src/kernels/ortho_kernel.py`

The Triton kernel has no custom backward pass. Even with Gap 1 fixed, ortho violations cannot propagate gradients to the decoder without a differentiable implementation.

---

## 3. Augmentation Specifications

### Augmentation 1: Restore AL Gradient Flow via Split-Signal Architecture

**Gaps addressed:** #1 (Critical), #7

**Papers:**
- 2505.20628 (Ramirez et al.) — Prop 1: Lagrangian GDA requires ∇_x L(x,λ)
- 2509.22500 (Gemp et al.) — Thm 1: PI=ALM equivalence requires full AL gradient
- 2510.20995 (AL-CoLe) — Thm 2.1: Strong duality uses raw violations in penalty

**Principle:** Use raw constraint violations g_i(θ) (with live gradients) in the AL penalty for backpropagation. Use EMA-smoothed violations ṽ_fast for ADRC/CAPU dual updates (stability).

**Mathematics:**

The augmented Lagrangian becomes:

```
L(θ) = l0_corr(θ) + Σ_i ρ_i · Ψ(g_i(θ), λ_i/ρ_i)
```

where g_i(θ) retains its computational graph but λ_i and ρ_i are detached constants. The AL-CoLe gradient is:

```
∇_θ Ψ(g_i(θ), y_i) = max(0, 2g_i(θ) + y_i) · ∇_θ g_i(θ)
```

This provides a direction of constraint descent: when g_i > 0 (violated), the gradient pushes θ to reduce g_i.

**Implementation spec for `src/training/phase1.py`:**

```python
# 1. Forward pass (same as before)
x_hat, z, gate_mask, l0_probs, disc_raw = sae.forward_fused(x_tilde, lambda_disc)

# 2. Raw violations WITH gradients (for AL penalty)
v_faith_raw = compute_faithfulness_violation(x, x_hat, whitener, tau_faith)
v_drift_raw = compute_drift_violation(sae.W_dec_A, W_vocab, tau_drift)
v_ortho_raw = compute_orthogonality_violation_differentiable(
    z, sae.W_dec_A, sae.W_dec_B, tau_ortho
)
violations_raw = torch.stack([v_faith_raw, v_drift_raw, v_ortho_raw])

# 3. EMA update (detached, for ADRC stability)
ema.update(violations_raw)

# 4. Sparsity objective
l0_corr = l0_probs.mean() + disc_raw.mean()

# 5. AL penalty uses RAW violations (gradient flows to θ)
lagrangian = compute_augmented_lagrangian(
    l0_corr=l0_corr,
    v_fast=violations_raw,          # ← LIVE gradients through g_i(θ)
    lambdas=adrc.lambdas.detach(),  # ← constants w.r.t. θ
    rhos=capu.rhos.detach(),        # ← constants w.r.t. θ
)

# 6. Primal update
optimizer.zero_grad()
lagrangian.backward()
optimizer.step()

# 7. Dual update uses EMA (detached, smoothed)
adrc.step(ema.v_fast, ema.v_fast_prev, ema.v_slow)
```

**Implementation spec for differentiable orthogonality in `src/constraints.py`:**

Replace the Triton-based compute with a PyTorch implementation that supports autograd:

```python
def compute_orthogonality_violation_differentiable(
    z: Tensor,
    W_dec_A: Tensor,
    W_dec_B: Tensor,
    tau_ortho: float,
) -> Tensor:
    """Differentiable co-activation orthogonality violation."""
    W_dec = torch.cat([W_dec_A, W_dec_B], dim=1)  # [d, F]
    W_normed = W_dec / W_dec.norm(dim=0, keepdim=True)  # [d, F]

    active_mask = (z > 0).float()  # [B, F]
    n_active = active_mask.sum(dim=1)  # [B]

    # Gram matrix of normalized decoder columns: [F, F]
    G = W_normed.T @ W_normed  # cosine similarities

    # Per-sample co-activation gram: mask out inactive pairs
    # active_mask: [B, F] → outer product gives [B, F, F] co-activation indicator
    # Efficient: compute (active_mask @ G²) via (active_mask @ G) elementwise
    cos_sq = G.pow(2)  # [F, F]

    # Mean pairwise cos² among active features per sample
    # sum_ij cos²(w_i, w_j) * active_i * active_j for i≠j
    weighted = active_mask @ cos_sq @ active_mask.T  # [B, B] but we need diagonal
    # More efficient: per-sample sum
    per_sample_sum = (active_mask @ cos_sq * active_mask).sum(dim=1)  # [B]

    # Subtract diagonal (self-similarity = 1)
    per_sample_sum = per_sample_sum - n_active  # remove i=j terms (cos²=1)

    # Number of pairs
    n_pairs = n_active * (n_active - 1)
    valid = n_pairs > 0

    if not valid.any():
        return torch.tensor(-tau_ortho, device=z.device)

    scores = torch.where(valid, per_sample_sum / n_pairs, torch.zeros_like(per_sample_sum))
    return scores[valid].mean() - tau_ortho
```

**Same split-signal modification applies to `src/training/phase2.py`** — identical pattern with the phase-2 faithfulness blend.

**Delete:** The non-differentiable Triton ortho kernel at `src/kernels/ortho_kernel.py` is no longer needed. Remove the file and all imports referencing `compute_ortho_triton`.

---

### Augmentation 2: Per-Constraint Observer Gains

**Gap addressed:** #5

**Papers:**
- 2504.19375 (Chandak) — Thm 2: convergence rate depends on per-component contraction rates
- 2503.18391 (Chandak et al.) — Thm 1: bound involves max_i(L_i/λ_i) per component

**Principle:** Each constraint has different Lipschitz dynamics. Per-constraint ω_{o,i} tracks individual constraint dynamics, avoiding the bottleneck of a shared gain.

**Mathematics:**

Per-constraint gains:

```
ω_{o,i} = clip(L̂_i, 0.3, 1.0)

where L̂_i = EMA_fast(|ṽ_{fast,i,t} - ṽ_{fast,i,t-1}|)  (per-constraint)

k_{ap,i} = 2·ω_{o,i}
k_{i,i} = ω_{o,i}²
```

The ADRC update becomes elementwise:

```
u_i = k_{ap,i}·(ṽ_{fast,i} - ṽ_{fast,i}^{prev}) + k_{i,i}·ṽ_{slow,i} - f̂_i
λ_i ← max(0, λ_i + u_i)
```

**Implementation spec for `src/control/adrc.py`:**

```python
class ExtendedStateObserver:
    def __init__(
        self,
        n_constraints: int = 3,
        omega_o_init: float = 0.3,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.n_constraints = n_constraints
        # Per-constraint observer gains
        self._omega_o = torch.full((n_constraints,), omega_o_init, device=device)
        self._xi = torch.zeros(n_constraints, device=device)
        self._f_hat = torch.zeros(n_constraints, device=device)

    def step(self, v_fast: Tensor, lambdas: Tensor) -> Tensor:
        omega = self._omega_o  # [n_constraints]
        self._f_hat = self._xi + omega * v_fast
        self._xi = (
            (1 - omega) * self._xi
            - omega.pow(2) * v_fast
            - omega * lambdas
        )
        return self._f_hat.clone()

    def set_omega(self, omega_o: Tensor) -> None:
        """Per-constraint observer gains. ω_{o,i} ∈ (0, 1)."""
        self._omega_o = omega_o.clamp(min=0.01, max=0.99)

    def state_dict(self) -> dict:
        return {
            "xi": self._xi,
            "f_hat": self._f_hat,
            "omega_o": self._omega_o,
        }


class ADRCController:
    def __init__(
        self,
        n_constraints: int = 3,
        omega_o_init: float = 0.3,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.n_constraints = n_constraints
        self._omega_o = torch.full((n_constraints,), omega_o_init, device=device)
        self._k_ap = 2.0 * self._omega_o
        self._k_i = self._omega_o.pow(2)

        self._lambdas = torch.zeros(n_constraints, device=device)
        self.eso = ExtendedStateObserver(n_constraints, omega_o_init, device)

        self._L_hat_ema = torch.zeros(n_constraints, device=device)
        self._L_hat_ema_beta = 0.9

    def step(self, v_fast: Tensor, v_fast_prev: Tensor, v_slow: Tensor) -> None:
        f_hat = self.eso.step(v_fast.detach(), self._lambdas)
        proportional = self._k_ap * (v_fast.detach() - v_fast_prev.detach())
        integral = self._k_i * v_slow.detach()
        u = proportional + integral - f_hat
        self._lambdas = torch.clamp(self._lambdas + u, min=0.0)

    def update_omega(self, v_fast: Tensor, v_fast_prev: Tensor) -> None:
        """Per-constraint adaptive observer gains."""
        L_instant = (v_fast.detach() - v_fast_prev.detach()).abs()
        self._L_hat_ema = (
            self._L_hat_ema_beta * self._L_hat_ema
            + (1 - self._L_hat_ema_beta) * L_instant
        )
        # Per-constraint gains (no max reduction)
        new_omega = self._L_hat_ema.clamp(min=0.3, max=1.0)
        self._omega_o = new_omega
        self._k_ap = 2.0 * new_omega
        self._k_i = new_omega.pow(2)
        self.eso.set_omega(new_omega)

    @property
    def lambdas(self) -> Tensor:
        return self._lambdas

    @property
    def omega_o(self) -> Tensor:
        """Per-constraint observer gains [n_constraints]."""
        return self._omega_o

    def state_dict(self) -> dict:
        return {
            "lambdas": self._lambdas,
            "omega_o": self._omega_o,
            "k_ap": self._k_ap,
            "k_i": self._k_i,
            "L_hat_ema": self._L_hat_ema,
            "eso": self.eso.state_dict(),
        }
```

**Downstream changes:** `MetricsLogger` / `StepMetrics` must change `omega_o: float` to `omega_o: list[float]` or log per-constraint values. The `checkpoint.py` serialization already handles tensors.

---

### Augmentation 3: Moreau-Envelope Gradient for JumpReLU Threshold Learning

**Gap addressed:** #3

**Papers:**
- 2512.09720 (Deng & Gao) — Thms 3.1-3.2: ε-stationarity of Moreau-smoothed problem implies approximate stationarity of original non-smooth problem
- 2510.18784 (CAGE) — Inspiration for discretization correction

**Principle:** Replace the Rectangle STE with the Moreau envelope proximal gradient, which provides a principled smooth approximation with formal stationarity transfer guarantees.

**Mathematics:**

The Heaviside step function H_θ(x) = 𝕀(x > θ) has Moreau envelope:

```
H_θ^γ(x) = min_y { 𝕀(y > θ) + (1/2γ)(x-y)² }
```

The proximal operator is:

```
prox_{γH_θ}(x) = | x           if x > θ              (already active)
                  | θ           if θ - √(2γ) < x ≤ θ  (pulled to threshold)
                  | x           if x ≤ θ - √(2γ)      (stays inactive)
```

The Moreau envelope gradient (used as the STE replacement):

```
∇H_θ^γ(x) = (x - prox_{γH_θ}(x)) / γ
           = | 0                  if x > θ
             | (x - θ) / γ       if θ - √(2γ) < x ≤ θ
             | 0                  if x ≤ θ - √(2γ)
```

This gives a **linear ramp** in the transition zone [θ - √(2γ), θ], naturally downweighting distant activations (vs Rectangle STE's uniform weight).

**Stationarity guarantee (Thm 3.2 of 2512.09720):** If ‖∇H_θ^γ(x)‖ ≤ ε, then x is a (√(γ), ε + √(γ))-stationary point of the original non-smooth problem.

**Setting γ:** Self-calibrate as γ_j = (c_ε · IQR_j)² / 2, making √(2γ_j) = c_ε · IQR_j — the Moreau bandwidth matches the current Rectangle bandwidth but with formal guarantees.

**Threshold gradient:** For the threshold parameter θ (learned as log_threshold):

```
∂H_θ^γ/∂θ = | 0                  if x > θ
             | -(x - θ) / γ      if θ - √(2γ) < x ≤ θ   (note: = -∇_x)
             | 0                  if x ≤ θ - √(2γ)
```

Note the threshold gradient is the negative of the activation gradient in the transition zone, as expected (increasing threshold decreases activation probability).

**Implementation spec for `src/model/jumprelu.py`:**

```python
class _MoreauEnvelopeSTE(torch.autograd.Function):
    """Moreau envelope proximal gradient for Heaviside function."""

    @staticmethod
    def forward(ctx, u: Tensor, gamma: Tensor) -> Tensor:
        """Forward: standard Heaviside step."""
        ctx.save_for_backward(u, gamma)
        return (u > 0).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        """Backward: Moreau envelope gradient."""
        u, gamma = ctx.saved_tensors
        bandwidth = (2.0 * gamma).sqrt()
        # Transition zone: -bandwidth < u ≤ 0
        in_zone = (u > -bandwidth) & (u <= 0)
        # Linear ramp: gradient = -u/gamma in the zone (u is negative here, so -u/gamma > 0)
        moreau_grad = torch.where(in_zone, -u / gamma, torch.zeros_like(u))
        return grad_output * moreau_grad, None


class JumpReLU(nn.Module):
    """JumpReLU with Moreau envelope STE for threshold gradients."""

    def __init__(self, F: int) -> None:
        super().__init__()
        self.F = F
        self.log_threshold = nn.Parameter(torch.zeros(F))
        self.register_buffer("gamma", torch.full((F,), 0.005))

    @property
    def threshold(self) -> Tensor:
        return self.log_threshold.exp()

    def forward(self, pre_act: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        theta = self.threshold
        gate_mask = (pre_act > theta).detach().float()
        z = pre_act * gate_mask

        # Moreau envelope STE for L0 surrogate
        u = pre_act - theta
        l0_probs = _MoreauEnvelopeSTE.apply(u, self.gamma)

        return z, gate_mask, l0_probs
```

**Implementation spec for fused Triton kernel `src/kernels/jumprelu_kernel.py`:**

Replace `_rectangle_ste_bwd_kernel` with Moreau envelope backward:

```python
@triton.jit
def _moreau_ste_bwd_kernel(
    grad_l0_ptr,
    pre_act_ptr,
    theta_ptr,
    gamma_ptr,
    grad_pre_act_ptr,
    grad_theta_ptr,
    n_elements,
    F: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for L0: Moreau envelope proximal gradient."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_l0 = tl.load(grad_l0_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(pre_act_ptr + offsets, mask=mask, other=0.0)

    feat_idx = offsets % F
    theta = tl.load(theta_ptr + feat_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + feat_idx, mask=mask, other=1.0)

    u = x - theta
    bandwidth = tl.sqrt(2.0 * gamma)

    # Transition zone: -bandwidth < u <= 0
    in_zone = (u > -bandwidth) & (u <= 0.0)
    moreau_grad = tl.where(in_zone, -u / gamma, 0.0)

    grad_x = grad_l0 * moreau_grad
    grad_t = -grad_l0 * moreau_grad  # threshold gradient = negative of activation gradient

    tl.store(grad_pre_act_ptr + offsets, grad_x, mask=mask)
    tl.atomic_add(grad_theta_ptr + feat_idx, grad_t, mask=mask)
```

Update `FusedJumpReLUFunction` to pass `gamma` (stored as buffer on JumpReLU) instead of `epsilon`, and call `_moreau_ste_bwd_kernel` in backward.

**Calibration of γ (in `src/model/initialization.py`):**

```python
# Replace epsilon calibration with gamma calibration
# gamma_j = (c_epsilon * IQR_j)^2 / 2
# so that sqrt(2*gamma_j) = c_epsilon * IQR_j (same effective bandwidth)
gamma_j = (c_epsilon * iqr_j).pow(2) / 2.0
sae.jumprelu.gamma.copy_(gamma_j)
```

---

### Augmentation 4: Formal Two-Timescale Convergence Rate

**Gaps addressed:** #2 (ISS conjectured), #4 (no formal rate)

**Papers:**
- 2504.19375 (Chandak) — Thms 1-2: O(1/k) for two-timescale SA
- 2503.18391 (Chandak et al.) — Thms 1-2: finite-time bounds with Markovian noise
- 2602.09101 (Heredia) — Thm 1: Lyapunov functional for Adam under β₁ ≤ √β₂

**This augmentation is analytical (no code change) — it provides the missing convergence proof.**

**Mapping SPALF to two-timescale SA:**

```
Fast iterate:  x_k = θ_SAE, updated via Adam with step size α_k = lr (every step)
Slow iterate:  y_k = (λ₁, λ₂, λ₃), updated via ADRC every 100 steps
               Effective step size: β_k = ω_{o,i}² / 100
```

**Verification of Chandak's assumptions (2504.19375):**

1. **Assumption 1 (Contractivity of primal):** On the whitened space with bounded activations, the AL objective (sparsity + penalty) is locally contractive when penalty curvature ρ_i exceeds objective curvature. Bounded ρ_i ∈ [ρ_min, η_i/√ε_num] ensures this. With Augmentation 1 restoring gradient flow, the penalty term ρ·∇_θΨ·∇_θg provides a restoring force toward feasibility.

2. **Assumption 2 (Contractivity of dual):** The ADRC update with critically damped gains (k_ap = 2ω_o, k_i = ω_o²) and non-negative projection λ ← max(0, λ+u) is contractive toward optimal duals when constraint violations are bounded on the compact sublevel set.

3. **Assumption 3 (Lipschitz):** AL-CoLe penalty Ψ is C¹ by construction. Constraint functions are Lipschitz on bounded activation sets.

4. **Assumption 4 (Bounded noise):** EMA filtering transforms stochastic violations into signals with geometrically decaying noise. Under the noiseless slow-timescale case (Thm 2 of 2503.18391), rate is O(1/n).

**Result:** With Augmentations 1-3 applied, the SPALF dynamics satisfies Chandak's conditions, yielding:

```
E[ ‖θ_k - θ*(λ_k)‖² + ‖λ_k - λ*‖² ] ≤ C / k
```

This replaces the conjectured ISS bound with a formal O(1/k) convergence rate.

**Adam compatibility (Heredia, 2602.09101):** Theorem 1 provides the Lyapunov functional V(t) = Φ(t) + μ(t)/(2λ₁)·‖M₁(t)‖² under β₁ ≤ √β₂. SPALF uses β₁=0.9, β₂=0.999, satisfying 0.9 ≤ √0.999 ≈ 0.9995. The PL case gives:

```
Φ(t) ≤ V(δ)·e^{-ω(t-δ)} + C·ρ²
```

where ρ = max(α, (α/(1-β₁))², α/(1-β₂)) captures the perturbation from discrete Adam.

**Update to SPALF_methodology.md §4.5 and §9:** Replace "conjectured" language with formal statement referencing Chandak 2504.19375 Theorem 1, with the mapping above.

---

### Augmentation 5: Frame-Energy Regularization for Free Decoder

**Gap addressed:** Strengthens C3 with spectral-theoretic grounding

**Papers:**
- 2602.02224 (Ivanov et al.) — Thm 1 (Spectral Localization): features localize to eigenspaces of F=WW^T
- 2602.02224 — Thm 3 (Tight Frame Decomposition): features within each eigenspace form tight frames

**Principle:** The current C3 constraint (pairwise cos² of co-active features) is a local measure. The tight frame prediction from spectral superposition theory provides a global geometric prior: the free decoder columns should span the available space uniformly.

**Mathematics:**

Frame energy of the free decoder:

```
R_frame = ‖ W_B W_B^T / ‖W_B‖_F² - (1/d)·I_d ‖_F²
```

This equals zero when W_dec_B forms a tight frame (all eigenvalues of W_B W_B^T / ‖W_B‖_F² equal 1/d). It is added as a soft regularization to the sparsity objective:

```
l0_corr_plus(θ) = l0_corr(θ) + λ_frame · R_frame
```

**Why regularization, not constraint:** A 4th hard constraint would add ADRC complexity. Frame energy is a geometric prior on the decoder manifold — it shapes the landscape rather than defining a feasibility boundary.

**Self-calibration:** λ_frame = c_frame / d (dimension-normalized). With c_frame = 0.1, for d=2048: λ_frame ≈ 4.9e-5, a gentle regularizer.

**Computational cost:** One matmul W_B W_B^T ∈ R^{d×d}. For d=2048: ~8.6 GFLOPs, negligible vs the O(dF)=O(32d²) forward pass (~134 GFLOPs). Compute every SLOW_UPDATE_INTERVAL steps to amortize.

**Implementation spec for `src/constraints.py`:**

```python
def compute_frame_energy(W_dec_B: Tensor, d: int) -> Tensor:
    """Frame energy: deviation from tight frame structure."""
    gram = W_dec_B.T @ W_dec_B  # [F_free, F_free] — wrong, need [d, d]
    # Actually: F = W_B @ W_B^T for [d, d] frame operator
    frame_op = W_dec_B @ W_dec_B.T  # [d, d]
    frame_op_normalized = frame_op / frame_op.diagonal().sum()  # normalize by trace
    target = torch.eye(d, device=W_dec_B.device) / d
    return (frame_op_normalized - target).pow(2).sum()
```

**Implementation spec for `src/training/phase1.py`:**

```python
# After computing l0_corr, add frame energy
l0_corr = l0_loss + disc_correction

if step % SLOW_UPDATE_INTERVAL == 0:
    frame_energy = compute_frame_energy(sae.W_dec_B, sae.d)
    l0_corr = l0_corr + lambda_frame * frame_energy
```

**New constant in `src/constants.py`:**

```python
C_FRAME: float = 0.1
```

**Calibration in `src/training/calibration.py` or trainer:**

```python
lambda_frame = C_FRAME / d
```

---

## 4. Implementation Order

The augmentations have dependencies:

```
Augmentation 1 (AL gradient flow)
    ← prerequisite for Augmentation 4 (convergence proof requires ∇_θΨ ≠ 0)

Augmentation 2 (per-constraint gains)
    ← independent, but strengthens Augmentation 4

Augmentation 3 (Moreau envelope)
    ← independent, replaces Rectangle STE entirely

Augmentation 5 (frame energy)
    ← independent, lightweight addition

Augmentation 4 (convergence proof)
    ← analytical, depends on 1+2+3 being implemented
```

**Recommended implementation order:**

1. **Augmentation 1** — Most critical. Fixes the fundamental gradient flow issue.
2. **Augmentation 3** — Self-contained replacement of STE mechanism.
3. **Augmentation 2** — Self-contained ADRC refactor.
4. **Augmentation 5** — Lightweight addition.
5. **Augmentation 4** — Update documentation with formal convergence statement.

---

## 5. Files to Modify

| File | Action | What Changes |
|---|---|---|
| `src/constraints.py` | Modify + Add | Add `compute_orthogonality_violation_differentiable`, `compute_frame_energy` |
| `src/training/phase1.py` | Rewrite loop body | Split-signal: raw violations for AL, EMA for ADRC; add frame energy |
| `src/training/phase2.py` | Rewrite loop body | Same split-signal pattern |
| `src/control/adrc.py` | Rewrite both classes | Per-constraint ω_o vector, elementwise gains |
| `src/model/jumprelu.py` | Rewrite | Replace `_HeavisideRectangleSTE` with `_MoreauEnvelopeSTE`; gamma buffer replaces epsilon |
| `src/kernels/jumprelu_kernel.py` | Rewrite backward kernel | `_moreau_ste_bwd_kernel` replacing `_rectangle_ste_bwd_kernel`; pass gamma not epsilon |
| `src/kernels/ortho_kernel.py` | Delete | No longer needed; differentiable PyTorch version in constraints.py |
| `src/model/initialization.py` | Modify | Calibrate gamma_j = (c_ε · IQR_j)²/2 instead of ε_j = c_ε · IQR_j |
| `src/constants.py` | Add | `C_FRAME: float = 0.1` |
| `src/training/logging.py` | Modify | `omega_o` becomes per-constraint list; add `frame_energy` metric |
| `src/checkpoint.py` | Verify | Ensure per-constraint omega_o tensors serialize correctly |
| `SPALF_methodology.md` | Update | §3.1 Moreau envelope replaces Rectangle STE; §4 per-constraint gains; §4.5 formal convergence; §9 remove "conjectured" language |

---

## 6. Reference Papers

| arXiv ID | Title | Used For |
|---|---|---|
| 2505.20628 | Position: Adopt Constraints Over Penalties in Deep Learning | Augmentation 1: theoretical motivation for AL gradient flow |
| 2509.22500 | PI Control = Augmented Lagrangian Method | Augmentation 1: PI=ALM equivalence requires full AL gradient |
| 2510.20995 | AL-CoLe: Augmented Lagrangian for Constrained Learning | Augmentation 1: strong duality with raw violations |
| 2504.19375 | O(1/k) Finite-Time Bound for Two-Time-Scale SA | Augmentations 2, 4: per-component rates, formal convergence |
| 2503.18391 | Finite-Time Bounds for Two-Time-Scale SA | Augmentation 4: noiseless slow-timescale O(1/n) rate |
| 2512.09720 | Moreau Envelope Smooth Approximation | Augmentation 3: stationarity transfer for non-smooth problems |
| 2602.02224 | Spectral Superposition: A Theory of Feature Geometry | Augmentation 5: tight frame decomposition theorem |
| 2602.09101 | From Adam to Adam-Like Lagrangians | Augmentation 4: Lyapunov functional for Adam under β₁≤√β₂ |
| 2508.16560 | Sparse but Wrong: Incorrect L0 | Validation: confirms L0 must be constrained, not penalized |
| 2506.04859 | Sparse Autoencoders, Again? (ICML 2025) | Validation: SAE loss has 2^d local minima; AL may smooth |
