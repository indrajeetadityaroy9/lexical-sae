# SPALF: Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds

**Authoritative Implementation Reference — Consolidated & Corrected**

---

## 0. Overview

SPALF trains interpretable sparse autoencoders by solving a single constrained optimization problem:

> **Minimize the expected activation rate of a JumpReLU dictionary over a spectrally-preconditioned activation space, subject to faithfulness, anchoring, and diversity constraints, solved via an augmented Lagrangian whose dual dynamics use active disturbance rejection.**

The six components — Soft-ZCA whitening, stratified decoder, CAGE-inspired threshold correction, ADRC-controlled dual ascent, modified CAPU penalty adaptation, and two-phase training — are unified under this formulation. The system is configured by **4 user knobs**; all other parameters are self-calibrated from data.

**Central Lagrangian:**
$$\mathscr{L}(\theta, \lambda, \rho) = \ell_0^{corr}(\theta) + \sum_{i=1}^{3} \rho_i \cdot \Psi\!\left(g_i(\theta),\; \frac{\lambda_i}{\rho_i}\right)$$

where $\theta = (W_{enc}, W_{dec}, \vartheta, b)$ are primal variables, $\lambda \in \mathbb{R}^3_+$ are dual variables governed by ADRC, $\rho \in \mathbb{R}^3_+$ are per-constraint penalties, $\ell_0^{corr}$ is the discretization-corrected sparsity objective, and $\Psi$ is AL-CoLe's smooth inequality penalty.

---

## 1. Spectrally-Preconditioned Activation Space

### 1.1 Motivation

Transformer residual stream activations $x \in \mathbb{R}^d$ are anisotropic: their covariance $\Sigma = \mathbb{E}[(x-\mu)(x-\mu)^\top]$ has condition numbers $\kappa(\Sigma) \sim 10^3$–$10^5$ ([2508.16929], [2511.13981]). This corrupts encoder dot products (dominated by high-variance directions), threshold calibration (variance varies by orders of magnitude across features), and constraint scales (MSE, drift, orthogonality become incommensurable). Whitening eliminates these pathologies simultaneously.

### 1.2 Soft-ZCA Whitening

**Definition (Whitening Map).** Given the activation covariance eigendecomposition $\Sigma = U\Lambda U^\top$ and regularization $\alpha > 0$:

$$\phi(x) = W_{white}(x - \mu), \quad W_{white} := U(\Lambda + \alpha I)^{-1/2} U^\top$$

**Properties ([2411.17538], [1804.08450]):**
- At $\alpha = 0$: full ZCA whitening ($\Sigma^{-1/2}$). At $\alpha \to \infty$: identity (no whitening).
- ZCA preserves maximal alignment with the original coordinate system, unlike PCA whitening ($\Lambda^{-1/2}U^\top$). This ensures vocabulary directions remain geometrically meaningful after whitening.
- The map is an isometry from $(\mathbb{R}^d, G_\alpha)$ to $(\mathbb{R}^d, I)$ where $G_\alpha = U(\Lambda + \alpha I)^{-1}U^\top$ is the regularized Mahalanobis metric: $\phi(u)^\top\phi(v) = u^\top G_\alpha v$.

**Self-Calibrating $\alpha$:** Set $\alpha = \lambda_k / \kappa_{target}$ where $k = \min\{k : \sum_{i \leq k} \lambda_i / \text{tr}(\Lambda) \geq 1 - \delta_{rank}\}$ and $\kappa_{target} = 100$. This bounds the effective condition number of the whitened space to approximately $\kappa_{target}$.

**Low-Rank Variant (for $d > 4096$):** Retain only the top-$k$ eigenvectors:
$$\phi_k(x) = U_k(\Lambda_k + \alpha I_k)^{-1/2} U_k^\top (x - \mu) + \frac{1}{\sqrt{\bar{\lambda} + \alpha}} P_\perp (x - \mu)$$
where $P_\perp = I - U_k U_k^\top$ and $\bar{\lambda} = \frac{1}{d-k}\sum_{i>k} \lambda_i$. Storage: $O(dk)$ vs. $O(d^2)$.

### 1.3 Role of the Metric in the Algorithm

The whitening defines the geometry of the encoder space. After whitening, all forward-pass operations occur in the isotropic space $\tilde{x} = \phi(x)$. The **faithfulness constraint** (C1) measures reconstruction error as standard MSE in this whitened space, which is equivalent to Mahalanobis-norm error in the original space. The **drift constraint** (C2) and **orthogonality constraint** (C3) operate in the original coordinate system using Frobenius and Euclidean cosine norms respectively — this is deliberate: drift is measured against $W_{vocab}$ which lives in the original space, and decoder column angles are geometric properties independent of the activation metric.

---

## 2. Stratified Feature Manifold

### 2.1 Motivation

The SDL non-identifiability theorem ([2512.05534], Theorem 3.4) proves that zero-loss solutions exist that recover no ground-truth features. Empirical results in [2512.05534] (Table 4) demonstrate that partial anchoring — constraining a subset of dictionary atoms to known reference directions — steers optimization away from spurious minima, with 12.5% anchoring achieving 100% ground-truth recovery in synthetic settings. However, the vocabulary size $V$ and dictionary size $F = 32d$ satisfy $V \not\approx F$ in general, requiring a partitioned architecture.

**Vocabulary scaling note:** For large-vocabulary models (e.g., Llama-3-8B with $V \approx 128$K, $F \approx 131$K), the anchored stratum consumes most of the dictionary. In such cases, increase $F$ beyond $32d$ (e.g., $F = 64d$) to ensure sufficient free capacity, or cap the anchored stratum at the top-$V_{cap}$ most frequent tokens.

### 2.2 Definition

The decoder $W_{dec} \in \mathbb{R}^{d \times F}$ is partitioned into two strata:

**Stratum A — Anchored ($V$ features):**
$$\mathcal{M}_A = \left\{ W_{dec}^{(A)} \in \mathbb{R}^{d \times V} : \|W_{dec}^{(A)} - W_{vocab}\|_F^2 \leq \tau_{drift} \right\}$$
- Initialized to $W_{vocab}$ (unembedding matrix columns)
- Drift budget: $\tau_{drift} = \delta_{drift}^2 \cdot \|W_{vocab}\|_F^2$ ($\delta_{drift} = 0.1$, scale-invariant)
- Provides vocabulary-grounded, token-level interpretability

**Stratum B — Free ($F - V$ features):**
$$\mathcal{M}_B = \left\{ W_{dec}^{(B)} \in \mathbb{R}^{d \times (F-V)} : \|w_j\|_2 = 1 \;\forall j \right\}$$
- Initialized as random unit-norm columns $\sim \mathcal{N}(0, I/d)$
- Not subject to drift constraint; regulated by orthogonality constraint
- Captures abstract, compositional, multi-dimensional features

The full decoder is $W_{dec} = [W_{dec}^{(A)} \;|\; W_{dec}^{(B)}]$.

### 2.3 Matched-Filter Encoder Initialization

The encoder must invert the whitening in its initial dot products. For anchored features:

$$w_{vocab}^\top (x - \mu) = w_{enc,init}^\top \cdot W_{white}(x - \mu) \implies w_{enc,init}^\top = w_{vocab}^\top W_{white}^{-1}$$

Therefore:
$$W_{enc}^{(A),init} = W_{vocab}^\top \cdot U(\Lambda + \alpha I)^{+1/2} U^\top \in \mathbb{R}^{V \times d}$$

The **positive** exponent $+1/2$ undoes the whitening's variance suppression, ensuring the encoder's initial response in $\tilde{x}$-space reproduces original-space vocabulary correlations. For free features:

$$W_{enc}^{(B),init} \in \mathbb{R}^{(F-V) \times d}: \text{rows} \sim \mathcal{N}(0, I/d), \text{Gram-Schmidt against } W_{enc}^{(A)}$$

In the overcomplete regime ($F - V \gg d - V$), only the first $d - V$ free rows can be exactly orthogonal; remaining rows are random unit-norm vectors.

### 2.4 Justification

- [2512.05534] (Theorem 3.4) proves SDL non-identifiability; empirical results (Table 4) show partial anchoring restores recovery
- [2501.17727] shows many SAE features are inherently lexical, validating the anchored stratum
- [2602.02385] shows transformers learn factored representations with orthogonal subspaces, validating the free stratum
- [2409.14507] identifies feature absorption from rigid anchoring; the drift budget $\tau_{drift}$ allows controlled departure

---

## 3. Constrained Optimization Problem

### 3.1 Primal Objective: Discretization-Corrected Sparsity

**JumpReLU encoding:**
$$z = \text{JumpReLU}_\vartheta(\tilde{x}) := (W_{enc}\tilde{x} + b_{enc}) \cdot \mathbb{I}(W_{enc}\tilde{x} + b_{enc} > \vartheta)$$

**The non-smoothness problem:** The $L_0$ rate $\|z\|_0$ and JumpReLU operator $Q(\tilde{x}; \vartheta) = \tilde{x} \cdot \mathbb{I}(\tilde{x} > \vartheta)$ are piecewise constant in $\vartheta$, providing zero gradient.

**STE-smoothed $L_0$ (from [2407.14435]):**
$$\ell_{L_0} = \frac{1}{N}\sum_{n,j} \Phi\!\left(\frac{\tilde{x}_{n,j} - \vartheta_j}{\epsilon_j}\right)$$
where $\Phi$ is the Heaviside step function in the forward pass, with gradients routed through the Rectangle STE kernel $\mathbb{I}(|\tilde{x} - \vartheta| < \epsilon_j)$ in the backward pass. Per-feature adaptive bandwidth $\epsilon_j = c_\epsilon \cdot \text{IQR}_j$ replaces the fixed $\epsilon = 0.01$, providing ~13.5$\times$ more gradient coverage on whitened activations.

**Discretization correction (SPALF contribution, inspired by [2510.18784]):** CAGE ([2510.18784]) demonstrates that adding a discretization penalty $\lambda \cdot \|x - Q(x)\|^2$ to the loss eliminates the persistent STE bias. CAGE itself uses a fixed hyperparameter $\lambda$ with a deterministic linear ramp schedule. We adapt this principle with a **scheduled discretization penalty**:

$$\ell_0^{corr}(\theta) = \ell_{L_0} + \lambda_{disc}(t) \sum_j \|\tilde{x}_j - Q(\tilde{x}_j; \vartheta_j)\|^2$$

where the discretization weight follows a linear ramp:
$$\lambda_{disc}(t) = \begin{cases} 0 & \text{if } t/T \leq s_{disc} \\ \lambda_{disc}^{max} \cdot \frac{t/T - s_{disc}}{1 - s_{disc}} & \text{if } t/T > s_{disc} \end{cases}$$

with silence ratio $s_{disc} = 0.8$ (no correction for the first 80% of training) and $\lambda_{disc}^{max} = 1.0$.

**Theoretical status:** CAGE's convergence theorem (Theorem 1, [2510.18784]) provides $O(1/\sqrt{T})$ convergence to $\lambda$-Pareto optimal points for **Lipschitz-continuous** quantization operators. JumpReLU has a jump discontinuity at $x = \vartheta$, violating CAGE's Assumption 3 (Lipschitz continuity of $Q$). Therefore, the formal convergence guarantee does **not** directly transfer. The discretization correction is motivated by CAGE's empirical success and the general principle that penalizing $\|x - Q(x)\|^2$ reduces the STE's persistent bias, but its convergence for JumpReLU must be validated empirically. The late-onset schedule ($s_{disc} = 0.8$) ensures the correction does not interfere with early-stage feature discovery.

### 3.2 Constraint Set

Three inequality constraints encode structural priors ($g_i(\theta) \leq 0$):

**C1 — Faithfulness (Reconstruction Quality):**
$$g_{faith}(\theta) = \frac{1}{N}\sum_n \|\tilde{x}_n - \widetilde{\hat{x}}_n\|_2^2 - \tau_{faith}$$
where $\widetilde{\hat{x}} = W_{white}(\hat{x} - \mu)$ is the reconstruction in whitened space. This is equivalent to $\|x - \hat{x}\|_{G_\alpha}^2$ (Mahalanobis norm in original space). Threshold: $\tau_{faith} = (1 - R^2_{target}) \cdot d$.

**C2 — Vocabulary Drift (Anchoring Fidelity):**
$$g_{drift}(\theta) = \|W_{dec}^{(A)} - W_{vocab}\|_F^2 - \tau_{drift}$$
Uses the Frobenius norm in the original coordinate system, since $W_{vocab}$ is defined in original coordinates. Threshold: $\tau_{drift} = \delta_{drift}^2 \cdot \|W_{vocab}\|_F^2$.

**C3 — Co-Activation Orthogonality (Feature Diversity):**
$$g_{ortho}(\theta) = \frac{1}{|B|}\sum_{b=1}^{B} \left(\frac{1}{|A_b|^2 - |A_b|} \sum_{\substack{i,j \in A_b \\ i \neq j}} \cos^2(w_{dec}^{(i)}, w_{dec}^{(j)})\right) - \tau_{ortho}$$
where $A_b = \{j : z_{b,j} > 0\}$ spans both strata. Uses standard Euclidean cosine similarity. Threshold: $\tau_{ortho} = c_{ortho}/d$ ($c_{ortho} = 3$, i.e., $3\times$ the random baseline $\mathbb{E}[\cos^2] = 1/d$). Since $|A_b| \approx 50$–$100$, this is $O(L_0^2)$, negligible vs. the $O(dF)$ forward pass.

**Sufficiency:** Together, C1 ensures faithfulness, C2 ensures a subset of features are identifiable (breaking SDL non-identifiability), and C3 prevents feature collapse/absorption. Random SAEs ([2602.14111]) satisfy none of these meaningfully.

### 3.3 Smooth Augmented Lagrangian (AL-CoLe Penalty)

The constraint penalty uses AL-CoLe's smooth function ([2510.20995], Eq. 3):

$$\Psi(x, y) = \frac{\max\{0,\; 2x + y\}^2 - y^2}{4}$$

**Verified properties:**
- $\Psi(x, y) \leq 0$ when $x \leq 0$ (inactive constraints contribute non-positively). *Proof:* When $x \leq 0$ and $2x + y > 0$: $\Psi = ((2x+y)^2 - y^2)/4 = x(x+y) \leq 0$ since $x \leq 0$ and $x + y \geq 0$.
- $\nabla_x \Psi(x, y) = \max\{0, 2x+y\}/2$ (everywhere continuously differentiable)
- At $x = 0, y > 0$: $\Psi(0, y) = 0$ (smooth transition at the boundary)

**Why $\Psi$ instead of standard $\frac{\rho}{2}\max(0, g)^2$:** The standard quadratic penalty has a kink at $g = 0$ that creates spurious high-frequency content in the constraint signal. The smooth $\Psi$ function eliminates this, which is important for:
1. The ADRC observer receiving clean signals
2. Adam's momentum/variance estimates not being corrupted by gradient discontinuities

**Combined Lagrangian:**
$$\boxed{\mathscr{L}(\theta, \lambda, \rho) = \ell_0^{corr}(\theta) + \sum_{i=1}^{3} \rho_i \cdot \Psi\!\left(g_i(\theta),\; \frac{\lambda_i}{\rho_i}\right)}$$

**Duality note:** AL-CoLe's strong duality theorem (Theorem 2.1, [2510.20995]) is proven for a single shared penalty $\alpha$ across all constraints and AL-CoLe's specific dual update rule. SPALF uses per-constraint $\rho_i$ and ADRC dual updates, so AL-CoLe's theorem does not apply directly. The underlying Rockafellar (1974) theory supports per-constraint penalties, and we adopt $\Psi$ for its smoothness properties rather than claiming AL-CoLe's convergence guarantees transfer. SPALF's convergence relies on the ADRC ISS tracking analysis (§4) and CAPU penalty boundedness (§5) independently.

---

## 4. Dual Dynamics: ADRC-Controlled Multiplier Updates

### 4.1 Conceptual Foundation: PI = ALM

The PI = ALM equivalence ([2509.22500], Theorem 1) establishes that dual optimistic ascent with proportional gain $\omega$ generates identical primal iterates to the augmented Lagrangian method with penalty $c = \omega$, under single-step first-order primal updates with strict complementary slackness. This provides the key insight: **proportional gain in the controller corresponds to penalty strength in the ALM**.

SPALF uses this insight as **conceptual motivation**, not as a formal guarantee, because three departures break the strict equivalence:
1. **Adam momentum:** Multi-step implicit primal updates (standard in constrained deep learning, [2510.20995], [2406.04558])
2. **EMA filtering:** Constraint violations are smoothed before entering the dual update, introducing history-dependent dynamics
3. **ESO cancellation:** The disturbance rejection term $-\hat{f}_i$ has no analog in the PI = ALM framework

### 4.2 Extended State Observer (ESO)

**Adaptation from [2601.18142]:** The ADRC paper models a **second-order** plant ($\ddot{x}_1 = f - \lambda$) with a reduced-order ESO. SPALF models constraint dynamics as **first-order** ($\dot{\lambda}_i = u_i + f_i$), where $f_i$ is the lumped disturbance (stochastic noise, constraint coupling, nonstationarity). The ESO equations are algebraically identical but semantically re-mapped: the observed variable $\tilde{v}_{fast,i}$ plays the role of the paper's $x_2$, and $\lambda_i$ plays the role of the control input.

**ESO Update (discrete-time, $\Delta t = 1$):**
$$\xi_{i,t+1} = (1 - \omega_o)\xi_{i,t} - \omega_o^2 \cdot \tilde{v}_{fast,i,t} - \omega_o \cdot \lambda_{i,t}$$
$$\hat{f}_{i,t} = \xi_{i,t} + \omega_o \cdot \tilde{v}_{fast,i,t}$$

where $\xi_i$ is the auxiliary observer state, $\hat{f}_i$ is the estimated disturbance, and $\omega_o \in (0, 1)$ is the observer gain (discrete stability requires $\omega_o < 1$).

**Estimation Error Bound ([2601.18142], Lemma C.6):**
$$|\hat{f}_{i,t} - f_{i,t}| \leq e^{-\omega_o t} |\hat{f}_{i,0} - f_{i,0}| + \frac{L_f}{\omega_o}$$
where $L_f = \sup_t |\dot{f}_i(t)|$. This bound is proven for [2601.18142]'s second-order system; its transfer to SPALF's first-order setting is conjectured based on the algebraic identity of the observer dynamics and must be validated empirically.

### 4.3 ADRC Multiplier Update

$$u_{i,t} = \underbrace{k_{ap}(\tilde{v}_{fast,i,t} - \tilde{v}_{fast,i,t-1})}_{\text{Proportional (transient)}} + \underbrace{k_i \cdot \tilde{v}_{slow,i,t}}_{\text{Integral (steady-state)}} - \underbrace{\hat{f}_{i,t}}_{\text{ESO cancellation}}$$
$$\lambda_{i,t+1} = \max(0, \;\lambda_{i,t} + u_{i,t})$$

**Gain Parameterization (ADRC bandwidth):**
$$k_{ap} = 2\omega_o, \quad k_i = \omega_o^2$$

For the second-order closed-loop arising from the first-order plant + integral controller, this yields the characteristic polynomial $(s + \omega_o)^2 = s^2 + 2\omega_o s + \omega_o^2$ — critically damped response (no oscillation, fastest non-oscillatory convergence).

**Self-Calibrating Observer Gain (SPALF heuristic):** Inspired by the gain-selection framework in [2601.18142] (Eq. 20), which involves solving a quartic polynomial with Lipschitz constants, we use a simplified adaptive rule:
$$\omega_o = \min\!\left(\max\!\left\{0.3,\; \hat{L}\right\},\; 1.0\right)$$
where $\hat{L}_t = \text{EMA}_{fast}(|\tilde{v}_{fast,i,t} - \tilde{v}_{fast,i,t-1}|)$ is the online Lipschitz estimate of the constraint dynamics, updated every 100 steps. The lower bound 0.3 is a conservative initialization; the upper bound 1.0 ensures discrete stability. This is a practical heuristic, not a theorem-derived formula.

### 4.4 Dual-Rate EMA Filtering

The constraint violation signal $v_{i,t} = g_i(\theta_t)$ is filtered at two timescales:
$$\tilde{v}_{fast,i,t} = 0.9 \cdot \tilde{v}_{fast,i,t-1} + 0.1 \cdot v_{i,t}$$
$$\tilde{v}_{slow,i,t} = 0.99 \cdot \tilde{v}_{slow,i,t-1} + 0.01 \cdot v_{i,t}$$

The fast signal ($\beta_{fast} = 0.9$, ~10-step smoothing) feeds the ESO and proportional term. The slow signal ($\beta_{slow} = 0.99$, ~100-step smoothing) feeds the integral accumulator. The timescale separation ratio $r = (1-\beta_{fast})/(1-\beta_{slow}) = 10$ satisfies the minimum $r \geq 10$ for two-timescale stochastic approximation convergence ([2112.03515]).

### 4.5 ISS Tracking Bound (Conjectured)

**From [2601.18142], Theorem C.7 (second-order system):** Under critically damped gains with $k_{ap}^2 \geq 4k_i$ (holds for $k_{ap} = 2\omega_o$, $k_i = \omega_o^2$), the $L^1$ norm of the impulse response is $\|h\|_{L^1} = 1/k_i = 1/\omega_o^2$, giving:

$$\limsup_{t \to \infty} |e(t)| \leq \frac{L_f}{\omega_o^2 \cdot \omega_o} = \frac{L_f}{\omega_o^3}$$

This bound was proven for [2601.18142]'s second-order plant. For SPALF's first-order plant + PI controller (which creates a second-order closed-loop), the bound is **conjectured** to hold with similar structure. Formal verification via Lyapunov analysis for the discrete-time first-order setting is future work. Practically, with $\omega_o = 0.5$ and $L_f \approx 0.1$, the tracking tube is $\approx 0.8$.

### 4.6 Advantages over PID

- Lumped disturbance rejection handles stochastic noise and constraint coupling simultaneously
- Principled gain structure (bandwidth parameterization) vs. three independent PID gains
- PID and standard gradient ascent are recoverable as special cases ([2601.18142], Proposition 4.1)
- The ESO replaces the noisy finite-difference D-term with a model-based estimate

---

## 5. Per-Constraint Adaptive Penalty (Modified CAPU)

### 5.1 Motivation

The Boyd doubling/halving heuristic for $\rho$ has two flaws: uniform scaling lets one poorly-conditioned constraint dominate, and monotone increase eventually freezes the primal variables.

### 5.2 Formulation (SPALF Modification of CAPU)

The original CAPU method ([2508.15695], Eq. 10) uses a **monotone** update: $\mu_i \leftarrow \max(\mu_i, \eta_i/\sqrt{\bar{v}_i + \epsilon})$, where the max against the previous value creates a ratchet that can only increase penalties. SPALF modifies this to a **non-monotone** variant that allows penalty relaxation:

For each constraint $i$, maintain a running second moment:
$$\bar{v}_{i,t} = \beta_{slow} \cdot \bar{v}_{i,t-1} + (1-\beta_{slow}) \cdot \tilde{v}_{fast,i,t}^2$$

Compute the adaptive penalty with a **fixed floor** (not the previous value):
$$\rho_{i,t} = \max\!\left(\rho_{min},\; \frac{\eta_i}{\sqrt{\bar{v}_{i,t} + \epsilon_{num}}}\right)$$

where:
- $\eta_i = c_\eta / \sqrt{|v_{i,0}| + \epsilon_{num}}$ is self-calibrated from the initial violation ($c_\eta = 1.0$)
- $\rho_{min} = 0.1 \cdot \rho_0$ prevents complete deactivation
- The max against $\rho_{min}$ (not $\rho_{i,t-1}$) allows $\rho_i$ to **decrease** when violations grow, preventing runaway penalties

**Key difference from original CAPU:** When violations are large, $\bar{v}_i$ increases and $\rho_i^{target}$ decreases (RMSprop-style normalization), preventing the penalty from dominating the objective. When violations are small, $\bar{v}_i$ decreases and $\rho_i$ increases, tightening enforcement. The original CAPU's monotone ratchet would prevent this relaxation.

**Convergence note:** CAPU's Theorem 1 ([2508.15695]) proves multiplier convergence under the monotone update. The non-monotone variant loses this guarantee, though boundedness is preserved by the floor $\rho_{min}$ and the fact that $\bar{v}_i > 0 \implies \rho_i < \eta_i/\sqrt{\epsilon_{num}} < \infty$.

### 5.3 Interaction with ADRC

CAPU and ADRC operate on complementary aspects: CAPU adapts the penalty surface curvature while ADRC adapts the dual trajectory. Both update on the slow timescale (every 100 steps), maintaining two-timescale separation from primal updates.

---

## 6. Two-Phase Training

### 6.1 Phase 1 — Constrained Sparse Optimization (Convergence-Gated)

Reconstruction is treated as a **constraint** (C1), not a base objective. The base objective is sparsity minimization $\ell_0^{corr}$; reconstruction quality is enforced via the faithfulness multiplier $\lambda_{faith}$:

$$\mathscr{L} = \ell_0^{corr}(\theta) + \sum_{i \in \{faith, drift, ortho\}} \rho_i \cdot \Psi\!\left(\tilde{v}_{fast,i},\; \frac{\lambda_i}{\rho_i}\right)$$

MSE enters the loss only through the $\Psi$ penalty on $\tilde{v}_{fast,faith}$. No transformer forward pass is needed beyond the SAE's layer.

### 6.2 Phase Transition

Enter Phase 2 when $\tilde{v}_{fast,i} < 0$ for all $i$ for 100 consecutive steps (all constraints satisfied under the slow EMA). Fallback: transition at $0.97 \cdot T_{total}$.

### 6.3 Phase 2 — End-to-End Causal Calibration

Replace the faithfulness violation with a scale-normalized MSE/KL blend:
$$v_{faith}^{(2)} = 0.5 \cdot \frac{\|x - \hat{x}\|^2 - \tau_{faith}}{\tau_{faith}} + 0.5 \cdot \frac{D_{KL}(p_{orig} \| p_{patched})}{\bar{D}_{KL}}$$
where $\bar{D}_{KL}$ is the running mean KL divergence (normalizing to unit scale). Continue ADRC updates. Freeze CAPU $\rho$ updates to prevent transient penalty spikes from the new KL signal.

**Justification:** [2503.17272] shows brief e2e fine-tuning (~3% of budget) captures most e2e benefits. [2602.14111] shows structural priors (anchoring + orthogonality) are essential to distinguish from random baselines; Phase 2 ensures these structurally-grounded features also satisfy causal validity.

---

## 7. Complete Algorithm

### 7.1 Calibration Phase (Convergence-Gated)

```
INPUTS: F, L0_target (default ⌈F/400⌉), R²_target (default 0.97), lr (default 3e-4)

1.  Stream tokens, update μ, Σ via Welford.
    Stop when ‖Σ_t − Σ_{t−Δ}‖_F / ‖Σ_t‖_F < δ_cov
    (δ_cov = 1e-3, Δ = min(⌈d²⌉, 10⁶) tokens)

2.  Eigendecompose: Σ = UΛU^T

3.  Rank selection: k = min{k : Σ_{i≤k} λ_i / tr(Λ) ≥ 0.99}
    For d ≤ 4096 AND k = d: store full U, Λ.
    Otherwise: store U_k, Λ_k.

4.  α = λ_k / κ_target                            (κ_target = 100)

5.  W_white = U(Λ + αI)^{-1/2}U^T

6.  Freeze: W_white, μ, U, Λ, α
```

### 7.2 Initialization Phase

```
7.   W_enc^(A) = W_vocab^T · U(Λ+αI)^{+1/2}U^T          [V × d]
8.   W_enc^(B): rows ~ N(0, I/d), Gram-Schmidt vs W_enc^(A)   [(F−V) × d]
9.   W_dec^(A) = W_vocab                                   [d × V]
10.  W_dec^(B): cols ~ N(0, I/d), unit-normalized           [d × (F−V)]
11.  θ_j = Quantile_j(W_enc · W_white · x, 1 − L0_target/F)
12.  ε_j = c_ε · IQR(x̃_j near θ_j)                       (c_ε = 0.1)
13.  τ_faith = (1−R²_target)·d
     τ_drift = δ²_drift · ‖W_vocab‖²_F                    (δ_drift = 0.1)
     τ_ortho = c_ortho / d                                 (c_ortho = 3)
14.  First batch → measure initial violations |v_{i,0}|
     η_i = c_η / √(|v_{i,0}| + ε_num)                     (c_η = 1.0)
15.  Initialize:
     λ_i = 0, ρ_i = 1.0, v̄_i = 1.0, ξ_i = 0, ω_o = 0.3
     ρ_min = 0.1 · ρ_0 = 0.1
     λ_disc^max = 1.0, s_disc = 0.8
```

### 7.3 Phase 1 — Constrained Sparse Optimization

```
For each batch B at step t:

  FORWARD:
  16.  x̃ = W_white · (x − μ)                               [whiten]
  17.  z = JumpReLU(W_enc · x̃ + b_enc; θ)                  [encode]
  18.  x̂ = W_dec · z + b_dec                                [decode]

  CONSTRAINTS:
  19.  v_faith = (1/N)Σ‖x̃ − W_white·(x̂−μ)‖² − τ_faith    [MSE in whitened space]
       v_drift = ‖W_dec^(A) − W_vocab‖²_F − τ_drift        [Frobenius norm]
       v_ortho = CoActOrth(A_b, W_dec) − τ_ortho            [Euclidean cosine]

  DUAL-RATE EMA:
  20.  ṽ_fast_i ← 0.9·ṽ_fast_i + 0.1·v_i
       ṽ_slow_i ← 0.99·ṽ_slow_i + 0.01·v_i

  DISCRETIZATION SCHEDULE:
  21a. r_t = t / T_total
       λ_disc = 0                          if r_t ≤ s_disc
       λ_disc = λ_disc^max·(r_t − s_disc)/(1 − s_disc)   if r_t > s_disc

  LOSS:
  21b. ℓ_sparse = L0_STE(ε_j) + λ_disc · Σ_j ‖x̃_j − Q(x̃_j;θ_j)‖²
       𝓛 = ℓ_sparse + Σ_i ρ_i · Ψ(ṽ_fast_i, λ_i/ρ_i)

  PRIMAL UPDATE:
  22.  θ_primal ← Adam(∇_θ 𝓛, lr, β₁=0.9, β₂=0.999)

  ADRC DUAL UPDATE:
  23.  For each constraint i:
         ξ_i ← (1−ω_o)·ξ_i − ω_o²·ṽ_fast_i − ω_o·λ_i
         f̂_i = ξ_i + ω_o·ṽ_fast_i
         u_i = 2ω_o·(ṽ_fast_i − ṽ_fast_i_prev) + ω_o²·ṽ_slow_i − f̂_i
         λ_i ← max(0, λ_i + u_i)

  SLOW-TIMESCALE UPDATES (every 100 steps):
  24.  Modified CAPU:
         v̄_i ← 0.99·v̄_i + 0.01·ṽ_fast_i²
         ρ_i = max(ρ_min, η_i / √(v̄_i + ε_num))
       Observer gain:
         L̂ = EMA_fast(|ṽ_fast_i − ṽ_fast_i_prev|)  [per-constraint max]
         ω_o ← min(max(0.3, L̂), 1.0)
         k_ap = 2ω_o,  k_i = ω_o²
```

### 7.4 Phase Transition

```
  25.  Enter Phase 2 when ṽ_fast_i < 0 ∀i for 100 consecutive steps.
       Fallback: transition at 0.97·T_total.
```

### 7.5 Phase 2 — E2E Causal Calibration

```
  26.  Replace v_faith with scale-normalized blend:
         v_faith^(2) = 0.5·(‖x−x̂‖² − τ_faith)/τ_faith + 0.5·D_KL/D̄_KL
       Continue ADRC updates.
       Freeze CAPU ρ updates.
       Continue discretization schedule (λ_disc ramps to λ_disc^max).
```

---

## 8. Parametric Self-Calibration

### 8.1 The 4 User-Facing Knobs

| Knob | Default | Controls |
|:---|:---|:---|
| $F$ (dictionary size) | $32 \times d_{model}$ | Capacity of the sparse dictionary |
| $L_{0,target}$ (target sparsity) | $\lceil F/400 \rceil$ | Expected active features per input |
| $R^2_{target}$ (reconstruction target) | 0.97 | Explained variance for faithfulness |
| $lr$ (learning rate) | $3 \times 10^{-4}$ | Adam step size |

### 8.2 The 7 Structural Constants

| Constant | Value | Justification |
|:---|:---|:---|
| $\delta_{cov}$ | $10^{-3}$ | Numerical convergence for covariance estimation |
| $\delta_{rank}$ | $10^{-2}$ | 99% variance retention (information-theoretic) |
| $c_\epsilon$ | 0.1 | ~10% gradient coverage near threshold |
| $c_{ortho}$ | 3 | $3\times$ random baseline $\mathbb{E}[\cos^2] = 1/d$ |
| $\delta_{drift}$ | 0.1 | 10% relative Frobenius drift budget |
| $\kappa_{target}$ | 100 | Target condition number for whitened space |
| $c_\eta$ | 1.0 | Unit-scale normalization for penalty base rate |

### 8.3 Complete Derivation Chain

```
User sets: F, L0_target, R²_target, lr

Derived from calibration data:
  N = stream until ‖Σ_t − Σ_{t-Δ}‖_F / ‖Σ_t‖_F < δ_cov     [convergence-gated]
  k = min{k : Σ_{i≤k} λ_i / tr(Λ) ≥ 1 − δ_rank}              [99% variance]
  α = λ_k / κ_target                                            [spectrum-conditioned]
  ε_j = c_ε · IQR_j(x̃_j near θ_j)                             [per-feature adaptive]

Derived from user knobs:
  θ quantile = 1 − L0_target / F                                [from L0_target, F]
  τ_faith = (1 − R²_target) · d                                 [from R²_target, d]

Derived from architecture:
  τ_drift = δ²_drift · ‖W_vocab‖²_F                            [scale-invariant]
  τ_ortho = c_ortho / d                                          [dimension-aware]
  k_ap = 2ω_o,  k_i = ω_o²                                      [ADRC bandwidth]
  ρ_min = 0.1 · ρ_0                                              [minimum penalty]
  Update interval = ⌈1/(1−β_slow)⌉ = 100                        [EMA-derived]
  Phase transition = convergence-gated (fallback 0.97·T_total)
  w_kl = 0.5, β_fast = 0.9, β_slow = 0.99                      [two-timescale r≥10]
  λ_disc^max = 1.0, s_disc = 0.8                                [discretization schedule]

Derived from first batch:
  η_i = c_η / √(|v_{i,0}| + ε_num)                             [initial-violation scaled]
```

**Total:** 4 user knobs + 18 self-calibrating + 7 structural constants = 29 parameters, of which only 4 require human choice.

---

## 9. Convergence Properties

### 9.1 Phase 1 (Informal Claims, to be validated empirically)

1. **Primal convergence:** $\mathbb{E}\|\nabla_\theta \mathscr{L}\|^2 = O(1/\sqrt{T})$ — standard non-convex SGD rate on smooth (within activation regions) augmented Lagrangian.

2. **Dual tracking:** $\limsup_{t\to\infty} |g_i(\theta_t)| = O(L_f/\omega_o^3)$ — conjectured from [2601.18142] Theorem C.7 (proven for second-order systems, transferred to first-order setting algebraically; formal re-derivation is future work).

3. **Penalty boundedness:** $\rho_{i,t} \in [\rho_{min}, \eta_i/\sqrt{\epsilon_{num}}]$ — by construction of the modified CAPU update.

4. **Discretization correction:** The STE bias is reduced by the $\lambda_{disc} \cdot \|x - Q(x)\|^2$ penalty. CAGE's formal guarantee ([2510.18784], Theorem 1) does not transfer to JumpReLU due to the discontinuity (Assumption 3 violated), but the empirical principle is well-motivated.

**Key caveats:**
- The PI = ALM equivalence provides conceptual grounding but is not an operative guarantee (three departures: Adam, EMA, ESO)
- The ISS tracking bound is from a different system order and requires formal re-derivation
- The CAGE convergence guarantee does not hold for discontinuous JumpReLU; the discretization correction is empirically motivated

### 9.2 Phase 2

Phase 2 inherits Phase 1's dual state (all constraints satisfied) and adds KL divergence. With frozen CAPU and continued ADRC, the dual variables track the new constraint surface. The brief duration (~3% of training) limits transient effects.

---

## 10. Design Motivations

Each component addresses a distinct concern in SAE training:

| Component | Concern Addressed | What Happens Without It |
|:---|:---|:---|
| Soft-ZCA | Activation anisotropy | Encoder dominated by high-variance directions; constraint scales incommensurable |
| Stratified decoder | SDL non-identifiability | Features are equally unconstrained; no vocabulary grounding |
| Discretization correction | STE persistent bias | Thresholds learn slower than encoder; persistent quantization error |
| AL-CoLe $\Psi$ | Penalty non-smoothness | Kink at $g=0$ corrupts ESO signals and Adam momentum |
| ADRC (ESO + PI) | Stochastic dual dynamics | PID D-term amplifies noise; no disturbance rejection |
| Modified CAPU | Penalty ill-conditioning | Uniform penalty lets one constraint dominate; monotone ratchet freezes primal |

These are **design motivations** supported by the referenced literature, not formal necessity proofs. Ablation studies (§11) are required to quantify each component's empirical contribution.

---

## 11. Experimental Protocol

### 11.1 Baselines

| Baseline | Reference | Purpose |
|:---|:---|:---|
| TopK SAE | [2406.04093] | Sparsity efficiency roofline |
| BatchTopK | [2412.06410] | Stability + adaptive sparsity |
| OrtSAE | [2509.22033] | Orthogonality-only (no anchoring, no whitening) |
| A-SAE / RA-SAE | [2502.12892] | Geometric anchoring (no whitening, no control) |
| Matryoshka SAE | [2503.17547] | Hierarchical/multi-scale |

### 11.2 Ablations (Cumulative)

Build from JumpReLU baseline, adding one component at a time:
1. JumpReLU (baseline)
2. \+ Soft-ZCA whitening
3. \+ Stratified decoder (anchoring + free columns)
4. \+ Orthogonality constraint
5. \+ ADRC dual dynamics (replacing fixed $\lambda$)
6. \+ Modified CAPU penalties
7. \+ Discretization correction
8. \+ Phase 2 E2E calibration (= full SPALF)

Additionally, compare ADRC vs. PID vs. standard gradient ascent for the dual update.

### 11.3 Evaluation Suite

| Metric | Source | Measures |
|:---|:---|:---|
| CE Loss vs. $L_0$ Pareto | [2406.04093] | Sparsity-fidelity efficiency |
| SAEBench (8 metrics) | [2503.09532] | Comprehensive practical evaluation |
| SynthSAEBench (ground truth) | [2602.14687] | Feature recovery accuracy |
| PW-MCC (consistency) | [2505.20254] | Cross-run feature stability |
| Control Error Energy $\int v^2 dt$ | Novel | Constraint satisfaction quality |
| Settling Time | Novel | Steps to $\Delta\lambda \approx 0$ |
| Drift-Fidelity (CosSim) | Novel | Semantic retention of anchored features |
| Absorption Rate | [2409.14507] | Hierarchical feature absorption |

### 11.4 Implementation Specifications

| Parameter | Value | Justification |
|:---|:---|:---|
| Model | Pythia-1.4B (L6), Llama-3-8B (L16) | Standard SAE benchmarking targets |
| Dictionary Size $F$ | $32 \times d_{model}$ | Consistent with prior work |
| Target Sparsity $L_{0,target}$ | $\lceil F/400 \rceil$ | ~0.25% activation rate |
| Reconstruction $R^2_{target}$ | 0.97 | 97% explained variance |
| Learning Rate | $3 \times 10^{-4}$ | Standard Adam default |
| Batch Size | 4096 tokens | Consistent with [2406.04093], [2407.14435] |
| Training Budget $T_{total}$ | ~1B tokens (~250K steps) | Standard; Phase 2 uses last 3% |
| Optimizer | Adam ($\beta_1=0.9, \beta_2=0.999$) | $\beta_1 \leq \sqrt{\beta_2}$ per [2602.09101] |
| Seeds | 3 per configuration | For PW-MCC consistency metric |

### 11.5 Generalization Validation

**Cross-Dataset Invariance:** Fix 4 knobs at defaults. Train on 6 settings (3 models $\times$ 2 layers): Pythia-160M (L3, L6), Pythia-1.4B (L6, L12), Llama-3-8B (L8, L16). Report all self-calibrated values. **Success criterion:** SPALF with defaults matches or exceeds Pareto frontier of manually-tuned baselines.

**Sensitivity Matrix:** Perturb each structural constant by $2\times$ and $0.5\times$. **Pass criterion:** No perturbation degrades CE Loss by more than 5%.

---

## 12. Novelty Assessment

### 12.1 Component-Level

| Component | Closest Prior | Novelty |
|:---|:---|:---|
| Whitened encoder space | [2511.13981] | Integrates whitening into architecture (matched-filter init, Soft-ZCA). **Incremental.** |
| Vocabulary-anchored decoder | [2502.12892], [2512.05534] | Anchors to unembedding with learned drift. **Moderate.** |
| Stratified decoder ($V$ anchored + $F{-}V$ free) | [2503.17547] | Stratifies by semantic grounding, not size. **Novel.** |
| ADRC-controlled dual ascent | [2601.18142], [2406.04558] | First application to SAE/dictionary learning. **Novel.** |
| Discretization correction | [2510.18784] | CAGE-inspired schedule adapted for JumpReLU. **Incremental.** |
| Modified CAPU (non-monotone) | [2508.15695] | Non-monotone variant is SPALF's modification. **Incremental.** |
| Self-calibrating parameter surface | — | No prior SAE derives all parameters from 4 knobs. **Novel.** |

### 12.2 System-Level

No existing work combines geometric preconditioning, semantically-stratified decoding, control-theoretic constraint management, curvature-aware non-smooth correction, and self-calibrating parameters in a single constrained optimization framework for sparse dictionary learning.

---

## 13. References

### Core SAE Methods
| ID | Title | Venue/Year |
|:---|:---|:---|
| [2407.14435](https://arxiv.org/abs/2407.14435) | JumpReLU Sparse Autoencoders | ICLR 2025 |
| [2406.04093](https://arxiv.org/abs/2406.04093) | Scaling and Evaluating Sparse Autoencoders (TopK) | OpenAI 2024 |
| [2412.06410](https://arxiv.org/abs/2412.06410) | BatchTopK Sparse Autoencoders | 2024 |
| [2502.12892](https://arxiv.org/abs/2502.12892) | Archetypal SAE (A-SAE / RA-SAE) | ICML 2025 |
| [2503.17547](https://arxiv.org/abs/2503.17547) | Matryoshka Sparse Autoencoders | 2025 |
| [2509.22033](https://arxiv.org/abs/2509.22033) | OrtSAE: Orthogonal Sparse Autoencoders | 2025 |
| [2503.17272](https://arxiv.org/abs/2503.17272) | Revisiting End-to-End SAE Training | 2025 |

### Feature Geometry & Interpretability
| ID | Title | Venue/Year |
|:---|:---|:---|
| [2512.05534](https://arxiv.org/abs/2512.05534) | Theoretical Foundation of Sparse Dictionary Learning | ICLR 2026 |
| [2602.02385](https://arxiv.org/abs/2602.02385) | Transformers Learn Factored Representations | 2026 |
| [2409.14507](https://arxiv.org/abs/2409.14507) | Feature Absorption in SAEs | NeurIPS 2025 |
| [2501.17727](https://arxiv.org/abs/2501.17727) | Not All Language Model Features Are Linear | ICLR 2025 |
| [2506.03093](https://arxiv.org/abs/2506.03093) | Matching Pursuit SAEs (Conditional Orthogonality) | 2025 |
| [2505.22255](https://arxiv.org/abs/2505.22255) | KronSAE: Kronecker Factorization | 2025 |
| [2508.16929](https://arxiv.org/abs/2508.16929) | Dimensional Collapse in Attention Outputs | 2025 |
| [2601.22966](https://arxiv.org/abs/2601.22966) | Outlier-Driven Rescaling in Transformers | 2026 |

### Whitening & Preconditioning
| ID | Title | Venue/Year |
|:---|:---|:---|
| [2511.13981](https://arxiv.org/abs/2511.13981) | Data Whitening Improves SAE Learning | AAAI 2026 |
| [2411.17538](https://arxiv.org/abs/2411.17538) | Soft-ZCA Whitening | 2024 |
| [1804.08450](https://arxiv.org/abs/1804.08450) | ZCA Whitening Properties | 2018 |
| [2506.07254](https://arxiv.org/abs/2506.07254) | SPlus: Stable Whitening Optimizer | 2025 |
| [2502.07752](https://arxiv.org/abs/2502.07752) | Structured Fisher Approximation (Alice) | 2025 |

### Control Theory & Constrained Optimization
| ID | Title | Venue/Year |
|:---|:---|:---|
| [2509.22500](https://arxiv.org/abs/2509.22500) | PI Control = Augmented Lagrangian | ICLR 2026 |
| [2601.18142](https://arxiv.org/abs/2601.18142) | ADRC-Lagrangian Methods | 2026 |
| [2406.04558](https://arxiv.org/abs/2406.04558) | nuPI: PI Controllers for Constrained Optimization | ICML 2024 |
| [2510.20995](https://arxiv.org/abs/2510.20995) | AL-CoLe: Augmented Lagrangian for Constrained Learning | NeurIPS 2025 |
| [2508.15695](https://arxiv.org/abs/2508.15695) | CAPU: Conditionally Adaptive Penalty Updates | 2025 |
| [2511.21210](https://arxiv.org/abs/2511.21210) | Accelerated ADMM with Automated Tuning | 2025 |
| [2510.17564](https://arxiv.org/abs/2510.17564) | Empirical Study of Lagrangian Methods | 2025 |
| [2112.03515](https://arxiv.org/abs/2112.03515) | Multi-Timescale Stochastic Approximation | 2025 |

### Non-Smooth Optimization & STE Theory
| ID | Title | Venue/Year |
|:---|:---|:---|
| [2510.18784](https://arxiv.org/abs/2510.18784) | CAGE: Curvature-Aware Gradient Estimation | 2025 |
| [2512.09720](https://arxiv.org/abs/2512.09720) | Moreau Envelope Smooth Approximation | 2025 |
| [2602.09101](https://arxiv.org/abs/2602.09101) | Adam-Like Lagrangians: Lyapunov Analysis | 2026 |

### Evaluation & Benchmarks
| ID | Title | Venue/Year |
|:---|:---|:---|
| [2602.14111](https://arxiv.org/abs/2602.14111) | SAE Sanity Checks: Random Baselines | 2026 |
| [2602.14687](https://arxiv.org/abs/2602.14687) | SynthSAEBench | 2026 |
| [2503.09532](https://arxiv.org/abs/2503.09532) | SAEBench | 2025 |
| [2505.20254](https://arxiv.org/abs/2505.20254) | Feature Consistency (PW-MCC) | 2025 |
