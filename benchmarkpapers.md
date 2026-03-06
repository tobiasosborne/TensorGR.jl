That's the correct strategy. Let me find the best candidates.That's a goldmine — 1170 papers to choose from. Let me pick the optimal stress-test suite. The goal is to cover different subsystems and find different classes of bugs.

**Tier 1: Direct reproduceability, exercises your exact stack (linearised perturbation theory + SVT)**

| Paper | Why it's a good test | What it stresses |
|---|---|---|
| **Brizuela, Martín-García, Mena Marugán 2009** ([0807.0824](https://arxiv.org/abs/0807.0824)) — the xPert paper itself | Contains explicit tables of linearised curvature tensors at orders n=1..5. These are the canonical reference outputs. | Perturbation engine, every order |
| **Pitrou, Roy, Umeh 2013** ([1302.6174](https://arxiv.org/abs/1302.6174)) — the xPand paper | Full SVT decomposition in FLRW backgrounds, multiple gauges, timing benchmarks. Your 3+1 code should reproduce their examples. | SVT decomposition, gauge choice, Fourier space |
| **Grumiller, Irakleidou, Lovrekovic, McNees 2013** ([1310.0819](https://arxiv.org/abs/1310.0819)) — Conformal gravity holography in 4D | Uses $C_{\mu\nu\rho\sigma}C^{\mu\nu\rho\sigma}$ (your $\beta=1/3$ case) explicitly. Propagator structure, boundary terms. | Weyl tensor, your exact action at a special point |
| **Nutma 2013** ([1308.3493](https://arxiv.org/abs/1308.3493)) — the xTras paper | Contains a worked example: construct the most general free spin-2 Lagrangian on flat background, find all contractions. Essentially your fourth-derivative gravity calculation done from scratch via ansatz. | Ansatz construction, contraction engine, canonicalisation |

**Tier 2: Increasing complexity, will find real bugs**

| Paper | Why | What it stresses |
|---|---|---|
| **Brizuela, Martín-García, Tiglio 2009** ([0903.1134](https://arxiv.org/abs/0903.1134)) — Second-order perturbation of Schwarzschild | Complete gauge-invariant formalism for **arbitrary second-order** perturbations. This is where expression swell hits. If your engine survives this, it's real. | Second-order perturbation, curved background, massive expressions |
| **Stein, Yunes 2011** ([1012.3144](https://arxiv.org/abs/1012.3144)) — GW stress-energy in alternative gravity | Computes effective stress-energy tensor for GWs in Chern-Simons and scalar-tensor theories. Non-standard curvature couplings. | Modified gravity Lagrangians beyond $R^2$, IBP on non-trivial terms |
| **Fröb, Roura, Verdaguer 2014** ([1403.3335](https://arxiv.org/abs/1403.3335)) — Riemann correlator in de Sitter | Loop corrections to the Riemann two-point function. Requires linearised Riemann, perturbation theory on a curved (de Sitter) background. | Curved background perturbation, tensor two-point functions |
| **Deffayet, Esposito-Farese, Vikman 2009** ([0901.1314](https://arxiv.org/abs/0901.1314)) — Covariant Galileon | Constructs the most general scalar-tensor Lagrangian with second-order equations. Lots of Levi-Civita contractions, derivative-heavy. | Higher-derivative scalar-tensor, Levi-Civita tensor |

**Tier 3: Will probably break things, but worth trying**

| Paper | Why | What it stresses |
|---|---|---|
| **Barker, Marzo, Rigouzzo 2024** ([2406.09500](https://arxiv.org/abs/2406.09500)) — PSALTer paper | Contains the spin-projection operator algorithm, full particle spectrum for multiple theories including Fierz-Pauli. If you can reproduce their outputs, you've built the PSALTer equivalent. | Spin-projection operators, systematic Lagrangian analysis |
| **Levi, Steinhoff 2016** ([1705.06309](https://arxiv.org/abs/1705.06309)) — EFTofPNG | Post-Newtonian gravity to very high order. Enormous expressions, dozens of terms. This is the scalability test. | Performance under expression swell, PN expansion |
| **Green, Peeters, Stahn 2005** ([hep-th/0506161](https://arxiv.org/abs/hep-th/0506161)) — Superfield integrals | Higher-dimensional tensors with complicated symmetry groups. Exercises `xperm.c` canonicalisation with many indices. | Canonicaliser at scale, high-rank tensors |

**My recommended order:**

Start with the xPert paper (0807.0824) Table 1 — reproduce every entry. That's the ground truth, directly comparable, and exercises the core. Then the xTras spin-2 example (1308.3493). Then the xPand examples (1302.6174) for your 3+1 code. Those three give you comprehensive coverage of your v0.1 scope.

Then Brizuela-Martín-García-Tiglio 2009 (0903.1134) for second-order perturbations. This is where the first real bug will appear — I'd bet on dummy index scoping in nested second-order expressions. After that, the Covariant Galileon paper (0901.1314) to test non-standard Lagrangians. Then PSALTer.

Each paper you reproduce is both a stress test and a publishable validation. By the time you've done five of these, you have a paper: "TensorGR.jl: reproducing 20 years of xAct calculations without Mathematica."
