# Ground Truth Reference Library

Local copies of papers and textbooks cited in `test/test_ground_truth_*.jl` verification tests.
All copyrighted material is `.gitignore`d — only this README, the NIST snapshot, and .gitignore are tracked.

## Papers (cited in ground-truth tests)

| File | Citation | Tests citing it |
|------|----------|-----------------|
| `martel_poisson_2005.pdf` | Martel & Poisson, Phys. Rev. D 71, 104003 (2005), arXiv:gr-qc/0502028 | `test_ground_truth_harmonics.jl`, `test_harmonic_orthogonality.jl`, `test_tensor_harmonic_orthogonality.jl` |
| `nutma_2014_xtras.pdf` | Nutma, Comp. Phys. Comm. 185, 1719 (2014), arXiv:1308.3493 | `test_ground_truth_contractions.jl` |
| `fulling_1992.pdf` | Fulling, King, Wybourne & Cummins, CQG 9, 1151 (1992) | `test_quadratic_riemann_invariants.jl` |
| `eguchi_gilkey_hanson_1980.pdf` | Eguchi, Gilkey & Hanson, Phys. Rep. 66, 213 (1980) | `test_field_strength.jl` |
| `iyer_wald_1994.pdf` | Iyer & Wald, Phys. Rev. D 50, 846 (1994), arXiv:gr-qc/9403028 | Covariant phase space issues (TGR-s50.*) |
| `kobayashi_2019_horndeski.pdf` | Kobayashi, Rep. Prog. Phys. 82, 086901 (2019), arXiv:1901.04778 | Horndeski theory issues (TGR-ble.*) |
| `nist_dlmf_34.3.md` | NIST DLMF Sec 34.3 (public domain) | `test_ground_truth_3j.jl` |

## Textbooks

| File | Citation | Key equations used |
|------|----------|--------------------|
| `wald_1984.djvu` | Wald, *General Relativity*, U. Chicago Press (1984) | Eqs 3.2.14, 3.2.15, 3.2.16, 3.2.28 |
| `nakahara_2003.pdf` | Nakahara, *Geometry, Topology and Physics*, 2nd ed., CRC (2003) | Eqs 11.5, 11.7, 11.12, 11.28, 11.76 |
| `misner_thorne_wheeler_1973_gravitation.djvu` | Misner, Thorne & Wheeler, *Gravitation*, Freeman (1973) | Sign conventions, Ch 21 |
| `hawking_ellis_1973_large_scale_structure.pdf` | Hawking & Ellis, *The Large Scale Structure of Space-Time*, CUP (1973) | Penrose diagrams, singularity theorems |
| `weinberg_1972_gravitation_cosmology.djvu` | Weinberg, *Gravitation and Cosmology*, Wiley (1972) | Conventions cross-check |
| `schutz_1985_first_course_gr.djvu` | Schutz, *A First Course in General Relativity*, CUP (1985) | Pedagogical reference |
| `hartle_2003_gravity.djvu` | Hartle, *Gravity*, Benjamin Cummings (2003) | Pedagogical reference |
| `zee_2013_einstein_gravity.pdf` | Zee, *Einstein Gravity in a Nutshell*, Princeton (2013) | EFT perspective |
| `warner_1971_differentiable_manifolds.pdf` | Warner, *Foundations of Differentiable Manifolds and Lie Groups*, Springer (1971) | Differential geometry foundations |
| `feynman_1995_lectures_gravitation.djvu` | Feynman, *Lectures on Gravitation*, Addison-Wesley (1995) | Field-theoretic GR perspective |

## Obtaining copyrighted material

Papers: available via arXiv (free) or institutional access (TIB).
Textbooks: obtain from your institution or publisher.
