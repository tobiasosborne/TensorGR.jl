# NIST DLMF Section 34.3 — Properties of 3j Symbols

Source: https://dlmf.nist.gov/34.3
Retrieved: 2026-03-15

## Special Cases

**34.3.1:** (j j 0; m -m 0) = (-1)^(j-m) / sqrt(2j+1)

**34.3.4-5:** (j1 j2 j3; 0 0 0) = 0 if j1+j2+j3 is odd;
otherwise = (-1)^(J/2) * sqrt((J-2j1)!(J-2j2)!(J-2j3)!(J/2)! / ((J+1)!(J/2-j1)!(J/2-j2)!(J/2-j3)!))
where J = j1+j2+j3

## Symmetry Relations

**34.3.8:** Even permutations leave symbol unchanged:
(j1 j2 j3; m1 m2 m3) = (j2 j3 j1; m2 m3 m1) = (j3 j1 j2; m3 m1 m2)

**34.3.9:** Odd permutations introduce phase (-1)^(j1+j2+j3):
(j1 j2 j3; m1 m2 m3) = (-1)^(j1+j2+j3) * (j2 j1 j3; m2 m1 m3)

**34.3.10:** Sign reversal of all m:
(j1 j2 j3; m1 m2 m3) = (-1)^(j1+j2+j3) * (j1 j2 j3; -m1 -m2 -m3)

## Orthogonality

**34.3.16:** Sum over m1,m2: (2j3+1) * (j1 j2 j3; m1 m2 m3) * (j1 j2 j3'; m1 m2 m3') = delta(j3,j3') delta(m3,m3')

**34.3.17:** Sum over j3,m3: (2j3+1) * (j1 j2 j3; m1 m2 m3) * (j1 j2 j3; m1' m2' m3) = delta(m1,m1') delta(m2,m2')

## Relation to Spherical Harmonics

**34.3.22:** Triple integral (Gaunt):
integral Y_{l1,m1} Y_{l2,m2} Y_{l3,m3} dOmega = sqrt((2l1+1)(2l2+1)(2l3+1)/(4pi)) * (l1 l2 l3; 0 0 0) * (l1 l2 l3; m1 m2 m3)
