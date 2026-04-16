(* mixed_product.wl — TGR-bhs5.14

   Product of two Riemann tensors with a mixed contraction pattern:
     R_{abcd} R^{cdef}
   Contracts slots (c, d) on right with (c, d) on left -> 2 dummy pairs,
   leaving 4 free slots (a, b, e, f). Exercises xperm product
   canonicalization with partial contractions. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

input    = RiemannCD[-a, -b, -c, -d] RiemannCD[c, d, -e, -f];
expected = ToCanonical[input];
Print["input:    ", input];
Print["expected: ", expected];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|"signature" -> "-+++", "name_map" -> tgrNameMap|>,
  "cases" -> {
    <|
      "name"      -> "mixed_symmetry_product",
      "input"     -> tgrEmitJSON[input],
      "op"        -> "simplify",
      "op_args"   -> <||>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/mixed_product.wl",
      "notes"     -> "R_{abcd} R^{cd}_{ef}: partial contraction, exercises xperm on mixed product under Riemann symmetries."
    |>
  }
|>;

Export[FileNameJoin[{Directory[], "test/golden/data/mixed_product.json"}], caseFile, "JSON"];
