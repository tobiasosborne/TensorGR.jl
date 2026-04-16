(* kretschmann.wl — TGR-bhs5.12

   Identity: R_{abcd} R^{abcd} (Kretschmann scalar) canonical form.
   Exercises xperm's product canonicalization under the full Riemann
   symmetry group. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

input    = RiemannCD[-a, -b, -c, -d] RiemannCD[a, b, c, d];
expected = ToCanonical[input];
Print["input:    ", input];
Print["expected: ", expected];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|
    "signature" -> "-+++",
    "name_map"  -> tgrNameMap
  |>,
  "cases" -> {
    <|
      "name"      -> "kretschmann_canonical",
      "input"     -> tgrEmitJSON[input],
      "op"        -> "simplify",
      "op_args"   -> <||>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/kretschmann.wl",
      "notes"     -> "R_{abcd} R^{abcd}. All dummies contracted. xperm full symmetry reduces to a canonical dummy ordering."
    |>
  }
|>;

outPath = FileNameJoin[{Directory[], "test/golden/data/kretschmann.json"}];
Export[outPath, caseFile, "JSON"];
Print["wrote: ", outPath];
