(* weyl_trace_free.wl — TGR-bhs5.19

   Identity: g^{ac} C_{abcd} = 0 (Weyl is trace-free on any slot pair).
   Both xAct and TensorGR should produce 0 under canonicalization. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

input    = g[a, c] WeylCD[-a, -b, -c, -d];
expected = ToCanonical[input];
Print["input:    ", input];
Print["expected: ", expected];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|"signature" -> "-+++", "name_map" -> tgrNameMap|>,
  "cases" -> {
    <|
      "name"      -> "weyl_trace_free",
      "input"     -> tgrEmitJSON[input],
      "op"        -> "simplify",
      "op_args"   -> <||>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/weyl_trace_free.wl",
      "notes"     -> "g^{ac} C_{abcd} = 0. xAct auto-reduces via Weyl trace-free rule."
    |>
  }
|>;

Export[FileNameJoin[{Directory[], "test/golden/data/weyl_trace_free.json"}], caseFile, "JSON"];
