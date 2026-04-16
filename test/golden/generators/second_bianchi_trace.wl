(* second_bianchi_trace.wl — TGR-bhs5.13

   Contracted (differential) second Bianchi:
     nabla_a R^{a}{}_{bcd} = nabla_c R_{bd} - nabla_d R_{bc}

   We emit the LHS - RHS as the "input"; under ToCanonical this should
   collapse to 0. The case then asserts TensorGR simplify agrees. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

(* xAct writes differential Bianchi via CD chains. *)
lhs = CD[-e][RiemannCD[e, -b, -c, -d]];
rhs = CD[-c][RicciCD[-b, -d]] - CD[-d][RicciCD[-b, -c]];

diff     = lhs - rhs;
expected = ToCanonical[diff];
Print["diff:     ", diff];
Print["expected: ", expected];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|"signature" -> "-+++", "name_map" -> tgrNameMap|>,
  "cases" -> {
    <|
      "name"      -> "second_bianchi_trace",
      "input"     -> tgrEmitJSON[diff],
      "op"        -> "simplify",
      "op_args"   -> <||>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/second_bianchi_trace.wl",
      "notes"     -> "Contracted second Bianchi: nabla_a R^a_{bcd} = nabla_c R_{bd} - nabla_d R_{bc}. LHS-RHS should be 0 under canonicalization."
    |>
  }
|>;

outPath = FileNameJoin[{Directory[], "test/golden/data/second_bianchi_trace.json"}];
Export[outPath, caseFile, "JSON"];
Print["wrote: ", outPath];
