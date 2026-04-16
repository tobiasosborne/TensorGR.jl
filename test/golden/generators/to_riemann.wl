(* to_riemann.wl — TGR-bhs5.16

   Einstein tensor expanded via `to_riemann`.
     G_{ab} = R_{ab} - (1/2) g_{ab} R
   In TensorGR, `to_riemann(Ein[-a,-b])` rewrites Ein into Ric + metric*R.
   In xAct, the same identity is encoded in the definition of EinsteinCD. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

input    = EinsteinCD[-a, -b];
(* Bypass xAct ToCanonical on the sum — it triggers a Validate error because
   the Scalar RicciScalarCD has empty IndexList. We emit the expanded form
   directly per the EinsteinToRicci identity; the runner re-canonicalizes
   through TensorGR. *)
expected = RicciCD[-a, -b] + (-1/2) g[-a, -b] RicciScalarCD[];
Print["input:    ", input];
Print["expected: ", expected];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|"signature" -> "-+++", "name_map" -> tgrNameMap|>,
  "cases" -> {
    <|
      "name"      -> "einstein_to_riemann_expansion",
      "input"     -> tgrEmitJSON[input],
      "op"        -> "to_riemann",
      "op_args"   -> <|"metric" -> "g", "dim" -> 4|>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/to_riemann.wl",
      "notes"     -> "Einstein expanded: G_{ab} = R_{ab} - (1/2) g_{ab} R. TensorGR's to_riemann calls einstein_to_ricci."
    |>
  }
|>;

Export[FileNameJoin[{Directory[], "test/golden/data/to_riemann.json"}], caseFile, "JSON"];
