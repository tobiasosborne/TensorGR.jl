(* einstein_identity.wl — TGR-bhs5.18

   Test G_{ab} - (R_{ab} - (1/2) g_{ab} R) = 0.
   Input is the LHS; op = to_riemann (expands G_{ab} → R_{ab} - (1/2) g_{ab} R),
   so the canonical result should be 0. Expected = 0. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

input    = EinsteinCD[-a, -b] - (RicciCD[-a, -b] + (-1/2) g[-a, -b] RicciScalarCD[]);
expected = 0;

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|"signature" -> "-+++", "name_map" -> tgrNameMap|>,
  "cases" -> {
    <|
      "name"      -> "einstein_identity_vanishes",
      "input"     -> tgrEmitJSON[input],
      "op"        -> "to_riemann",
      "op_args"   -> <|"metric" -> "g", "dim" -> 4|>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/einstein_identity.wl",
      "notes"     -> "G_{ab} - (R_{ab} - (1/2) g_{ab} R) = 0 after to_riemann expands G."
    |>
  }
|>;

Export[FileNameJoin[{Directory[], "test/golden/data/einstein_identity.json"}], caseFile, "JSON"];
