(* to_ricci.wl — TGR-bhs5.17

   Einstein expanded via `to_ricci` (should produce the same result as
   `to_riemann` for Einstein input since both reduce Ein -> Ric+g*R). *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

input    = EinsteinCD[-a, -b];
expected = RicciCD[-a, -b] + (-1/2) g[-a, -b] RicciScalarCD[];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|"signature" -> "-+++", "name_map" -> tgrNameMap|>,
  "cases" -> {
    <|
      "name"      -> "einstein_to_ricci_via_to_ricci",
      "input"     -> tgrEmitJSON[input],
      "op"        -> "to_ricci",
      "op_args"   -> <|"metric" -> "g", "dim" -> 4|>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/to_ricci.wl",
      "notes"     -> "Einstein through to_ricci: G_{ab} = R_{ab} - (1/2) g_{ab} R."
    |>
  }
|>;

Export[FileNameJoin[{Directory[], "test/golden/data/to_ricci.json"}], caseFile, "JSON"];
