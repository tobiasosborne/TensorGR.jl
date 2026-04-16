(* covd_of_ricci.wl — TGR-bhs5.15

   Identity: nabla_c R_{ab} canonical under derivative + Ricci symmetries.
   Because Ricci is symmetric, nabla_c R_{ab} = nabla_c R_{ba}; xperm must
   sort to the canonical slot order. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

raw1 = CD[-c][RicciCD[-a, -b]];
raw2 = CD[-c][RicciCD[-b, -a]];      (* swap ab — same via Ricci symmetry *)

canon1 = ToCanonical[raw1];
canon2 = ToCanonical[raw2];
Print["canon1: ", canon1];
Print["canon2: ", canon2];
Print["canon1 - canon2: ", ToCanonical[canon1 - canon2]];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|"signature" -> "-+++", "name_map" -> tgrNameMap|>,
  "cases" -> {
    <|
      "name"      -> "covd_of_ricci_slot_sort",
      "input"     -> tgrEmitJSON[raw2],
      "op"        -> "simplify",
      "op_args"   -> <||>,
      "expected"  -> tgrEmitJSON[canon2],
      "generator" -> "test/golden/generators/covd_of_ricci.wl",
      "notes"     -> "nabla_c R_{ba} -> nabla_c R_{ab} via Ricci symmetry + slot sort."
    |>
  }
|>;

Export[FileNameJoin[{Directory[], "test/golden/data/covd_of_ricci.json"}], caseFile, "JSON"];
