(* ricci_contraction.wl — TGR-bhs5.11

   Identity: R^c_{a c b} = R_{ab} (both slot conventions agree).
   Case: input = R^c_{a c b}, expected = Ric_{ab}, op = simplify. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

input    = RiemannCD[-a, -c, -b, c];  (* R^c_{a c b} — xAct slot 2/4 up/down trace *)
expected = ToCanonical[input];         (* xAct contracts slot-2&4 -> Ric[-a,-b] *)
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
      "name"      -> "ricci_contraction_slot_1_3",
      "input"     -> tgrEmitJSON[input],
      "op"        -> "simplify",
      "op_args"   -> <||>,
      "expected"  -> tgrEmitJSON[expected],
      "generator" -> "test/golden/generators/ricci_contraction.wl",
      "notes"     -> "R^c_{a c b} = Ric_{ab}. Both xAct and TensorGR agree; xAct auto-replaces Riemann contraction with Ricci."
    |>
  }
|>;

outPath = FileNameJoin[{Directory[], "test/golden/data/ricci_contraction.json"}];
Export[outPath, caseFile, "JSON"];
Print["wrote: ", outPath];
