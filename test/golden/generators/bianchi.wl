(* bianchi.wl — TGR-bhs5.9
   Emits the first-Bianchi monoterm-survival case.

   Identity: R_{abcd} + R_{acdb} + R_{adbc} = 0  (first Bianchi)
   But xperm's ToCanonical captures only monoterm symmetries, so the 3-term
   form survives canonicalization. Our own simplify must match whatever xAct
   selects as the canonical form of that sum.

   Input  = R_{abcd} + R_{acdb} + R_{adbc}
   Op     = simplify
   Expected = ToCanonical[input]
*)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

raw = RiemannCD[-a, -b, -c, -d] + RiemannCD[-a, -c, -d, -b] + RiemannCD[-a, -d, -b, -c];
canon = ToCanonical[raw];

Print["raw:     ", raw];
Print["canon:   ", canon];
Print["canon==0?", canon === 0];

caseFile = <|
  "schema_version" -> 1,
  "conventions" -> <|
    "signature" -> "-+++",
    "riemann_sign" -> "standard; R^c_{acb} = Ric_{ab}",
    "name_map" -> tgrNameMap
  |>,
  "cases" -> {
    <|
      "name"      -> "first_bianchi_monoterm_survival",
      "input"     -> tgrEmitJSON[raw],
      "op"        -> "simplify",
      "op_args"   -> <||>,
      "expected"  -> tgrEmitJSON[canon],
      "generator" -> "test/golden/generators/bianchi.wl",
      "notes"     -> "First algebraic Bianchi: R_{abcd} + R_{acdb} + R_{adbc} = 0. Under monoterm canonicalization (xperm / Butler-Portugal), the 3-term form survives. Expected output is xAct's ToCanonical of the sum."
    |>
  }
|>;

outPath = FileNameJoin[{Directory[], "test/golden/data/bianchi.json"}];
Export[outPath, caseFile, "JSON"];
Print["wrote: ", outPath];
