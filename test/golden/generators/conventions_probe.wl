(* ================================================================
   conventions_probe.wl — Phase 0 probe for TGR-bhs5 golden masters

   Dumps the three sanity identities required by CONVENTIONS.md into
   test/golden/data/conventions_probe.json. The gate issue (TGR-bhs5.3)
   compares these against TensorGR's output.

   Run:   wolframscript -f test/golden/generators/conventions_probe.wl
   Depends on: reference/xAct/xAct on $Path (auto-added below).

   Physics is ground truth. If any identity evaluates to non-zero, fail
   loud and do NOT write the JSON.
   ================================================================ *)

AppendTo[$Path, FileNameJoin[{Directory[], "reference/xAct"}]];

Off[General::shdw];            (* silence shadow warnings from re-load *)
$DefInfoQ = False;             (* quieter defs *)
$PrePrint = Identity;

<< xAct`xTensor`;

(* ---- setup --------------------------------------------------------- *)

DefManifold[M, 4, {a, b, c, d, e, f, g1, h1}];
DefMetric[-1, g[-a, -b], CD, PrintAs -> "g"];

(* ---- probe cases --------------------------------------------------- *)

SetAttributes[probe, HoldFirst];
probe[inputHeld_, name_String, description_String] := Module[
  {result, isZero, inputStr, resultStr},
  inputStr  = StringReplace[
    ToString[HoldForm[inputHeld], InputForm],
    {StartOfString ~~ "HoldForm[" -> "", "]" ~~ EndOfString -> ""}];
  result    = ToCanonical[inputHeld];
  isZero    = (result === 0);
  resultStr = ToString[result, InputForm];
  Print["---- ", name, " ----"];
  Print["input:       ", inputStr];
  Print["canonical:   ", resultStr];
  Print["is_zero:     ", isZero];
  If[!isZero,
    Print["FAIL: ", name, " did not reduce to 0 under ToCanonical."];
    Print["      CONVENTIONS.md says this MUST be 0 on both sides."];
  ];
  <|
    "name"        -> name,
    "description" -> description,
    "input_wl"    -> inputStr,
    "canonical_wl"-> resultStr,
    "is_zero"     -> isZero
  |>
];

cases = {
  probe[CD[-c][g[-a, -b]],
        "metric_compatibility",
        "Metric compatibility: grad of metric is zero."],
  probe[RiemannCD[-a, -c, -b, c] - RicciCD[-a, -b],
        "ricci_contraction",
        "Ricci is R contracted on slots 1 (up) and 3 (down): R^c_{acb} - R_{ab} = 0."],
  probe[g[a, b] RicciCD[-a, -b] - RicciScalarCD[],
        "ricci_scalar_trace",
        "Ricci scalar is the trace: g^{ab} R_{ab} - R = 0."]
};

allZero = And @@ (#["is_zero"] & /@ cases);

Print[];
Print["============================================================"];
Print["Probe summary: ", Length[cases], " cases, all_zero = ", allZero];
Print["============================================================"];

If[!allZero,
  Print["ABORT: one or more identities did not reduce to 0."];
  Print["       Not writing JSON. Fix xAct convention or setup."];
  Exit[1]
];

(* ---- emit JSON ---------------------------------------------------- *)

outPath = FileNameJoin[{Directory[], "test/golden/data/conventions_probe.json"}];

payload = <|
  "schema"      -> "conventions_probe/v0",
  "generator"   -> "test/golden/generators/conventions_probe.wl",
  "conventions" -> <|
    "signature"  -> "-+++",
    "dimension"  -> 4,
    "generator_xact_version" -> ToString[xAct`xTensor`$Version]
  |>,
  "cases"       -> cases,
  "all_zero"    -> allZero
|>;

Export[outPath, payload, "JSON"];
Print["Wrote: ", outPath];
