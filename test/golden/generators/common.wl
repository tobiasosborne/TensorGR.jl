(* ================================================================
   common.wl — xAct <-> neutral-schema-v1 emitter library

   TGR-bhs5.7. Provides:
     tgrEmitJSON[expr]                Association (dict), schema v1.
     tgrEmitJSONString[expr]          Same, rendered as JSON text.
     tgrNormalizeDummies[expr]        Rename dummy pairs to d1..dN.
     tgrSetupXAct[]                   DefManifold M4 + DefMetric g.
     tgrNameMap                       xAct head -> neutral name.

   All symbols live in Global` — no package contexts.
   ================================================================ *)

If[!TrueQ[$TGRGoldenLoaded],

  (* --- xAct setup ------------------------------------------------- *)
  tgrSetupXAct[] := Module[{},
    AppendTo[$Path, FileNameJoin[{Directory[], "reference/xAct"}]];
    Off[General::shdw];
    $DefInfoQ = False;
    If[!ValueQ[xAct`xTensor`$Version], Needs["xAct`xTensor`"]];
    Quiet[
      If[!NameQ["Global`M"] || !xAct`xTensor`ManifoldQ[M],
        xAct`xTensor`DefManifold[M, 4, {a, b, c, d, e, f, g1, h1}];
        xAct`xTensor`DefMetric[-1, g[-a, -b], CD, PrintAs -> "g"]
      ], xAct`xTensor`DefManifold::named
    ];
  ];

  (* --- xPert setup: load xPert, define metric perturbation h ---- *)
  tgrSetupXPert[] := Module[{},
    tgrSetupXAct[];
    If[!ValueQ[xAct`xPert`$Version], Needs["xAct`xPert`"]];
    Quiet[
      If[!NameQ["Global`h"] || !xAct`xTensor`xTensorQ[h],
        xAct`xPert`DefMetricPerturbation[g, h, eps]
      ], xAct`xTensor`DefTensor::named
    ];
  ];

  (* --- name map --------------------------------------------------- *)
  tgrNameMap = <|
    "g"             -> "g",
    "delta"         -> "delta",
    "RiemannCD"     -> "Riem",
    "RicciCD"       -> "Ric",
    "RicciScalarCD" -> "RicScalar",
    "EinsteinCD"    -> "Ein",
    "WeylCD"        -> "Weyl",
    "SchoutenCD"    -> "Sch",
    "TFRicciCD"     -> "TFRicci",
    "KretschmannCD" -> "Kretschmann",
    "epsilong"      -> "epsilon",
    "h"             -> "h"
  |>;

  tgrNeutralName[head_Symbol] :=
    Lookup[tgrNameMap, SymbolName[head], SymbolName[head]];

  (* --- reserved heads (to exclude from tensor dispatch) ---------- *)
  tgrReservedHead[h_Symbol] :=
    MemberQ[{Plus, Times, Power, List, Hold, HoldForm,
             Rule, RuleDelayed, Pattern, Blank}, h] ||
    SameQ[h, xAct`xTensor`CD];

  (* --- index helpers --------------------------------------------- *)
  tgrIndexName[ s_Symbol] := s;
  tgrIndexName[-s_Symbol] := s;
  tgrIndexPos[  s_Symbol] := "up";
  tgrIndexPos[ -s_Symbol] := "down";

  tgrEmitIndex[i_] := <|
    "name"    -> SymbolName[tgrIndexName[i]],
    "pos"     -> tgrIndexPos[i],
    "vbundle" -> "Tangent"
  |>;

  (* --- collect indices at top-level (pattern-based, not recursive) --- *)
  (* An `index-expression` is any atom matching `_Symbol` or `-_Symbol`
     that sits in a tensor-head slot. We reach them by Cases with depth
     Infinity, filtering to those inside any known tensor-head. *)

  tgrIsIndex[x_] := MatchQ[x, _Symbol | (-_Symbol)];

  ClearAll[tgrCollectIndices];
  tgrCollectIndices[expr_] := Which[
    Head[expr] === Plus || Head[expr] === Times,
      Join @@ (tgrCollectIndices /@ (List @@ expr)),
    MatchQ[expr, xAct`xTensor`CD[_][_]],
      Join[{expr[[0, 1]]}, tgrCollectIndices[expr[[1]]]],
    And[MatchQ[Head[expr], _Symbol],
        MatchQ[Head[Head[expr]], Symbol],
        !tgrReservedHead[Head[expr]],
        Length[expr] > 0],
      List @@ expr,
    True, {}
  ];

  tgrFindDummies[expr_] := Module[{idxs, grouped},
    idxs    = tgrCollectIndices[expr];
    grouped = GroupBy[idxs, tgrIndexName];
    Keys[Select[grouped, Length[Union[tgrIndexPos /@ #]] >= 2 &]]
  ];

  tgrOrderedDummies[expr_] := Module[{idxs, dummySet, seen = <||>, out = {}, nm},
    idxs     = tgrCollectIndices[expr];
    dummySet = tgrFindDummies[expr];
    Do[
      nm = tgrIndexName[idx];
      If[MemberQ[dummySet, nm] && !KeyExistsQ[seen, nm],
        AppendTo[out, nm];
        seen[nm] = True
      ],
      {idx, idxs}];
    out
  ];

  tgrNormalizeDummies[expr_] := Module[{ordered, phase1, phase2, step1},
    ordered = tgrOrderedDummies[expr];
    If[ordered === {}, Return[expr]];
    phase1 = MapIndexed[
      #1 -> Symbol["tgrGoldDum" <> ToString[First[#2]]] &, ordered];
    step1 = expr /. phase1;
    phase2 = MapIndexed[
      Symbol["tgrGoldDum" <> ToString[First[#2]]] ->
        Symbol["d" <> ToString[First[#2]]] &, ordered];
    step1 /. phase2
  ];

  (* --- emission dispatch ---------------------------------------- *)

  ClearAll[tgrEmitRaw];

  tgrEmitRaw::unknownHead = "Unknown tensor head `1`; emitting with raw name.";

  tgrEmitRaw[expr_] := Which[
    (* Integer scalar *)
    IntegerQ[expr],
      <|"type"  -> "scalar",
        "value" -> <|"rational" -> <|"num" -> expr, "den" -> 1|>|>|>,

    (* Rational scalar *)
    Head[expr] === Rational,
      <|"type"  -> "scalar",
        "value" -> <|"rational" -> <|"num" -> Numerator[expr],
                                    "den" -> Denominator[expr]|>|>|>,

    (* Sum *)
    Head[expr] === Plus,
      <|"type"  -> "sum",
        "terms" -> tgrEmitRaw /@ (List @@ expr)|>,

    (* Product *)
    Head[expr] === Times,
      Module[{parts, coefs, tensors, coef},
        parts   = List @@ expr;
        coefs   = Cases[parts, _Integer | _Rational];
        tensors = DeleteCases[parts, _Integer | _Rational];
        coef    = If[coefs === {}, 1, Times @@ coefs];
        <|"type"    -> "product",
          "coef"    -> <|"num" -> Numerator[coef], "den" -> Denominator[coef]|>,
          "factors" -> tgrEmitRaw /@ tensors|>
      ],

    (* CovD: CD[ci][arg] *)
    MatchQ[expr, xAct`xTensor`CD[_][_]],
      <|"type"  -> "deriv",
        "covd"  -> "D",
        "index" -> tgrEmitIndex[expr[[0, 1]]],
        "arg"   -> tgrEmitRaw[expr[[1]]]|>,

    (* Tensor head known by name *)
    And[MatchQ[Head[expr], _Symbol],
        !tgrReservedHead[Head[expr]],
        KeyExistsQ[tgrNameMap, SymbolName[Head[expr]]]],
      <|"type"    -> "tensor",
        "name"    -> tgrNeutralName[Head[expr]],
        "indices" -> tgrEmitIndex /@ (List @@ expr)|>,

    (* Fallback for tensors: emit raw symbol name *)
    And[MatchQ[Head[expr], _Symbol],
        !tgrReservedHead[Head[expr]],
        Length[expr] > 0],
      (Message[tgrEmitRaw::unknownHead, SymbolName[Head[expr]]];
       <|"type"    -> "tensor",
         "name"    -> SymbolName[Head[expr]],
         "indices" -> tgrEmitIndex /@ (List @@ expr)|>),

    True,
      (Message[tgrEmitRaw::unknownHead, ToString[expr]];
       <|"type" -> "scalar",
         "value" -> <|"symbol" -> ToString[expr]|>|>)
  ];

  tgrEmitJSON[expr_]       := tgrEmitRaw[tgrNormalizeDummies[expr]];
  tgrEmitJSONString[expr_] := ExportString[tgrEmitJSON[expr], "JSON"];

  $TGRGoldenLoaded = True;
];
