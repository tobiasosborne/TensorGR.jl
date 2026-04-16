(* _emitter_probe.wl — TGR-bhs5.7 xAct emitter probe.
   Loads common.wl, emits three known cases, writes JSON array. *)

Get[FileNameJoin[{Directory[], "test/golden/generators/common.wl"}]];
tgrSetupXAct[];

cases = {
  <|"name" -> "g_down2",
    "input" -> "g[-a, -b]",
    "emitted" -> tgrEmitJSON[g[-a, -b]]|>,

  <|"name" -> "g_up2",
    "input" -> "g[a, b]",
    "emitted" -> tgrEmitJSON[g[a, b]]|>,

  <|"name" -> "ric_trace",
    "input" -> "g[a, b] RicciCD[-a, -b]",
    "emitted" -> tgrEmitJSON[g[a, b] RicciCD[-a, -b]]|>
};

outPath = FileNameJoin[{Directory[], "test/golden/data/_emitter_probe.actual.json"}];
Export[outPath, cases, "JSON"];
Print["wrote: ", outPath];
