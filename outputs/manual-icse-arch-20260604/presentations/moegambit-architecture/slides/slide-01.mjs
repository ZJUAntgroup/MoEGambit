const C = {
  bg: "#F8F7F2",
  paper: "#FFFFFF",
  ink: "#18202B",
  muted: "#5B6472",
  hair: "#D7DCE3",
  blue: "#315D9B",
  blueSoft: "#EAF1FB",
  green: "#1F7A4D",
  greenSoft: "#E9F5EF",
  red: "#B34B3D",
  redSoft: "#FBEDEB",
  violet: "#7257A7",
  violetSoft: "#F0ECF8",
  amber: "#C88224",
  amberSoft: "#FFF3DE",
};

function box(ctx, slide, x, y, w, h, fill = C.paper, stroke = C.hair, name = "box") {
  return ctx.addShape(slide, {
    name,
    x,
    y,
    w,
    h,
    fill,
    line: ctx.line(stroke, 1),
  });
}

function label(ctx, slide, text, x, y, w, h, opts = {}) {
  return ctx.addText(slide, {
    name: opts.name || "label",
    text,
    x,
    y,
    w,
    h,
    fontSize: opts.size || 15,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    align: opts.align || "left",
    valign: opts.valign || "top",
    typeface: opts.face || "Aptos",
    insets: opts.insets || { left: 0, right: 0, top: 0, bottom: 0 },
    fill: opts.fill || "#00000000",
    line: ctx.line("#00000000", 0),
  });
}

function tag(ctx, slide, text, x, y, w, color, fill, name) {
  box(ctx, slide, x, y, w, 24, fill, color, name);
  return label(ctx, slide, text, x + 8, y + 4, w - 16, 16, {
    size: 10,
    color,
    bold: true,
    align: "center",
    valign: "middle",
    name: `${name}-text`,
  });
}

function hArrow(ctx, slide, x1, y, x2, color, name) {
  const w = Math.max(1, x2 - x1 - 13);
  ctx.addShape(slide, {
    name: `${name}-line`,
    x: x1,
    y: y - 1,
    w,
    h: 2,
    fill: color,
    line: ctx.line(color, 0),
  });
  label(ctx, slide, ">", x2 - 16, y - 13, 16, 26, {
    size: 20,
    color,
    bold: true,
    align: "center",
    valign: "middle",
    name: `${name}-head`,
  });
}

function vArrow(ctx, slide, x, y1, y2, color, name) {
  const h = Math.max(1, y2 - y1 - 10);
  ctx.addShape(slide, {
    name: `${name}-line`,
    x: x - 1,
    y: y1,
    w: 2,
    h,
    fill: color,
    line: ctx.line(color, 0),
  });
  label(ctx, slide, "v", x - 8, y2 - 16, 16, 16, {
    size: 13,
    color,
    bold: true,
    align: "center",
    valign: "middle",
    name: `${name}-head`,
  });
}

function dot(ctx, slide, x, y, fill, name) {
  return ctx.addShape(slide, {
    name,
    geometry: "ellipse",
    x,
    y,
    w: 11,
    h: 11,
    fill,
    line: ctx.line("#FFFFFF", 1),
  });
}

function metric(ctx, slide, x, value, labelText, note, color) {
  box(ctx, slide, x, 566, 268, 82, "#FFFFFF", "#E1E5EA", `metric-${labelText}`);
  label(ctx, slide, value, x + 18, 584, 128, 28, {
    size: value.length > 7 ? 20 : 24,
    bold: true,
    color,
    name: `metric-${labelText}-value`,
  });
  label(ctx, slide, labelText, x + 154, 582, 92, 20, {
    size: 13,
    bold: true,
    color: C.ink,
    name: `metric-${labelText}-label`,
  });
  label(ctx, slide, note, x + 154, 604, 94, 28, {
    size: 10,
    color: C.muted,
    name: `metric-${labelText}-note`,
  });
}

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();

  ctx.addShape(slide, {
    name: "background",
    x: 0,
    y: 0,
    w: 1280,
    h: 720,
    fill: C.bg,
    line: ctx.line(C.bg, 0),
  });

  ctx.addShape(slide, {
    name: "kicker-01-marker",
    x: 52,
    y: 38,
    w: 8,
    h: 18,
    fill: C.blue,
    line: ctx.line(C.blue, 0),
  });
  label(ctx, slide, "ICSE ARCHITECTURE FIGURE", 66, 34, 230, 26, {
    size: 11,
    color: C.blue,
    bold: true,
    valign: "middle",
    name: "kicker-01-label",
  });

  label(ctx, slide, "MoEGambit repairs failed MoE ranks under a runtime contract", 52, 60, 810, 36, {
    size: 28,
    bold: true,
    face: "Aptos Display",
    name: "claim-title",
  });
  label(ctx, slide, "Dense/router state is refreshed from peers; only rank-local experts come from checkpoint shards.", 54, 100, 790, 22, {
    size: 14,
    color: C.muted,
    name: "claim-subtitle",
  });
  tag(ctx, slide, "CONTROL PLANE", 1022, 48, 154, C.blue, C.blueSoft, "control-plane-tag");
  label(ctx, slide, "R1 safe point  |  R2 staleness guard  |  R3 audit trail", 902, 82, 324, 18, {
    size: 12,
    color: C.muted,
    align: "right",
    name: "control-plane-note",
  });

  // Left: failure event and state provenance.
  box(ctx, slide, 52, 142, 220, 154, C.redSoft, C.red, "failure-card");
  label(ctx, slide, "Failure event", 70, 160, 160, 22, {
    size: 18,
    bold: true,
    color: C.red,
    name: "failure-title",
  });
  label(ctx, slide, "failed rank r", 70, 196, 100, 20, { size: 14, bold: true, name: "failed-rank-label" });
  label(ctx, slide, "detected at step t", 70, 220, 120, 18, { size: 12, color: C.muted, name: "step-t-label" });
  label(ctx, slide, "latest checkpoint c", 70, 244, 120, 18, { size: 12, color: C.muted, name: "step-c-label" });
  ctx.addShape(slide, {
    name: "failed-rank-chip",
    x: 202,
    y: 190,
    w: 44,
    h: 44,
    fill: "#FFFFFF",
    line: ctx.line(C.red, 1.5),
  });
  label(ctx, slide, "X", 202, 196, 44, 34, {
    size: 26,
    bold: true,
    color: C.red,
    align: "center",
    valign: "middle",
    name: "failed-rank-x",
  });

  box(ctx, slide, 52, 316, 220, 208, "#FFFFFF", C.hair, "state-card");
  label(ctx, slide, "State provenance", 70, 334, 160, 22, {
    size: 17,
    bold: true,
    name: "state-title",
  });
  tag(ctx, slide, "fresh at step t", 72, 370, 118, C.blue, C.blueSoft, "fresh-tag");
  label(ctx, slide, "dense + router", 76, 402, 120, 18, { size: 13, bold: true, name: "dense-router-label" });
  label(ctx, slide, "DP replicated", 76, 422, 110, 16, { size: 11, color: C.muted, name: "dp-replicated-label" });
  for (let i = 0; i < 6; i += 1) dot(ctx, slide, 202 + (i % 3) * 17, 398 + Math.floor(i / 3) * 17, C.blue, `dp-dot-${i}`);
  tag(ctx, slide, "stale by Delta", 72, 452, 118, C.violet, C.violetSoft, "stale-tag");
  label(ctx, slide, "expert shard", 76, 482, 120, 18, { size: 13, bold: true, name: "expert-shard-label" });
  label(ctx, slide, "EP unique", 76, 502, 110, 16, { size: 11, color: C.muted, name: "ep-unique-label" });
  for (let i = 0; i < 6; i += 1) dot(ctx, slide, 202 + (i % 3) * 17, 478 + Math.floor(i / 3) * 17, C.violet, `ep-dot-${i}`);

  hArrow(ctx, slide, 276, 258, 312, C.blue, "failure-to-contract");

  // Center: contract layer.
  box(ctx, slide, 312, 142, 310, 372, "#FFFFFF", C.blue, "contract-layer");
  label(ctx, slide, "Runtime recovery contract", 334, 160, 240, 24, {
    size: 18,
    bold: true,
    color: C.blue,
    name: "contract-title",
  });
  label(ctx, slide, "Decides when hybrid repair is allowed.", 336, 188, 250, 18, {
    size: 12,
    color: C.muted,
    name: "contract-subtitle",
  });

  box(ctx, slide, 334, 226, 266, 62, C.blueSoft, C.blue, "r1-card");
  label(ctx, slide, "R1  Safe-point repair", 352, 238, 214, 19, { size: 14, bold: true, color: C.blue, name: "r1-title" });
  label(ctx, slide, "block partial optimizer commits", 352, 260, 220, 16, { size: 11, color: C.muted, name: "r1-note" });
  vArrow(ctx, slide, 467, 292, 320, C.blue, "r1-to-r2");

  box(ctx, slide, 334, 320, 266, 76, C.greenSoft, C.green, "r2-card");
  label(ctx, slide, "R2  Staleness guard", 352, 332, 190, 19, { size: 14, bold: true, color: C.green, name: "r2-title" });
  label(ctx, slide, "Phi_prime(t) <= Phi_max", 352, 355, 180, 17, { size: 13, bold: true, color: C.ink, name: "r2-formula" });
  label(ctx, slide, "expert-weighted, window-level debt", 352, 368, 210, 16, { size: 10.5, color: C.muted, name: "r2-note" });
  vArrow(ctx, slide, 467, 400, 428, C.blue, "r2-to-r3");

  box(ctx, slide, 334, 428, 266, 54, C.amberSoft, C.amber, "r3-card");
  label(ctx, slide, "R3  Reintegration log", 352, 439, 200, 19, { size: 14, bold: true, color: C.amber, name: "r3-title" });
  label(ctx, slide, "state trace + per-segment latency", 352, 455, 220, 14, { size: 10.5, color: C.muted, name: "r3-note" });
  tag(ctx, slide, "all guards pass", 394, 486, 146, C.green, C.greenSoft, "guards-pass-tag");

  hArrow(ctx, slide, 624, 328, 662, C.green, "contract-to-hybrid");
  hArrow(ctx, slide, 624, 508, 662, C.red, "contract-to-restart");

  // Right: hybrid path.
  box(ctx, slide, 662, 128, 566, 320, C.greenSoft, C.green, "hybrid-panel");
  label(ctx, slide, "Hybrid fast path", 686, 148, 190, 24, {
    size: 19,
    bold: true,
    color: C.green,
    name: "hybrid-title",
  });
  label(ctx, slide, "Rebuild the failed rank from the freshest source for each state class.", 686, 176, 430, 18, {
    size: 12,
    color: C.muted,
    name: "hybrid-subtitle",
  });

  box(ctx, slide, 690, 214, 165, 72, "#FFFFFF", C.blue, "peer-source");
  label(ctx, slide, "Healthy DP peer", 708, 229, 124, 18, { size: 13, bold: true, color: C.blue, name: "peer-title" });
  label(ctx, slide, "dense/router at t", 708, 250, 120, 16, { size: 11, color: C.muted, name: "peer-note" });
  tag(ctx, slide, "Path P", 768, 258, 62, C.blue, C.blueSoft, "path-p-tag");

  box(ctx, slide, 690, 308, 165, 72, "#FFFFFF", C.violet, "ckpt-source");
  label(ctx, slide, "Checkpoint shard", 708, 323, 125, 18, { size: 13, bold: true, color: C.violet, name: "ckpt-title" });
  label(ctx, slide, "experts at c", 708, 344, 110, 16, { size: 11, color: C.muted, name: "ckpt-note" });
  tag(ctx, slide, "Path C", 768, 352, 62, C.violet, C.violetSoft, "path-c-tag");

  hArrow(ctx, slide, 858, 250, 952, C.blue, "peer-to-rank");
  hArrow(ctx, slide, 858, 344, 952, C.violet, "ckpt-to-rank");

  box(ctx, slide, 952, 230, 206, 126, "#FFFFFF", C.green, "replacement-rank");
  label(ctx, slide, "Replacement rank", 976, 247, 158, 22, {
    size: 16,
    bold: true,
    color: C.green,
    align: "center",
    name: "replacement-title",
  });
  label(ctx, slide, "fresh dense/router", 976, 281, 140, 16, { size: 11, color: C.blue, name: "replacement-fresh" });
  label(ctx, slide, "bounded expert staleness", 976, 303, 155, 16, { size: 11, color: C.violet, name: "replacement-stale" });
  tag(ctx, slide, "resume at step t", 989, 320, 132, C.green, C.greenSoft, "resume-t-tag");

  box(ctx, slide, 690, 402, 468, 28, "#FFFFFF", C.green, "two-phase-rail");
  label(ctx, slide, "weights first", 708, 409, 90, 14, { size: 10.5, bold: true, color: C.green, name: "weights-first" });
  label(ctx, slide, "optimizer later", 862, 409, 100, 14, { size: 10.5, bold: true, color: C.green, name: "optimizer-later" });
  label(ctx, slide, "HEALTHY", 1038, 409, 70, 14, { size: 10.5, bold: true, color: C.green, name: "healthy-label" });
  hArrow(ctx, slide, 800, 416, 852, C.green, "phase-a-b");
  hArrow(ctx, slide, 966, 416, 1028, C.green, "phase-b-healthy");

  // Restart fallback.
  box(ctx, slide, 662, 476, 566, 76, C.redSoft, C.red, "restart-panel");
  label(ctx, slide, "Restart fallback", 686, 492, 142, 20, {
    size: 16,
    bold: true,
    color: C.red,
    name: "restart-title",
  });
  label(ctx, slide, "if any contract guard fails", 686, 516, 210, 16, {
    size: 10.5,
    color: C.muted,
    name: "restart-condition",
  });
  box(ctx, slide, 936, 493, 62, 34, "#FFFFFF", C.violet, "restart-ckpt");
  label(ctx, slide, "CKPT c", 942, 503, 50, 14, { size: 10, bold: true, color: C.violet, align: "center", name: "restart-ckpt-label" });
  hArrow(ctx, slide, 1001, 510, 1046, C.red, "restart-arrow-1");
  box(ctx, slide, 1046, 493, 74, 34, "#FFFFFF", C.red, "restart-replay");
  label(ctx, slide, "replay", 1056, 503, 54, 14, { size: 10, bold: true, color: C.red, align: "center", name: "restart-replay-label" });
  hArrow(ctx, slide, 1124, 510, 1164, C.red, "restart-arrow-2");
  label(ctx, slide, "resume", 1164, 500, 50, 18, { size: 12, bold: true, color: C.red, name: "restart-resume-label" });

  // Metrics rail.
  metric(ctx, slide, 52, "20.6-55.0%", "less latency", "raw recovery across two MoE models", C.green);
  metric(ctx, slide, 356, "0", "replay steps", "hybrid resumes directly at step t", C.green);
  metric(ctx, slide, 660, "1e-2", "quality guard", "empirical cliff for Phi_prime(t)", C.blue);
  metric(ctx, slide, 964, "JSON", "audit trail", "policy inputs + state transitions", C.amber);

  label(ctx, slide, "Figure intent: contract answers when the fast path is safe; mechanism answers how each state class is rebuilt.", 54, 674, 790, 18, {
    size: 10.5,
    color: C.muted,
    name: "figure-note",
  });
  label(ctx, slide, "MoEGambit architecture", 1070, 674, 156, 18, {
    size: 10.5,
    color: C.muted,
    align: "right",
    name: "footer-label",
  });

  return slide;
}
