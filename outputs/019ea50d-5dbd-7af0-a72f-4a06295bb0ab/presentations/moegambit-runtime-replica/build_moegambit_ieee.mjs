#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";

import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
} from "/Users/zds/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const workspace = "/Users/zds/bsr/outputs/019ea50d-5dbd-7af0-a72f-4a06295bb0ab/presentations/moegambit-runtime-replica";
const outputDir = "/Users/zds/Desktop";
const finalPptx = path.join(outputDir, "MoEGambit_Runtime_Architecture_IEEE双栏矢量复刻.pptx");
const previewPng = path.join(workspace, "preview", "moegambit-ieee-vector-replica.png");
const layoutJson = path.join(workspace, "layout", "moegambit-ieee-vector-replica.layout.json");

const W = 1491;
const H = 1055;

const C = {
  black: "#111111",
  text: "#171717",
  muted: "#4B5563",
  light: "#F8FAFC",
  panel: "#FFFFFF",
  line: "#8A8A8A",
  rule: "#6B7280",
  blue: "#0B5DB8",
  blueSoft: "#F3F8FF",
  red: "#D62828",
  redSoft: "#FFF3F3",
  green: "#1B7F37",
  greenSoft: "#F2FBF2",
  orange: "#E65A16",
  orangeSoft: "#FFF6ED",
  yellowSoft: "#FFF8D8",
  graySoft: "#F7F7F7",
};

await ensureArtifactToolWorkspace(workspace);
const artifact = await importArtifactTool(workspace);
const { Presentation, PresentationFile } = artifact;
const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const slide = presentation.slides.add();
const ctx = createSlideContext(artifact, {
  slideSize: { width: W, height: H },
  workspaceDir: workspace,
  titleFont: "Arial",
  bodyFont: "Arial",
  monoFont: "Arial",
});

function ln(color = C.line, width = 1.2, style = "solid") {
  return ctx.line(color, width, style);
}

function shape(x, y, w, h, options = {}) {
  const item = ctx.addShape(slide, {
    x,
    y,
    w,
    h,
    geometry: options.geometry ?? "rect",
    fill: options.fill ?? C.panel,
    line: options.line ?? ln(C.line, 1.2),
    name: options.name,
  });
  if (Number.isFinite(options.rotation)) {
    item.position = { left: x, top: y, width: w, height: h, rotation: options.rotation };
  }
  return item;
}

function text(x, y, w, h, value, options = {}) {
  return ctx.addText(slide, {
    x,
    y,
    w,
    h,
    text: value,
    fontSize: options.size ?? 16,
    color: options.color ?? C.text,
    bold: options.bold ?? false,
    align: options.align ?? "left",
    valign: options.valign ?? "top",
    fill: options.fill ?? "#00000000",
    line: options.line ?? ln("#00000000", 0),
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    face: options.face ?? "Arial",
    name: options.name,
  });
}

function anchor(x, y) {
  return shape(x - 0.5, y - 0.5, 1, 1, { fill: "#00000000", line: ln("#00000000", 0) });
}

function lineSeg(x1, y1, x2, y2, options = {}) {
  const from = anchor(x1, y1);
  const to = anchor(x2, y2);
  const connector = slide.shapes.connect(from, to, {
    kind: "straight",
    line: ln(options.color ?? C.black, options.width ?? 1.6, options.style ?? "solid"),
  });
  connector.bringToFront();
  return connector;
}

function arrow(x1, y1, x2, y2, options = {}) {
  const color = options.color ?? C.black;
  const width = options.width ?? 1.8;
  lineSeg(x1, y1, x2, y2, { color, width, style: options.style ?? "solid" });
  const size = options.headSize ?? 11;
  const rotation = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI + 90;
  shape(x2 - size / 2, y2 - size / 2, size, size, {
    geometry: "triangle",
    fill: color,
    line: ln(color, 0),
    rotation,
  }).bringToFront();
}

function panel(x, y, w, h, options = {}) {
  shape(x, y, w, h, {
    geometry: "roundRect",
    fill: options.fill ?? C.panel,
    line: ln(options.lineColor ?? C.line, options.lineWidth ?? 1.2),
  });
}

function stepBadge(x, y, value, color = C.blue) {
  shape(x, y, 31, 36, { geometry: "ellipse", fill: color, line: ln(color, 0) });
  text(x, y + 6, 31, 20, value, { size: 17, bold: true, color: "#FFFFFF", align: "center" });
}

function dashedRuleBox(x, y, w, h) {
  shape(x, y, w, h, {
    geometry: "roundRect",
    fill: "#00000000",
    line: ln(C.rule, 1.1, "dash"),
  });
}

function sectionHeader(x, y, number, title, options = {}) {
  stepBadge(x, y, number, options.badgeColor ?? C.blue);
  text(x + 46, y + 3, options.width ?? 200, 42, title, {
    size: options.size ?? 18,
    bold: true,
    color: options.color ?? C.black,
    align: options.align ?? "left",
  });
}

function gpuIcon(x, y, s = 1, options = {}) {
  const w = 41 * s;
  const h = 50 * s;
  const red = Boolean(options.failed);
  const border = red ? C.red : C.black;
  shape(x, y, w, h, {
    geometry: "roundRect",
    fill: red ? C.redSoft : "#FFFFFF",
    line: ln(border, red ? 1.8 : 1.3),
  });
  shape(x + 8 * s, y + 8 * s, 25 * s, 8 * s, {
    geometry: "roundRect",
    fill: red ? C.red : "#222222",
    line: ln(red ? C.red : "#222222", 0),
  });
  text(x + 8 * s, y + 19 * s, 25 * s, 11 * s, "GPU", {
    size: 7.6 * s,
    color: border,
    align: "center",
    bold: true,
  });
  for (let i = 0; i < 3; i += 1) {
    shape(x + (9 + i * 9) * s, y + 37 * s, 3.6 * s, 3.6 * s, {
      geometry: "ellipse",
      fill: "#FFFFFF",
      line: ln(border, 0.8 * s),
    });
  }
  if (red) {
    text(x + 4 * s, y + 3 * s, w - 8 * s, h - 4 * s, "X", {
      size: 20 * s,
      color: C.red,
      bold: true,
      align: "center",
      valign: "middle",
    });
  }
}

function workerRow(x, y, rowLabel, failedIndex = -1) {
  text(x - 42, y + 17, 34, 18, rowLabel, { size: 15, bold: true, color: C.blue, align: "right" });
  const xs = [x, x + 63, x + 126, x + 189];
  xs.forEach((gx, idx) => {
    gpuIcon(gx, y, 1, { failed: idx === failedIndex });
    if (idx < xs.length - 1) arrow(gx + 42, y + 25, xs[idx + 1] - 6, y + 25, { width: 1.6, headSize: 9 });
  });
}

function serverIcon(x, y, s = 1) {
  shape(x, y, 34 * s, 48 * s, { geometry: "roundRect", fill: "#F7F7F7", line: ln(C.black, 1.2 * s) });
  shape(x + 6 * s, y + 7 * s, 22 * s, 7 * s, { geometry: "rect", fill: "#333333", line: ln("#333333", 0) });
  text(x + 6 * s, y + 18 * s, 22 * s, 10 * s, "CPU", { size: 6.8 * s, color: C.black, align: "center" });
  shape(x + 13 * s, y + 36 * s, 6 * s, 6 * s, { geometry: "ellipse", fill: C.greenSoft, line: ln(C.green, 1 * s) });
}

function checkpoint(x, y, w = 62, h = 58, options = {}) {
  const stroke = options.color ?? C.black;
  const fill = options.fill ?? "#FFFFFF";
  shape(x, y + 12, w, h - 20, { geometry: "rect", fill, line: ln(stroke, 1.4) });
  shape(x, y, w, 24, { geometry: "ellipse", fill, line: ln(stroke, 1.4) });
  shape(x, y + h - 26, w, 24, { geometry: "ellipse", fill, line: ln(stroke, 1.4) });
  [0, 1, 2].forEach((i) => shape(x + 15 + i * 15, y + 31, 8, 17, {
    geometry: "rect",
    fill: "#F58B2B",
    line: ln(C.orange, 1),
  }));
}

function docIcon(x, y, w = 82, h = 110) {
  shape(x, y, w, h, { geometry: "roundRect", fill: "#F8FBFF", line: ln(C.blue, 3) });
  shape(x + w - 31, y + 1, 30, 36, { geometry: "triangle", fill: "#DDEBFF", line: ln(C.blue, 3), rotation: 135 });
  [0, 1, 2].forEach((i) => shape(x + 22, y + 43 + i * 20, 45, 4, {
    geometry: "roundRect",
    fill: C.blue,
    line: ln(C.blue, 0),
  }));
}

function magnifierIcon(x, y, s = 1) {
  shape(x, y, 27 * s, 27 * s, { geometry: "ellipse", fill: "#FFFFFF", line: ln(C.blue, 2.2 * s) });
  lineSeg(x + 22 * s, y + 22 * s, x + 35 * s, y + 37 * s, { color: C.blue, width: 2.3 * s });
}

function clockIcon(x, y, s = 1) {
  shape(x, y, 30 * s, 30 * s, { geometry: "ellipse", fill: "#FFFFFF", line: ln(C.blue, 2 * s) });
  lineSeg(x + 15 * s, y + 15 * s, x + 15 * s, y + 8 * s, { color: C.blue, width: 1.6 * s });
  lineSeg(x + 15 * s, y + 15 * s, x + 10 * s, y + 20 * s, { color: C.blue, width: 1.6 * s });
}

function traceIcon(x, y, s = 1) {
  const pts = [[10, 30], [24, 10], [38, 30]];
  lineSeg(x + pts[0][0] * s, y + pts[0][1] * s, x + pts[1][0] * s, y + pts[1][1] * s, { color: C.blue, width: 1.7 * s });
  lineSeg(x + pts[1][0] * s, y + pts[1][1] * s, x + pts[2][0] * s, y + pts[2][1] * s, { color: C.blue, width: 1.7 * s });
  pts.forEach(([px, py]) => shape(x + px * s - 4 * s, y + py * s - 4 * s, 8 * s, 8 * s, {
    geometry: "ellipse",
    fill: "#FFFFFF",
    line: ln(C.blue, 1.6 * s),
  }));
}

function metricsIcon(x, y, s = 1) {
  [12, 23, 34].forEach((bx, idx) => shape(x + bx * s, y + (28 - idx * 8) * s, 5 * s, (10 + idx * 8) * s, {
    geometry: "rect",
    fill: "#DDEBFF",
    line: ln(C.blue, 1.8 * s),
  }));
}

function infoCard(x, y, w, h, iconFn, label) {
  shape(x, y, w, h, { geometry: "roundRect", fill: C.blueSoft, line: ln(C.blue, 1.2) });
  iconFn(x + 12, y + 16, 0.8);
  text(x + 55, y + 17, w - 65, h - 25, label, { size: 13.5, color: C.black, align: "left", valign: "middle" });
}

function phaseBox(x, y, w, h, label, color, fill) {
  shape(x, y, w, h, { geometry: "roundRect", fill, line: ln(color, 1.3) });
  text(x + 7, y + 10, w - 14, h - 17, label, { size: 12, bold: true, color, align: "center", valign: "middle" });
}

function stateBox(x, y, w, h, label, color, fill, size = 11) {
  shape(x, y, w, h, { geometry: "roundRect", fill, line: ln(color, 1.2) });
  text(x + 4, y + 9, w - 8, h - 13, label, { size, color, bold: true, align: "center", valign: "middle" });
}

shape(0, 0, W, H, { fill: "#FFFFFF", line: ln("#FFFFFF", 0) });
text(0, 18, W, 55, "MoEGambit Runtime Architecture", {
  size: 39,
  bold: true,
  color: "#000000",
  align: "center",
});

// Column 1: training job.
panel(17, 106, 285, 790);
sectionHeader(35, 123, "1", "Training Job", { width: 180 });
text(55, 191, 210, 45, "Sparse MoE Distributed\nTraining (3 DP × 4 Pipeline)", {
  size: 15.5,
  bold: true,
  align: "center",
});
["S0", "S1", "S2", "S3"].forEach((s, i) => text(73 + i * 63, 278, 35, 18, s, {
  size: 16,
  bold: true,
  align: "center",
}));
workerRow(67, 311, "DP0");
workerRow(67, 444, "DP1", 2);
text(170, 522, 120, 22, "r (failed rank)", { size: 13.5, color: C.black, align: "center", italic: true });
workerRow(67, 585, "DP2");
lineSeg(33, 700, 285, 700, { color: C.rule, width: 1.1, style: "dash" });
gpuIcon(45, 735, 0.78);
text(95, 745, 170, 20, "Healthy worker", { size: 14, color: C.black });
gpuIcon(45, 798, 0.78, { failed: true });
shape(69, 829, 20, 20, { geometry: "ellipse", fill: C.red, line: ln(C.red, 0) });
text(69, 830, 20, 14, "×", { size: 18, bold: true, color: "#FFFFFF", align: "center" });
text(95, 807, 185, 24, "Failed worker (rank r)", { size: 14, color: C.black });

// Column 2: failure detection and repair.
panel(328, 106, 235, 790);
sectionHeader(344, 123, "2", "Failure Detection +\nSafe-Point Repair (R1)", { width: 170, size: 16 });
shape(381, 232, 138, 76, { geometry: "roundRect", fill: C.redSoft, line: ln(C.red, 1.4) });
text(395, 249, 108, 43, "Failure Event\nevent <r, t, c>", { size: 15.2, color: "#8B1111", align: "center" });
arrow(450, 308, 450, 383, { width: 1.9, headSize: 12 });
shape(346, 382, 202, 244, { geometry: "roundRect", fill: "#FFFFFF", line: ln(C.black, 1.3) });
shape(431, 394, 36, 48, { geometry: "roundRect", fill: "#DBECFF", line: ln(C.blue, 2) });
shape(440, 404, 18, 25, { geometry: "diamond", fill: "#FFFFFF", line: ln(C.blue, 1.6) });
text(438, 409, 22, 16, "+", { size: 14, bold: true, color: C.blue, align: "center" });
text(372, 456, 150, 48, "Repair Controller\n(Safe-Point Guard)", { size: 16, bold: true, align: "center" });
[
  "mark r as RECOVERING",
  "block optimizer commit",
  "discard current iteration",
].forEach((item, i) => {
  shape(360, 523 + i * 31, 7, 7, { geometry: "ellipse", fill: C.blue, line: ln(C.blue, 0) });
  text(377, 515 + i * 31, 165, 21, item, { size: 13.6, color: C.black });
});
dashedRuleBox(343, 706, 206, 122);
text(374, 724, 178, 92, "r  : failed rank\n\nt  : failure step (current)\n\nc  : last checkpoint step", {
  size: 13,
  color: C.black,
});
arrow(302, 469, 328, 469, { width: 1.9, headSize: 12 });
arrow(563, 469, 588, 469, { width: 1.9, headSize: 12 });

// Column 3: guarded policy.
panel(586, 106, 213, 790);
sectionHeader(601, 123, "3", "Guarded Policy (R2)", { width: 160, size: 16 });
text(602, 194, 120, 22, "Policy Metric", { size: 14.5, color: C.blue, bold: true });
shape(598, 219, 190, 76, { geometry: "roundRect", fill: "#FFFFFF", line: ln(C.blue, 1.2) });
text(606, 237, 174, 38, "Φ′(t) =  S(t) + |E_new| Δ\n          N_expert · W", {
  size: 16,
  color: C.black,
  align: "center",
});
text(602, 321, 125, 22, "Inputs", { size: 14, bold: true, color: C.blue });
["Δ = t − c", "peerAvail", "S(t)", "|E_new|"].forEach((item, i) => {
  shape(604, 356 + i * 26, 6, 6, { geometry: "ellipse", fill: C.blue, line: ln(C.blue, 0) });
  text(617, 348 + i * 26, 128, 18, item, { size: 13.2 });
});
text(602, 473, 125, 22, "Thresholds", { size: 14, bold: true, color: C.blue });
["Δ_min", "Δ_max", "Φ_max"].forEach((item, i) => {
  shape(604, 510 + i * 28, 6, 6, { geometry: "ellipse", fill: C.blue, line: ln(C.blue, 0) });
  text(617, 501 + i * 28, 100, 18, item, { size: 13.2 });
});
shape(626, 613, 94, 94, { geometry: "rect", fill: C.yellowSoft, line: ln(C.black, 1.2), rotation: 45 });
shape(659, 636, 27, 36, { geometry: "roundRect", fill: "#DBECFF", line: ln(C.blue, 1.6) });
shape(666, 645, 13, 17, { geometry: "diamond", fill: "#FFFFFF", line: ln(C.blue, 1.1) });
text(635, 666, 86, 45, "guards\npass?\nHYBRID ?", { size: 12.6, bold: true, align: "center" });
arrow(721, 685, 798, 685, { color: C.green, width: 2, headSize: 13 });
text(748, 657, 40, 18, "YES", { size: 14, color: C.green, bold: true, align: "center" });
lineSeg(671, 704, 671, 821, { color: C.red, width: 2 });
arrow(671, 821, 821, 821, { color: C.red, width: 2, headSize: 13 });
text(681, 788, 40, 20, "NO", { size: 14, color: C.red, bold: true });

// Section 4a: hybrid fast path.
panel(821, 94, 402, 610);
stepBadge(841, 107, "4a", C.green);
text(892, 116, 260, 24, "Hybrid Recovery", { size: 18, bold: true, color: C.green });
text(1049, 121, 120, 20, "(fast path)", { size: 13.5, bold: true, color: C.black });

shape(833, 145, 180, 116, { geometry: "roundRect", fill: "#FFFFFF", line: ln(C.blue, 1.1) });
text(847, 158, 145, 18, "Healthy Peer(s) at step t", { size: 12.4, color: C.blue, bold: true, align: "center" });
serverIcon(858, 185, 0.87);
serverIcon(905, 185, 0.87);
text(947, 205, 22, 14, "···", { size: 20, color: C.black, align: "center" });
serverIcon(975, 185, 0.87);

shape(1036, 145, 176, 116, { geometry: "roundRect", fill: "#FFFFFF", line: ln(C.orange, 1.1) });
text(1045, 158, 155, 18, "Checkpoint Shard (step c)", { size: 12.4, color: C.orange, bold: true, align: "center" });
checkpoint(1098, 185, 60, 55, { color: C.black });

text(836, 281, 118, 46, "Path P\n(peer → r')", { size: 14, color: C.blue, bold: true, align: "center" });
["dense/shared", "router", "replicated opt"].forEach((item, i) => {
  shape(831, 337 + i * 20, 5, 5, { geometry: "ellipse", fill: C.blue, line: ln(C.blue, 0) });
  text(844, 328 + i * 20, 132, 18, item, { size: 11.4, color: C.blue });
});
text(1093, 281, 125, 46, "Path C\n(checkpoint → r')", { size: 14, color: C.orange, bold: true, align: "center" });
["local experts", "expert opt state"].forEach((item, i) => {
  shape(1104, 347 + i * 22, 5, 5, { geometry: "ellipse", fill: C.orange, line: ln(C.orange, 0) });
  text(1117, 338 + i * 22, 120, 18, item, { size: 11.6, color: C.orange });
});

arrow(920, 261, 978, 331, { color: C.blue, width: 2, headSize: 12 });
arrow(1117, 261, 1054, 331, { color: C.orange, width: 2, headSize: 12 });
shape(934, 338, 168, 78, { geometry: "roundRect", fill: C.greenSoft, line: ln(C.green, 1.4) });
serverIcon(950, 354, 0.72);
text(986, 358, 92, 42, "Replacement\nRank r′", { size: 12.5, bold: true, align: "center", valign: "middle" });
arrow(1018, 416, 1018, 445, { color: C.green, width: 1.8, headSize: 11 });

dashedRuleBox(829, 458, 383, 83);
text(933, 448, 190, 18, "Two-Phase Recovery Protocol", { size: 12.6, color: C.green, bold: true, align: "center" });
phaseBox(842, 473, 96, 56, "Phase A:\nweights first", C.blue, C.blueSoft);
phaseBox(969, 473, 101, 56, "resume training\nunder barrier", C.green, C.greenSoft);
phaseBox(1095, 473, 105, 56, "Phase B:\noptimizer later", C.orange, C.orangeSoft);
arrow(938, 501, 969, 501, { width: 1.6, headSize: 9 });
arrow(1070, 501, 1095, 501, { color: "#7A2A1A", width: 1.6, headSize: 9 });
arrow(1018, 541, 1018, 571, { color: C.green, width: 1.7, headSize: 10 });

dashedRuleBox(829, 582, 383, 83);
text(909, 576, 230, 18, "Reintegration (Guarded State Machine)", { size: 12.2, color: C.green, bold: true, align: "center" });
stateBox(838, 600, 86, 52, "RECOVERING\n(r′)", C.blue, "#F7FBFF", 10.2);
stateBox(934, 600, 78, 52, "REPAIRED\n(synced)", C.green, C.greenSoft, 10.4);
stateBox(1022, 600, 78, 52, "BARRIER\n(global)", C.orange, C.orangeSoft, 10.4);
stateBox(1110, 600, 80, 52, "HEALTHY\n(active)", C.green, C.greenSoft, 10.4);
arrow(924, 626, 934, 626, { width: 1.4, headSize: 8 });
arrow(1012, 626, 1022, 626, { color: "#7A2A1A", width: 1.4, headSize: 8 });
arrow(1100, 626, 1110, 626, { width: 1.4, headSize: 8 });
shape(1189, 626, 22, 22, { geometry: "ellipse", fill: C.green, line: ln(C.green, 0) });
text(1188, 628, 24, 14, "✓", { size: 18, color: "#FFFFFF", bold: true, align: "center" });

arrow(1223, 272, 1266, 272, { width: 1.7, headSize: 11 });
arrow(1223, 615, 1266, 615, { width: 1.7, headSize: 11 });

// Section 4b: fallback.
panel(821, 720, 402, 185);
stepBadge(836, 728, "4b", C.red);
text(886, 738, 220, 24, "Checkpoint Restart", { size: 18, bold: true, color: C.red });
text(1053, 743, 90, 18, "(fallback)", { size: 13.5, bold: true, color: C.red });
checkpoint(844, 776, 58, 52, { color: C.black });
text(829, 844, 90, 43, "Full Checkpoint\n(step c)", { size: 12.5, align: "center" });
serverIcon(952, 775, 0.72);
serverIcon(984, 775, 0.72);
text(944, 844, 90, 43, "Reload\nAll Ranks", { size: 12.5, align: "center" });
shape(1066, 785, 56, 56, { geometry: "ellipse", fill: C.graySoft, line: ln("#BBBBBB", 1.2) });
text(1077, 795, 35, 30, "↻", { size: 33, bold: true, color: "#555555", align: "center" });
text(1047, 844, 100, 43, "Replay\nt − c", { size: 12.5, align: "center" });
shape(1148, 781, 62, 62, { geometry: "ellipse", fill: C.graySoft, line: ln("#BBBBBB", 1.2) });
shape(1172, 800, 18, 24, { geometry: "triangle", fill: "#555555", line: ln("#555555", 0), rotation: 90 });
text(1130, 844, 105, 43, "Resume\nTraining", { size: 12.5, align: "center" });
arrow(902, 805, 938, 805, { color: C.red, width: 1.8, headSize: 10 });
arrow(1010, 805, 1058, 805, { color: C.red, width: 1.8, headSize: 10 });
arrow(1122, 805, 1144, 805, { color: C.red, width: 1.8, headSize: 10 });
arrow(1223, 802, 1266, 802, { width: 1.7, headSize: 11 });

// Column 5: observability.
panel(1268, 106, 204, 790);
sectionHeader(1284, 123, "5", "Observability +\nLogs (R3)", { width: 140, size: 16 });
docIcon(1322, 204, 82, 120);
text(1304, 357, 150, 48, "Every recovery\ndecision is auditable.", { size: 14.5, align: "center" });
lineSeg(1282, 429, 1456, 429, { color: C.rule, width: 1.1, style: "dash" });
infoCard(1281, 457, 176, 64, magnifierIcon, "policy inputs + reason");
infoCard(1281, 548, 176, 64, clockIcon, "TTTR / TTTFR /\npath latency");
infoCard(1281, 647, 176, 64, traceIcon, "state-machine trace");
infoCard(1281, 732, 176, 86, metricsIcon, "quality metrics\n(loss, ppl,\nload balance)");

// Bottom legend.
const legendY = 958;
function legendArrow(x, label, color, width = 130) {
  arrow(x, legendY, x + 45, legendY, { color, width: 1.7, headSize: 10 });
  text(x + 58, legendY - 10, width, 22, label, { size: 12, color: C.black });
}
legendArrow(74, "Control Flow", C.black, 110);
legendArrow(263, "Path P (Peer State)", C.blue, 155);
legendArrow(475, "Path C (Checkpoint State)", C.orange, 175);
legendArrow(735, "Hybrid Path (Fast)", C.green, 155);
legendArrow(950, "Restart Path (Fallback)", C.red, 170);
dashedRuleBox(1209, 944, 47, 31);
text(1274, 948, 210, 22, "R1/R2/R3: MoEGambit Rules", { size: 12, color: C.black });

await fs.mkdir(path.dirname(previewPng), { recursive: true });
await fs.mkdir(path.dirname(layoutJson), { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(previewPng, Buffer.from(await preview.arrayBuffer()));
const layout = await presentation.export({ slide, format: "layout" });
await fs.writeFile(layoutJson, await layout.text(), "utf8");
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(finalPptx);

console.log(JSON.stringify({ finalPptx, previewPng, layoutJson }, null, 2));
