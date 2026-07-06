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
const finalPptx = path.join(outputDir, "MoEGambit_Runtime_Architecture_IEEE双栏矢量复刻_v2.pptx");
const finalSvg = path.join(outputDir, "MoEGambit_Runtime_Architecture_IEEE双栏矢量复刻_v2.svg");
const previewPng = path.join(workspace, "preview", "moegambit-ieee-vector-replica-v2.png");
const W = 1491;
const H = 1055;

const css = `
  .font { font-family: Arial, Helvetica, sans-serif; }
  .title { font: 700 40px Arial, Helvetica, sans-serif; fill: #000; }
  .head { font: 700 18px Arial, Helvetica, sans-serif; fill: #111; }
  .subhead { font: 700 15px Arial, Helvetica, sans-serif; fill: #111; }
  .text { font: 14px Arial, Helvetica, sans-serif; fill: #111; }
  .small { font: 12px Arial, Helvetica, sans-serif; fill: #111; }
  .mini { font: 9px Arial, Helvetica, sans-serif; fill: #111; }
  .blue { fill: #075db8; }
  .red { fill: #d62828; }
  .green { fill: #188038; }
  .orange { fill: #e65a16; }
  .muted { fill: #4b5563; }
`;

const out = [];
const esc = (s) => String(s)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;");

function push(s) {
  out.push(s);
}

const textStyles = {
  title: { size: 40, weight: 700, fill: "#000" },
  head: { size: 18, weight: 700, fill: "#111" },
  subhead: { size: 15, weight: 700, fill: "#111" },
  text: { size: 14, weight: 400, fill: "#111" },
  small: { size: 12, weight: 400, fill: "#111" },
  mini: { size: 9, weight: 400, fill: "#111" },
};

function text(x, y, value, cls = "text", options = {}) {
  const style = textStyles[cls] ?? textStyles.text;
  const anchor = options.anchor ? ` text-anchor="${options.anchor}"` : "";
  const fontSize = options.size ?? style.size;
  const weight = ` font-weight="${options.weight ?? style.weight}"`;
  const fill = ` fill="${options.fill ?? style.fill}"`;
  const size = ` font-size="${fontSize}"`;
  const family = ` font-family="Arial, Helvetica, sans-serif"`;
  const extra = [anchor, weight, fill, size, family].join("");
  const lines = String(value).split("\n");
  lines.forEach((line, i) => {
    const yy = y + i * (options.lineHeight ?? Math.round(fontSize * 1.22));
    push(`<text x="${x}" y="${yy}"${extra}>${esc(line)}</text>`);
  });
}

function rect(x, y, w, h, options = {}) {
  const rx = options.rx ?? 8;
  const fill = options.fill ?? "#fff";
  const stroke = options.stroke ?? "#8a8a8a";
  const sw = options.sw ?? 1.2;
  const dash = options.dash ? ` stroke-dasharray="${options.dash}"` : "";
  push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash}/>`);
}

function circle(cx, cy, r, fill, stroke = "none", sw = 0) {
  push(`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);
}

function line(x1, y1, x2, y2, color = "#111", sw = 1.7, marker = true, dash = "") {
  const m = marker ? ` marker-end="url(#arrow-${color.slice(1)})"` : "";
  const d = dash ? ` stroke-dasharray="${dash}"` : "";
  push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}" stroke-linecap="round"${m}${d}/>`);
}

function curve(d, color = "#111", sw = 1.7, marker = true, fill = "none") {
  const m = marker ? ` marker-end="url(#arrow-${color.slice(1)})"` : "";
  push(`<path d="${d}" fill="${fill}" stroke="${color}" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round"${m}/>`);
}

function badge(x, y, label, color = "#0b5db8") {
  circle(x + 15, y + 15, 15, color);
  text(x + 15, y + 21, label, "head", { anchor: "middle", fill: "#fff", size: 17 });
}

function panel(x, y, w, h) {
  rect(x, y, w, h, { rx: 8, fill: "#fff", stroke: "#8c8c8c", sw: 1.1 });
}

function gpu(x, y, s = 1, failed = false) {
  const stroke = failed ? "#d62828" : "#111";
  const fill = failed ? "#fff3f3" : "#fff";
  push(`<g transform="translate(${x},${y}) scale(${s})">`);
  push(`<rect x="0" y="0" width="41" height="50" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="${failed ? 1.8 : 1.2}"/>`);
  push(`<rect x="8" y="8" width="25" height="8" rx="3" fill="${failed ? "#d62828" : "#222"}"/>`);
  text(20.5, 27, "GPU", "mini", { anchor: "middle", fill: stroke, weight: 700 });
  [10, 20, 30].forEach((cx) => circle(cx, 39, 2.1, "#fff", stroke, 1));
  if (failed) {
    push(`<path d="M14 18 L27 33 M27 18 L14 33" stroke="#d62828" stroke-width="3" stroke-linecap="round"/>`);
  }
  push(`</g>`);
}

function server(x, y, s = 1) {
  push(`<g transform="translate(${x},${y}) scale(${s})">`);
  rect(0, 0, 34, 48, { rx: 5, fill: "#f7f7f7", stroke: "#111", sw: 1.1 });
  rect(7, 7, 20, 6, { rx: 2, fill: "#222", stroke: "#222", sw: 0 });
  text(17, 26, "CPU", "mini", { anchor: "middle", size: 7 });
  circle(17, 38, 3, "#f2fbf2", "#188038", 1);
  push(`</g>`);
}

function checkpoint(x, y, s = 1) {
  push(`<g transform="translate(${x},${y}) scale(${s})">`);
  push(`<path d="M0 16 C0 5 58 5 58 16 V50 C58 61 0 61 0 50 Z" fill="#fff" stroke="#111" stroke-width="1.4"/>`);
  push(`<ellipse cx="29" cy="16" rx="29" ry="11" fill="#fff" stroke="#111" stroke-width="1.4"/>`);
  [15, 29, 43].forEach((cx) => rect(cx - 4, 31, 8, 17, { rx: 0, fill: "#f58b2b", stroke: "#e65a16", sw: 1 }));
  push(`</g>`);
}

function documentIcon(x, y) {
  push(`<g transform="translate(${x},${y})">`);
  push(`<path d="M0 0 H58 L82 28 V110 H0 Z" fill="#f8fbff" stroke="#0b5db8" stroke-width="3.5" stroke-linejoin="round"/>`);
  push(`<path d="M58 0 V31 H82" fill="none" stroke="#0b5db8" stroke-width="3.5" stroke-linejoin="round"/>`);
  [43, 63, 83].forEach((yy) => rect(23, yy, 44, 4, { rx: 2, fill: "#0b5db8", stroke: "#0b5db8", sw: 0 }));
  push(`</g>`);
}

function magnifier(x, y, s = 1) {
  push(`<g transform="translate(${x},${y}) scale(${s})">`);
  circle(15, 15, 13, "#fff", "#0b5db8", 2.1);
  line(25, 25, 39, 39, "#0b5db8", 2.2, false);
  push(`</g>`);
}

function clock(x, y, s = 1) {
  push(`<g transform="translate(${x},${y}) scale(${s})">`);
  circle(16, 16, 15, "#fff", "#0b5db8", 2);
  line(16, 16, 16, 8, "#0b5db8", 1.7, false);
  line(16, 16, 10, 21, "#0b5db8", 1.7, false);
  push(`</g>`);
}

function trace(x, y, s = 1) {
  push(`<g transform="translate(${x},${y}) scale(${s})">`);
  line(8, 32, 24, 10, "#0b5db8", 1.8, false);
  line(24, 10, 40, 32, "#0b5db8", 1.8, false);
  [[8, 32], [24, 10], [40, 32]].forEach(([cx, cy]) => circle(cx, cy, 4, "#fff", "#0b5db8", 1.6));
  push(`</g>`);
}

function metrics(x, y, s = 1) {
  push(`<g transform="translate(${x},${y}) scale(${s})">`);
  [12, 24, 36].forEach((xx, i) => rect(xx, 31 - i * 9, 6, 11 + i * 9, { rx: 0, fill: "#ddebff", stroke: "#0b5db8", sw: 1.8 }));
  push(`</g>`);
}

function infoCard(x, y, h, icon, label) {
  rect(x, y, 176, h, { rx: 7, fill: "#f3f8ff", stroke: "#0b5db8", sw: 1.2 });
  icon(x + 13, y + 17, 0.82);
  text(x + 58, y + 30, label, "text");
}

function formula(x, y) {
  rect(x, y, 190, 76, { rx: 7, fill: "#fff", stroke: "#0b5db8", sw: 1.15 });
  text(x + 12, y + 42, "Φ′(t) =", "text", { size: 15 });
  text(x + 132, y + 29, "S(t) + |E_new| Δ", "small", { size: 12.6, anchor: "middle" });
  line(x + 80, y + 38, x + 176, y + 38, "#111", 1.1, false);
  text(x + 132, y + 59, "N_expert · W", "small", { size: 12.6, anchor: "middle" });
}

function stepHeader(x, y, number, title, color = "#0b5db8", width = 160) {
  badge(x, y, number, color);
  text(x + 44, y + 20, title, "head", { size: 16 });
}

function phase(x, y, w, h, label, color, fill) {
  rect(x, y, w, h, { rx: 7, fill, stroke: color, sw: 1.25 });
  text(x + w / 2, y + 22, label, "small", { anchor: "middle", fill: color, weight: 700 });
}

push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`);
push(`<defs>`);
["111111", "0b5db8", "e65a16", "188038", "d62828"].forEach((color) => {
  push(`<marker id="arrow-${color}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0 0 L9 4.5 L0 9 Z" fill="#${color}"/></marker>`);
});
push(`</defs>`);
push(`<style>${css}</style>`);
rect(0, 0, W, H, { rx: 0, fill: "#fff", stroke: "#fff", sw: 0 });
text(W / 2, 52, "MoEGambit Runtime Architecture", "title", { anchor: "middle" });

panel(17, 106, 285, 790);
stepHeader(36, 122, "1", "Training Job");
text(160, 205, "Sparse MoE Distributed\nTraining (3 DP × 4 Pipeline)", "subhead", { anchor: "middle" });
["S0", "S1", "S2", "S3"].forEach((s, i) => text(88 + i * 62, 292, s, "subhead", { anchor: "middle" }));
function row(y, label, failed = -1) {
  text(56, y + 34, label, "text", { fill: "#0b5db8", weight: 700, anchor: "end" });
  [0, 1, 2, 3].forEach((i) => {
    gpu(67 + i * 63, y, 1, i === failed);
    if (i < 3) line(108 + i * 63, y + 25, 130 + i * 63, y + 25);
  });
}
row(311, "DP0");
row(444, "DP1", 2);
text(222, 534, "r (failed rank)", "small", { anchor: "middle" });
row(585, "DP2");
line(35, 700, 285, 700, "#7d8797", 1.1, false, "6 5");
gpu(45, 735, 0.78);
text(95, 758, "Healthy worker", "text");
gpu(45, 798, 0.78, true);
circle(79, 840, 14, "#d62828");
text(79, 846, "×", "head", { anchor: "middle", fill: "#fff", size: 20 });
text(95, 823, "Failed worker (rank r)", "text");

panel(328, 106, 235, 790);
stepHeader(344, 122, "2", "Failure Detection +\nSafe-Point Repair (R1)");
rect(381, 232, 138, 76, { rx: 7, fill: "#fff3f3", stroke: "#d62828", sw: 1.3 });
text(450, 263, "Failure Event\nevent <r,t,c>", "text", { anchor: "middle", fill: "#8b1111" });
line(450, 308, 450, 382);
rect(346, 382, 202, 244, { rx: 8, fill: "#fff", stroke: "#111", sw: 1.25 });
rect(431, 394, 36, 48, { rx: 8, fill: "#ddebff", stroke: "#0b5db8", sw: 2 });
push(`<path d="M449 405 L458 414 L449 430 L440 414 Z" fill="#fff" stroke="#0b5db8" stroke-width="1.6"/>`);
text(449, 420, "+", "text", { anchor: "middle", fill: "#0b5db8", weight: 700 });
text(447, 476, "Repair Controller\n(Safe-Point Guard)", "head", { anchor: "middle", size: 15 });
["mark r as RECOVERING", "block optimizer commit", "discard current iteration"].forEach((s, i) => {
  circle(363, 526 + i * 31, 4, "#0b5db8");
  text(377, 531 + i * 31, s, "small");
});
rect(343, 706, 206, 122, { rx: 7, fill: "none", stroke: "#0b5db8", sw: 1.1, dash: "5 4" });
text(374, 739, "r  : failed rank", "small");
text(374, 781, "t  : failure step (current)", "small");
text(374, 823, "c  : last checkpoint step", "small");
line(302, 469, 328, 469);
line(563, 469, 588, 469);

panel(586, 106, 213, 790);
stepHeader(601, 122, "3", "Guarded Policy (R2)");
text(602, 207, "Policy Metric", "subhead", { fill: "#0b5db8" });
formula(598, 219);
text(602, 334, "Inputs", "subhead", { fill: "#0b5db8" });
["Δ = t − c", "peerAvail", "S(t)", "|E_new|"].forEach((s, i) => {
  circle(605, 358 + i * 26, 4, "#0b5db8");
  text(617, 363 + i * 26, s, "small");
});
text(602, 486, "Thresholds", "subhead", { fill: "#0b5db8" });
["Δ_min", "Δ_max", "Φ_max"].forEach((s, i) => {
  circle(605, 512 + i * 28, 4, "#0b5db8");
  text(617, 517 + i * 28, s, "small");
});
push(`<path d="M674 615 L728 669 L674 723 L620 669 Z" fill="#fff8d8" stroke="#111" stroke-width="1.2"/>`);
rect(660, 636, 28, 42, { rx: 8, fill: "#ddebff", stroke: "#0b5db8", sw: 1.8 });
push(`<path d="M674 647 L685 660 L674 674 L663 660 Z" fill="#fff" stroke="#0b5db8" stroke-width="1.1"/>`);
text(674, 690, "guards pass?\nHYBRID ?", "small", { anchor: "middle", weight: 700 });
line(722, 685, 798, 685, "#188038", 2);
text(762, 671, "YES", "subhead", { anchor: "middle", fill: "#188038" });
line(674, 723, 674, 822, "#d62828", 2, false);
line(674, 822, 821, 822, "#d62828", 2);
text(695, 802, "NO", "subhead", { fill: "#d62828" });

panel(821, 94, 402, 610);
badge(840, 107, "4a", "#188038");
text(892, 133, "Hybrid Recovery", "head", { fill: "#188038", size: 18 });
text(1054, 133, "(fast path)", "subhead", { size: 13 });
rect(833, 145, 180, 116, { rx: 7, fill: "#fff", stroke: "#0b5db8", sw: 1.1 });
text(923, 172, "Healthy Peer(s) at step t", "small", { anchor: "middle", fill: "#0b5db8", weight: 700 });
server(858, 185, 0.86); server(905, 185, 0.86); text(955, 216, "···", "head", { anchor: "middle" }); server(975, 185, 0.86);
rect(1036, 145, 176, 116, { rx: 7, fill: "#fff", stroke: "#e65a16", sw: 1.1 });
text(1124, 172, "Checkpoint Shard (step c)", "small", { anchor: "middle", fill: "#e65a16", weight: 700 });
checkpoint(1098, 185, 1);
text(881, 303, "Path P\n(peer → r′)", "subhead", { anchor: "middle", fill: "#0b5db8" });
["dense/shared", "router", "replicated opt"].forEach((s, i) => {
  circle(832, 342 + i * 20, 3, "#0b5db8");
  text(844, 346 + i * 20, s, "small", { fill: "#0b5db8", size: 11 });
});
text(1148, 303, "Path C\n(checkpoint → r′)", "subhead", { anchor: "middle", fill: "#e65a16" });
["local experts", "expert opt state"].forEach((s, i) => {
  circle(1105, 351 + i * 22, 3, "#e65a16");
  text(1117, 355 + i * 22, s, "small", { fill: "#e65a16", size: 11 });
});
curve("M920 261 C940 295 960 313 984 333", "#0b5db8", 2);
curve("M1117 261 C1098 294 1073 315 1054 333", "#e65a16", 2);
rect(934, 338, 168, 78, { rx: 7, fill: "#f2fbf2", stroke: "#188038", sw: 1.35 });
server(950, 354, 0.72);
text(1033, 367, "Replacement\nRank r′", "subhead", { anchor: "middle" });
line(1018, 416, 1018, 445, "#188038", 1.8);
rect(829, 458, 383, 83, { rx: 6, fill: "none", stroke: "#7d8797", sw: 1.1, dash: "5 5" });
text(1020, 463, "Two-Phase Recovery Protocol", "small", { anchor: "middle", fill: "#188038", weight: 700 });
phase(842, 473, 96, 56, "Phase A:\nweights first", "#0b5db8", "#f3f8ff");
phase(969, 473, 101, 56, "resume training\nunder barrier", "#188038", "#f2fbf2");
phase(1095, 473, 105, 56, "Phase B:\noptimizer later", "#e65a16", "#fff6ed");
line(938, 501, 969, 501);
line(1070, 501, 1095, 501, "#7a2a1a", 1.6);
line(1018, 541, 1018, 571, "#188038", 1.7);
rect(829, 582, 383, 83, { rx: 6, fill: "none", stroke: "#7d8797", sw: 1.1, dash: "5 5" });
text(1020, 590, "Reintegration (Guarded State Machine)", "small", { anchor: "middle", fill: "#188038", weight: 700 });
[
  [838, "RECOVERING\n(r′)", "#0b5db8", "#f7fbff", 86],
  [934, "REPAIRED\n(synced)", "#188038", "#f2fbf2", 78],
  [1022, "BARRIER\n(global)", "#e65a16", "#fff6ed", 78],
  [1110, "HEALTHY\n(active)", "#188038", "#f2fbf2", 80],
].forEach(([x, label, color, fill, w]) => {
  rect(x, 600, w, 52, { rx: 7, fill, stroke: color, sw: 1.15 });
  text(x + w / 2, 622, label, "small", { anchor: "middle", fill: color, weight: 700, size: 10.5 });
});
line(924, 626, 934, 626); line(1012, 626, 1022, 626, "#7a2a1a", 1.4); line(1100, 626, 1110, 626);
circle(1201, 626, 12, "#188038"); text(1201, 631, "✓", "head", { anchor: "middle", fill: "#fff", size: 19 });
line(1223, 272, 1266, 272); line(1223, 615, 1266, 615);

panel(821, 720, 402, 185);
badge(836, 728, "4b", "#d62828");
text(886, 755, "Checkpoint Restart", "head", { fill: "#d62828", size: 18 });
text(1054, 755, "(fallback)", "subhead", { fill: "#d62828", size: 13 });
checkpoint(844, 776, 1);
text(873, 862, "Full Checkpoint\n(step c)", "small", { anchor: "middle" });
server(952, 775, 0.72); server(984, 775, 0.72);
text(990, 862, "Reload\nAll Ranks", "small", { anchor: "middle" });
circle(1094, 813, 35, "#f7f7f7", "#bbb", 1.2); text(1094, 828, "↻", "head", { anchor: "middle", fill: "#555", size: 36 });
text(1094, 862, "Replay\nt − c", "small", { anchor: "middle" });
circle(1176, 813, 35, "#f7f7f7", "#bbb", 1.2); push(`<path d="M1164 797 L1193 813 L1164 829 Z" fill="#555"/>`);
text(1176, 862, "Resume\nTraining", "small", { anchor: "middle" });
line(902, 805, 938, 805, "#d62828", 1.8); line(1010, 805, 1058, 805, "#d62828", 1.8); line(1122, 805, 1144, 805, "#d62828", 1.8);
line(1223, 802, 1266, 802);

panel(1268, 106, 204, 790);
stepHeader(1284, 122, "5", "Observability +\nLogs (R3)");
documentIcon(1322, 204);
text(1370, 389, "Every recovery\ndecision is auditable.", "text", { anchor: "middle" });
line(1282, 429, 1456, 429, "#7d8797", 1.1, false, "5 6");
infoCard(1281, 457, 64, magnifier, "policy inputs +\nreason");
infoCard(1281, 548, 64, clock, "TTTR / TTTFR /\npath latency");
infoCard(1281, 647, 64, trace, "state-machine\ntrace");
infoCard(1281, 732, 86, metrics, "quality metrics\n(loss, ppl,\nload balance)");

function legend(x, label, color) {
  line(x, 958, x + 45, 958, color, 1.7);
  text(x + 58, 963, label, "small");
}
legend(74, "Control Flow", "#111111");
legend(263, "Path P (Peer State)", "#0b5db8");
legend(475, "Path C (Checkpoint State)", "#e65a16");
legend(735, "Hybrid Path (Fast)", "#188038");
legend(950, "Restart Path (Fallback)", "#d62828");
rect(1209, 944, 47, 31, { rx: 6, fill: "none", stroke: "#7d8797", sw: 1.1, dash: "5 5" });
text(1274, 963, "R1/R2/R3: MoEGambit Rules", "small");

push(`</svg>`);

const svg = out.join("\n");
await fs.mkdir(path.dirname(finalSvg), { recursive: true });
await fs.mkdir(path.dirname(previewPng), { recursive: true });
await fs.writeFile(finalSvg, svg, "utf8");

await ensureArtifactToolWorkspace(workspace);
const artifact = await importArtifactTool(workspace);
const { Presentation, PresentationFile } = artifact;
const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const slide = presentation.slides.add();
const ctx = createSlideContext(artifact, { slideSize: { width: W, height: H }, workspaceDir: workspace });
await ctx.addImage(slide, {
  x: 0,
  y: 0,
  w: W,
  h: H,
  dataUrl: `data:image/svg+xml;base64,${Buffer.from(svg, "utf8").toString("base64")}`,
  fit: "contain",
  alt: "MoEGambit Runtime Architecture IEEE double-column vector figure",
});

const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(previewPng, Buffer.from(await preview.arrayBuffer()));
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(finalPptx);

console.log(JSON.stringify({ finalPptx, finalSvg, previewPng }, null, 2));
