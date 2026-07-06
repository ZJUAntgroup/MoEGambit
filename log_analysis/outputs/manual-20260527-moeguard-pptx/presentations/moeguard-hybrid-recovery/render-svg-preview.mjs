import fs from "node:fs/promises";
import { Canvas, loadImage } from "/Users/zds/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/@oai+artifact-tool@file+local-deps+-oai-artifact-tool-oai-artifact_tool-2.8.0.tgz/node_modules/@oai/artifact-tool/node_modules/skia-canvas/lib/index.mjs";

const input = "/Users/zds/bsr/log_analysis/figures/innovation1_hybrid_recovery_ppt.svg";
const output = "/Users/zds/bsr/log_analysis/outputs/manual-20260527-moeguard-pptx/presentations/moeguard-hybrid-recovery/assets/innovation1_hybrid_recovery_fallback.png";

const svg = await fs.readFile(input, "utf8");
const image = await loadImage(Buffer.from(svg, "utf8"));
const canvas = new Canvas(1672, 941);
const ctx = canvas.getContext("2d");
ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, 1672, 941);
ctx.drawImage(image, 0, 0, 1672, 941);
await fs.writeFile(output, await canvas.toBuffer("png"));
console.log(output);
