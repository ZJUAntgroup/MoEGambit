import fs from "node:fs/promises";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  const svg = await fs.readFile("/Users/zds/bsr/log_analysis/figures/innovation1_hybrid_recovery.svg", "utf8");

  await ctx.addImage(slide, {
    dataUrl: `data:image/svg+xml;base64,${Buffer.from(svg, "utf8").toString("base64")}`,
    x: 0,
    y: 0,
    width: ctx.W,
    height: ctx.H,
    fit: "contain",
    alt: "MoEGuard hybrid recovery workflow",
    name: "moeguard-hybrid-recovery-svg",
  });

  return slide;
}
