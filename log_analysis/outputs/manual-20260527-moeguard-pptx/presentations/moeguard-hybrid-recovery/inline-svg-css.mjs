import fs from "node:fs/promises";

const input = "/Users/zds/bsr/log_analysis/figures/innovation1_hybrid_recovery.svg";
const output = "/Users/zds/bsr/log_analysis/figures/innovation1_hybrid_recovery_ppt.svg";

const styles = {
  title: "font-family:Arial,Helvetica,sans-serif;font-size:38px;font-weight:800;fill:#050505",
  "panel-title": "font-family:Arial,Helvetica,sans-serif;font-size:25px;font-weight:800",
  label: "font-family:Arial,Helvetica,sans-serif;font-size:18px;font-weight:700;fill:#050505",
  body: "font-family:Arial,Helvetica,sans-serif;font-size:18px;fill:#050505",
  small: "font-family:Arial,Helvetica,sans-serif;font-size:16px;fill:#050505",
  tiny: "font-family:Arial,Helvetica,sans-serif;font-size:14px;fill:#050505",
  "cell-text": "font-family:Arial,Helvetica,sans-serif;font-size:22px;font-weight:700;fill:#050505",
  sub: "font-size:14px;baseline-shift:sub",
  red: "fill:#e00008",
  blue: "fill:#004ad5",
  green: "fill:#147a12",
  purple: "fill:#6a3fb5",
  "panel-red": "fill:#fffafa;stroke:#e00008;stroke-width:1.4",
  "panel-blue": "fill:#f8fbff;stroke:#1f65d6;stroke-width:1.4",
  "panel-green": "fill:#f8fff7;stroke:#2c8d2d;stroke-width:1.4",
  healthy: "fill:#f8fbff;stroke:#2e73dc;stroke-width:1.2",
  "healthy-green": "fill:#f8fff7;stroke:#2c8d2d;stroke-width:1.2",
  failed: "fill:#fffafa;stroke:#e00008;stroke-width:1.3",
  replacement: "fill:#fffafa;stroke:#e00008;stroke-width:1.3;stroke-dasharray:8 7",
  phase: "fill:#ffffff;stroke:#2e73dc;stroke-width:1.2",
  "phase-green": "fill:#fbfff7;stroke:#2c8d2d;stroke-width:1.2",
  "arrow-blue": "fill:none;stroke:#1f65d6;stroke-width:4;marker-end:url(#arrowBlue)",
  "arrow-purple": "fill:none;stroke:#6a3fb5;stroke-width:3.2;marker-end:url(#arrowPurple)",
  "legend-box": "fill:#ffffff;stroke:#7f8794;stroke-width:1.1",
};

let svg = await fs.readFile(input, "utf8");
svg = svg.replace(/\sclass="([^"]+)"/g, (_match, classNames) => {
  const inline = classNames
    .trim()
    .split(/\s+/)
    .map((className) => styles[className])
    .filter(Boolean)
    .join(";");
  return inline ? ` style="${inline}"` : "";
});
await fs.writeFile(output, svg, "utf8");
console.log(output);
