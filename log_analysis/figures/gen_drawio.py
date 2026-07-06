#!/usr/bin/env python3
"""Generate native draw.io diagram for MoEGuard Hybrid Recovery.

Coordinates mirror the TikZ source in innovation1_hybrid_recovery.tex
(canvas 1672x941 px, top-left origin, y grows downward).

Every worker / panel / arrow / cylinder / legend swatch is a native
mxCell so users can edit text, color, position directly in draw.io.
"""
import html
import textwrap

# --- Palette (synced with TikZ \definecolor) -----------------------
RED      = "#E00008"
BLUE     = "#1F65D6"
BLUETXT  = "#004AD5"
GREEN    = "#2C8D2D"
GREENTXT = "#147A12"
PURPLE   = "#6A3FB5"
AMBER    = "#EEA320"

RED_FILL    = "#FFFAFA"
BLUE_FILL   = "#F8FBFF"
GREEN_FILL  = "#F8FFF7"
AMBER_FILL  = "#FFFAF0"
GRAY        = "#7F8794"

# --- Helpers --------------------------------------------------------
_cells = []
_uid = 0
def cid():
    global _uid
    _uid += 1
    return f"n{_uid}"

def cell(value, x, y, w, h, *, style="", vertex=1, edge=0, parent="1",
         source=None, target=None, points=None, fontStyle=0, fontSize=12,
         align="center", verticalAlign="middle", fontColor="#000000",
         html_v=1):
    """Append a vertex or edge cell."""
    nid = cid()
    geom_attrs = f' x="{x}" y="{y}" width="{w}" height="{h}"' if vertex else ""
    if edge:
        edge_extra = f' edge="1" source="{source}" target="{target}"' if (source and target) else ' edge="1"'
        # Edges still need geometry node with point lists optionally
        pts_xml = ""
        if points:
            pts_xml = "<Array as=\"points\">"
            for px, py in points:
                pts_xml += f'<mxPoint x="{px}" y="{py}"/>'
            pts_xml += "</Array>"
        if source is None and target is None:
            # raw coordinate edges (need source/target points)
            geom = f'<mxGeometry relative="1" as="geometry">' \
                   f'<mxPoint x="{x}" y="{y}" as="sourcePoint"/>' \
                   f'<mxPoint x="{x+w}" y="{y+h}" as="targetPoint"/>' \
                   f'{pts_xml}</mxGeometry>'
        else:
            geom = f'<mxGeometry relative="1" as="geometry">{pts_xml}</mxGeometry>'
        _cells.append(
            f'<mxCell id="{nid}" value="{html.escape(value)}" style="{style}"'
            f'{edge_extra} parent="{parent}">{geom}</mxCell>'
        )
        return nid
    base_style = f"fontSize={fontSize};fontColor={fontColor};align={align};verticalAlign={verticalAlign};html={html_v};"
    if fontStyle:
        base_style += f"fontStyle={fontStyle};"
    full_style = base_style + style
    # IMPORTANT: keep < / > / & escaped in the XML attribute (otherwise the
    # file is not well-formed XML). draw.io stores HTML markup as
    # double-escaped entities (e.g. &amp;lt;sub&amp;gt;) and renders them
    # at display time because cell style has html=1. We mimic that by
    # escaping the entire value once -- so "W<sub>1,2</sub>" becomes
    # "W&lt;sub&gt;1,2&lt;/sub&gt;" in the .drawio file, which draw.io
    # then renders correctly as W with subscript "1,2".
    raw = str(value)
    safe = html.escape(raw, quote=True)
    _cells.append(
        f'<mxCell id="{nid}" value="{safe}" style="{full_style}" vertex="1" parent="{parent}">'
        f'<mxGeometry{geom_attrs} as="geometry"/></mxCell>'
    )
    return nid

# --- Worker name helper: "W_{1,2}" -> "W<sub>1,2</sub>" ------------
import re as _re
_WSUB = _re.compile(r"W_\{([^}]+)\}")
def wname(s):
    return _WSUB.sub(lambda m: f"W<sub>{m.group(1)}</sub>", str(s))

# --- Panel containers ----------------------------------------------
def panel(x, y, w, h, color, fill):
    return cell("", x, y, w, h,
        style=f"rounded=1;arcSize=11;whiteSpace=wrap;fillColor={fill};strokeColor={color};strokeWidth=2;")

def panel_divider(x1, y, x2, color):
    return cell("", x1, y, x2-x1, 0,
        style=f"endArrow=none;strokeColor={color};strokeWidth=1.5;html=1;", vertex=0, edge=1)

def stage_badge(cx, cy, label, color, title_text):
    # ball
    cell("", cx-21, cy-21, 42, 42,
        style=f"ellipse;fillColor={color};strokeColor={color};fontColor=#ffffff;fontSize=22;fontStyle=1;",
        html_v=1)
    # number text overlaid (already in ellipse value-less; need a label cell)
    cell(label, cx-21, cy-21, 42, 42,
        style="fillColor=none;strokeColor=none;fontSize=22;fontStyle=1;fontColor=#ffffff;align=center;verticalAlign=middle;")
    # title
    cell(title_text, cx+30, cy-18, 360, 36,
        style=f"fillColor=none;strokeColor=none;fontSize=22;fontStyle=1;fontColor={color};align=left;verticalAlign=middle;")

def worker(x, y, w=75, h=86, label="", style_color=BLUE, fill=BLUE_FILL, dashed=False, text_color="#000000"):
    dash = "dashed=1;dashPattern=8 7;" if dashed else ""
    cell("", x, y, w, h,
        style=f"rounded=1;arcSize=7;fillColor={fill};strokeColor={style_color};strokeWidth=2;{dash}")
    # server icon (3-stack)
    sx, sy = x+20, y+13
    for i in range(3):
        cell("", sx, sy+i*8, 28, 8,
            style="rounded=1;arcSize=2;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.4;")
        # LED dot
        cell("", sx+5, sy+i*8+3, 3, 3,
            style="ellipse;fillColor=#000000;strokeColor=#000000;")
    # name label
    cell(wname(label), x, y+57, w, 20,
        style="fillColor=none;strokeColor=none;fontSize=15;fontStyle=1;align=center;verticalAlign=middle;",
        fontColor=text_color)

def peer_worker(x, y, label):
    """Large peer (donor) box: 102x108"""
    cell("", x, y, 102, 108,
        style=f"rounded=1;arcSize=8;fillColor={GREEN_FILL};strokeColor={GREEN};strokeWidth=2;")
    # server icon
    sx, sy = x+33, y+39
    for i in range(3):
        cell("", sx, sy+i*8, 36, 8,
            style="rounded=1;arcSize=2;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.4;")
        cell("", sx+5, sy+i*8+3, 3, 3,
            style="ellipse;fillColor=#000000;strokeColor=#000000;")
    cell(wname(label), x, y+8, 102, 22,
        style=f"fillColor=none;strokeColor=none;fontSize=18;fontStyle=1;align=center;verticalAlign=middle;fontColor=#000000;")
    cell("(Stage 2)", x, y+82, 102, 18,
        style="fillColor=none;strokeColor=none;fontSize=13;align=center;verticalAlign=middle;fontColor=#000000;")

def cross_icon(x, y, size=42):
    # red X
    cell("", x, y, size, size,
        style=f"shape=mxgraph.basic.x;fillColor=none;strokeColor={RED};strokeWidth=4;")

def warning_icon(x, y):
    cell("", x, y, 50, 44,
        style=f"shape=mxgraph.flowchart.or;fillColor=#ffffff;strokeColor={RED};strokeWidth=3;")
    cell("!", x, y, 50, 44,
        style=f"fillColor=none;strokeColor=none;fontSize=22;fontStyle=1;fontColor={RED};align=center;verticalAlign=middle;")

def checkmark_icon(x, y, size=50):
    cell("", x, y, size, size,
        style=f"ellipse;fillColor=#ffffff;strokeColor={GREEN};strokeWidth=3;")
    cell("✓", x, y, size, size,
        style=f"fillColor=none;strokeColor=none;fontSize=30;fontStyle=1;fontColor={GREEN};align=center;verticalAlign=middle;")

def bulb_icon(x, y, size=50):
    cell("💡", x, y, size, size,
        style="fillColor=none;strokeColor=none;fontSize=34;align=center;verticalAlign=middle;")

def cylinder(x, y, w, h, color=PURPLE, fill="#ffffff", label1="Checkpoint", label2="(latest)"):
    cell("", x, y, w, h,
        style=f"shape=cylinder3;whiteSpace=wrap;boundedLbl=1;backgroundOutline=1;size=15;fillColor={fill};strokeColor={color};strokeWidth=2;")
    cell(f"{label1}\n{label2}", x, y, w, h,
        style="fillColor=none;strokeColor=none;fontSize=14;align=center;verticalAlign=middle;")

def text(value, x, y, w, h, size=15, color="#000000", bold=False, align="left", italic=False):
    style = f"fillColor=none;strokeColor=none;"
    fs = 1 if bold else 0
    if italic:
        fs |= 2
    return cell(value, x, y, w, h,
        style=style, fontSize=size, fontColor=color, align=align, verticalAlign="middle", fontStyle=fs)

def arrow(x1, y1, x2, y2, color=BLUE, width=4, dashed=False, points=None):
    dash = "dashed=1;dashPattern=8 4;" if dashed else ""
    return cell("", x1, y1, x2-x1, y2-y1,
        style=f"endArrow=classic;strokeColor={color};strokeWidth={width};html=1;rounded=0;endSize=8;{dash}",
        vertex=0, edge=1, points=points)

# --- Layout (mirrors TikZ coordinates exactly) ---------------------
# Canvas
cell("", 0, 0, 1672, 941,
    style="fillColor=#ffffff;strokeColor=none;")
text("MoEGuard Hybrid Recovery", 0, 10, 1672, 50, size=30, bold=True, align="center")

# ====== Panel 1: Failure Occurs ======
panel(19, 74, 491, 710, RED, RED_FILL)
panel_divider(19, 138, 510, RED)
stage_badge(179, 106, "1", RED, "Failure Occurs")
warning_icon(115, 153)
text(wname("Worker W_{1,2} fails"), 179, 174, 320, 24, size=16, bold=True, color="#000000")

# stage column headers
for sx, label in [(160,"0"),(260,"1"),(360,"2"),(460,"3")]:
    text(f"Stage {label}", sx-40, 220, 80, 20, size=13, bold=True, align="center")

# pipeline row labels
for py, label in [(300,"Pipeline 0"),(418,"Pipeline 1"),(536,"Pipeline 2")]:
    text(label, 31, py-12, 100, 24, size=15)

# Workers grid
positions = [
    (122,255,"0,0"),(222,255,"0,1"),(322,255,"0,2"),(422,255,"0,3"),
    (122,374,"1,0"),(222,374,"1,1"),                 (422,374,"1,3"),
    (122,492,"2,0"),(222,492,"2,1"),(322,492,"2,2"),(422,492,"2,3"),
]
for x,y,name in positions:
    worker(x, y, label=f"W_{{{name}}}",
           style_color=BLUE, fill=BLUE_FILL)
# Failed worker W_{1,2}
worker(322, 374, label="W_{1,2}", style_color=RED, fill=RED_FILL, text_color=RED)
cross_icon(335, 387, 36)

# Big right arrow to panel 2
arrow(510, 383, 546, 383, color=BLUE, width=4)

# ====== Panel 2: Hybrid Recovery ======
panel(532, 74, 605, 710, BLUE, BLUE_FILL)
panel_divider(532, 138, 1137, BLUE)
stage_badge(733, 106, "2", BLUE, "Hybrid Recovery")

# Peer donors
peer_worker(562, 165, "W_{0,2}")
peer_worker(562, 449, "W_{2,2}")

# Curved peer-pull arrows
arrow(670, 218, 759, 283, color=BLUE, width=4,
      points=[(706,221),(736,245)])
arrow(670, 504, 759, 440, color=BLUE, width=4,
      points=[(706,501),(737,477)])

# Replacement worker (center)
cell("", 720, 300, 114, 132,
    style=f"rounded=1;arcSize=8;fillColor={RED_FILL};strokeColor={RED};strokeWidth=2;dashed=1;dashPattern=8 7;")
# server icon inside
sx,sy = 720+39, 300+17
for i in range(3):
    cell("", sx, sy+i*8, 36, 8,
        style="rounded=1;arcSize=2;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.4;")
    cell("", sx+5, sy+i*8+3, 3, 3,
        style="ellipse;fillColor=#000000;strokeColor=#000000;")
text(wname("W_{1,2}"), 720, 360, 114, 22, size=18, bold=True, align="center")
text("(Stage 2)", 720, 386, 114, 18, size=13, align="center")
text("(replacement)", 720, 415, 114, 16, size=12, bold=True, align="center", color=RED)

# Edge labels for the peer pulls
text("Dense / Router", 770, 200, 200, 22, size=15, bold=True, color=BLUETXT)
text("+ Opt State",    770, 230, 200, 22, size=15, bold=True, color=BLUETXT)
text("Dense / Router", 770, 490, 200, 22, size=15, bold=True, color=BLUETXT)
text("+ Opt State",    770, 520, 200, 22, size=15, bold=True, color=BLUETXT)

# Purple arrow from checkpoint to replacement
arrow(980, 359, 850, 359, color=PURPLE, width=3)
text("Expert State", 870, 376, 140, 20, size=13, bold=True, color=PURPLE, align="center")

# Checkpoint cylinder
cylinder(990, 302, 114, 120, color=PURPLE)

# Two-Phase Recovery sub-panel
tp_x, tp_y = 545, 574
cell("", tp_x, tp_y, 570, 194,
    style=f"rounded=1;arcSize=12;fillColor={BLUE_FILL};strokeColor={BLUE};strokeWidth=2;")
text("Two-Phase Recovery", tp_x, tp_y+12, 570, 24, size=16, bold=True, color=BLUETXT, align="center")

# Phase A
cell("", tp_x+16, tp_y+44, 255, 139,
    style=f"rounded=1;arcSize=10;fillColor=#ffffff;strokeColor={BLUE};strokeWidth=2;")
text("Phase A: Weights First", tp_x+16, tp_y+55, 255, 24, size=14, bold=True, color=BLUETXT, align="center")
# little server icons + labels
sx,sy = tp_x+34, tp_y+91
for i in range(3):
    cell("", sx, sy+i*7, 32, 7,
        style="rounded=1;arcSize=2;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.2;")
text("Dense / Router\nWeights", tp_x+82, tp_y+92, 160, 32, size=13, align="left")
sx,sy = tp_x+34, tp_y+140
for i in range(3):
    cell("", sx, sy+i*7, 32, 7,
        style="rounded=1;arcSize=2;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.2;")
text("Expert Weights", tp_x+82, tp_y+148, 160, 20, size=13, align="left")

# Arrow between phases
arrow(tp_x+278, tp_y+112, tp_x+310, tp_y+112, color=BLUE, width=3)

# Phase B
cell("", tp_x+320, tp_y+44, 230, 139,
    style=f"rounded=1;arcSize=10;fillColor={GREEN_FILL};strokeColor={GREEN};strokeWidth=2;")
text("Phase B: Optimizer Later", tp_x+320, tp_y+66, 230, 24, size=14, bold=True, color=GREENTXT, align="center")
sx,sy = tp_x+340, tp_y+113
for i in range(3):
    cell("", sx, sy+i*7, 32, 7,
        style="rounded=1;arcSize=2;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.2;")
text("Optimizer States", tp_x+387, tp_y+125, 160, 20, size=13, align="left")

# Arrow to panel 3
arrow(1135, 383, 1171, 383, color=BLUE, width=4)

# ====== Panel 3: Rebuild & Resume ======
panel(1159, 74, 493, 710, GREEN, GREEN_FILL)
panel_divider(1159, 138, 1652, GREEN)
stage_badge(1296, 106, "3", GREEN, "Rebuild & Resume")

for sx, label in [(1297,"0"),(1395,"1"),(1493,"2"),(1591,"3")]:
    text(f"Stage {label}", sx-40, 218, 80, 20, size=13, bold=True, align="center")
for py, label in [(300,"Pipeline 0"),(418,"Pipeline 1"),(536,"Pipeline 2")]:
    text(label, 1172, py-12, 100, 24, size=15)

green_positions = [
    (1260,255,"0,0"),(1358,255,"0,1"),(1456,255,"0,2"),(1554,255,"0,3"),
    (1260,374,"1,0"),(1358,374,"1,1"),(1456,374,"1,2"),(1554,374,"1,3"),
    (1260,492,"2,0"),(1358,492,"2,1"),(1456,492,"2,2"),(1554,492,"2,3"),
]
for x,y,name in green_positions:
    worker(x, y, w=74, label=f"W_{{{name}}}", style_color=GREEN, fill=GREEN_FILL)

checkmark_icon(1255, 644, 50)
text("All workers healthy.", 1330, 658, 280, 22, size=15, bold=True)
text("Training resumes.",    1330, 687, 280, 22, size=15, bold=True)

# ====== Legend ======
cell("", 73, 818, 1085, 110,
    style=f"rounded=1;arcSize=8;fillColor=#ffffff;strokeColor={GRAY};strokeWidth=1.5;")
text("Legend", 97, 832, 100, 22, size=16, bold=True)

# Healthy
worker(96, 864, w=42, h=42, label="", style_color=BLUE, fill=BLUE_FILL)
text("Healthy", 148, 884, 110, 20, size=13)
# Failed
worker(266, 864, w=42, h=42, label="", style_color=RED, fill=RED_FILL)
cross_icon(274, 872, 26)
text("Failed", 318, 884, 110, 20, size=13)
# Replacement
cell("", 435, 859, 50, 50,
    style=f"rounded=1;arcSize=5;fillColor={RED_FILL};strokeColor={RED};strokeWidth=2;dashed=1;dashPattern=8 7;")
text("Replacement", 497, 884, 130, 20, size=13)
# Pull from Peers
arrow(651, 876, 692, 876, color=BLUE, width=4)
text("Pull from Peers", 707, 866, 240, 18, size=13)
text("(Dense/Router + Opt State)", 707, 892, 240, 18, size=12)
# Restore from Checkpoint
arrow(963, 876, 1004, 876, color=PURPLE, width=3)
text("Restore from", 1018, 866, 160, 18, size=13)
text("Checkpoint",   1018, 892, 160, 18, size=13)

# ====== Example note (amber) ======
cell("", 1227, 828, 347, 84,
    style=f"rounded=1;arcSize=7;fillColor={AMBER_FILL};strokeColor={AMBER};strokeWidth=2;")
bulb_icon(1235, 850)
text(wname("Example: W_{1,2} recovered"),       1304, 856, 280, 22, size=15, bold=True)
text(wname("from W_{0,2} and W_{2,2}."),         1304, 885, 280, 22, size=15, bold=True)

# --- Emit XML ------------------------------------------------------
HEADER = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" agent="moeguard-generator" version="22.0.0">
  <diagram id="hybrid-recovery" name="MoEGuard Hybrid Recovery">
    <mxGraphModel dx="1672" dy="941" grid="1" gridSize="10" guides="1"
                  tooltips="1" connect="1" arrows="1" fold="1" page="1"
                  pageScale="1" pageWidth="1672" pageHeight="941"
                  math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
""")
FOOTER = textwrap.dedent("""\
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
""")

xml = HEADER + "\n".join("        "+c for c in _cells) + "\n" + FOOTER

OUT_DIR = "/Users/zds/bsr/log_analysis/figures"
with open(f"{OUT_DIR}/innovation1_hybrid_recovery.drawio", "w") as f:
    f.write(xml)
with open(f"{OUT_DIR}/innovation1_hybrid_recovery.drawio.xml", "w") as f:
    f.write(xml)

print(f"Wrote {OUT_DIR}/innovation1_hybrid_recovery.drawio")
print(f"Wrote {OUT_DIR}/innovation1_hybrid_recovery.drawio.xml")
print(f"Total cells: {len(_cells)}")
