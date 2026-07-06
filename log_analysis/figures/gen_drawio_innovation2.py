#!/usr/bin/env python3
"""Generate native draw.io vector diagram for Innovation #2.

The output recreates the supplied PNG as editable draw.io cells:
four recovery stages, native text, rounded panels, arrows, state blocks,
checkpoint cylinders, and a bottom recovery-time axis.
"""

import html
import os
import textwrap


WIDTH = 1448
HEIGHT = 1086

RED = "#E00008"
BLUE = "#0B55D9"
GREEN = "#0B7A24"
ORANGE = "#E96B00"
PURPLE = "#6A1FB0"
INK = "#1F2933"
AXIS = "#29323A"

RED_FILL = "#FFFAFA"
BLUE_FILL = "#F7FBFF"
GREEN_FILL = "#F8FFF7"
ORANGE_FILL = "#FFFBF3"
PURPLE_FILL = "#FCF6FF"

cells = []
uid = 0


def esc(value):
    return html.escape(str(value), quote=True)


def cid():
    global uid
    uid += 1
    return f"n{uid}"


def add_cell(value, x, y, w, h, style, *, vertex=True, edge=False, points=None):
    cell_id = cid()
    if edge:
        pts_xml = ""
        if points:
            pts_xml = '<Array as="points">' + "".join(
                f'<mxPoint x="{px}" y="{py}"/>' for px, py in points
            ) + "</Array>"
        geom = (
            '<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x}" y="{y}" as="sourcePoint"/>'
            f'<mxPoint x="{x + w}" y="{y + h}" as="targetPoint"/>'
            f"{pts_xml}</mxGeometry>"
        )
        cells.append(
            f'<mxCell id="{cell_id}" value="{esc(value)}" style="{style}" '
            f'edge="1" parent="1">{geom}</mxCell>'
        )
        return cell_id

    geom = f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
    cells.append(
        f'<mxCell id="{cell_id}" value="{esc(value)}" style="{style}" '
        f'vertex="1" parent="1">{geom}</mxCell>'
    )
    return cell_id


def base_text(size=16, color=INK, bold=False, align="center", valign="middle"):
    style = (
        "html=1;whiteSpace=wrap;fillColor=none;strokeColor=none;"
        f"fontSize={size};fontColor={color};align={align};verticalAlign={valign};"
        "fontFamily=Helvetica;"
    )
    if bold:
        style += "fontStyle=1;"
    return style


def text(value, x, y, w, h, size=16, color=INK, bold=False, align="center"):
    return add_cell(value, x, y, w, h, base_text(size, color, bold, align))


def rect(x, y, w, h, stroke, fill="#FFFFFF", width=2, rounded=0, dashed=False, arc=8):
    style = (
        "html=1;whiteSpace=wrap;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={width};"
    )
    if rounded:
        style += f"rounded=1;arcSize={arc};"
    if dashed:
        style += "dashed=1;dashPattern=8 6;"
    return add_cell("", x, y, w, h, style)


def ellipse(x, y, w, h, stroke, fill="#FFFFFF", width=2):
    return add_cell(
        "",
        x,
        y,
        w,
        h,
        "html=1;ellipse;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={width};",
    )


def line(x1, y1, x2, y2, color=INK, width=2, arrow="none", dashed=False, points=None):
    style = (
        "html=1;rounded=0;"
        f"strokeColor={color};strokeWidth={width};endArrow={arrow};endSize=8;"
    )
    if dashed:
        style += "dashed=1;dashPattern=8 6;"
    return add_cell("", x1, y1, x2 - x1, y2 - y1, style, vertex=False, edge=True, points=points)


def stage_badge(num, x, y, color, title, title_w=220):
    ellipse(x, y, 62, 62, color, "#FFFFFF", 4)
    text(str(num), x, y + 2, 62, 58, size=34, color=color, bold=True)
    text(title, x + 82, y + 2, title_w, 60, size=30, color=color, bold=True, align="left")


def panel(x, y, w, h, color, fill):
    rect(x, y, w, h, color, fill, width=1.6, rounded=1, arc=10)


def small_dot(x, y, color=INK, r=5):
    ellipse(x - r, y - r, r * 2, r * 2, color, color, 1)


def rank_icon(x, y, scale=1.0):
    w = 136 * scale
    h = 184 * scale
    rect(x, y, w, h, INK, "#FFFFFF", width=2.4, rounded=1, arc=8)
    for yy in (y + 61 * scale, y + 122 * scale):
        line(x, yy, x + w, yy, INK, width=2.2)
    for yy in (y + 32 * scale, y + 91 * scale, y + 151 * scale):
        small_dot(x + 24 * scale, yy, INK, r=5.5 * scale)


def red_cross_badge(cx, cy, r=45):
    ellipse(cx - r, cy - r, r * 2, r * 2, RED, RED, 1)
    line(cx - 20, cy - 20, cx + 20, cy + 20, "#FFFFFF", width=8)
    line(cx + 20, cy - 20, cx - 20, cy + 20, "#FFFFFF", width=8)


def peer_rank(x, y):
    rect(x, y, 32, 78, INK, "#FFFFFF", width=1.8, rounded=1, arc=6)
    for yy in (y + 26, y + 52):
        line(x, yy, x + 32, yy, "#D0D7DE", width=1)
    for yy in (y + 14, y + 40, y + 66):
        small_dot(x + 8, yy, BLUE, r=2.2)


def dense_icon(x, y, color=BLUE, size=7, gap=4):
    for row in range(3):
        for col in range(3):
            rect(x + col * (size + gap), y + row * (size + gap), size, size, color, color, width=1, rounded=1, arc=3)


def expert_icon(cx, cy, color=PURPLE):
    ellipse(cx - 7, cy - 24, 14, 14, color, "#FFFFFF", 3)
    ellipse(cx - 24, cy + 1, 14, 14, color, "#FFFFFF", 3)
    ellipse(cx + 10, cy + 1, 14, 14, color, "#FFFFFF", 3)


def optimizer_icon(cx, cy, color=ORANGE):
    ellipse(cx - 22, cy - 22, 44, 44, color, "#FFFFFF", 3)
    rect(cx - 28, cy - 28, 56, 28, ORANGE_FILL, ORANGE_FILL, width=0)
    line(cx, cy, cx + 15, cy - 14, color, width=3)
    line(cx - 14, cy - 6, cx - 9, cy - 14, color, width=2)
    line(cx + 14, cy - 6, cx + 20, cy - 14, color, width=2)
    small_dot(cx, cy, color, r=3.5)


def checkpoint(x, y, color=ORANGE):
    add_cell(
        "",
        x,
        y,
        66,
        78,
        "html=1;shape=cylinder3;whiteSpace=wrap;boundedLbl=1;backgroundOutline=1;"
        f"size=15;fillColor=#FFFFFF;strokeColor={color};strokeWidth=3;",
    )


def component_box(x, y, w, h, label, color, fill, kind, dashed=False):
    rect(x, y, w, h, color, fill, width=1.6, rounded=1, dashed=dashed, arc=8)
    if kind == "dense":
        dense_icon(x + w / 2 - 17, y + 20, BLUE)
    elif kind == "experts":
        expert_icon(x + w / 2, y + 37, PURPLE)
    elif kind == "optimizer":
        optimizer_icon(x + w / 2, y + 48, ORANGE)
    text(label, x + 8, y + h - 65, w - 16, 52, size=20, color=color, bold=True)


def replacement_worker(x, y, w, h, *, optimizer_solid=False, optimizer_dashed=False):
    text("Replacement<br>Worker", x, y - 82, w, 70, size=25, color=INK, bold=True)
    rect(x, y, w, h, INK, "#FFFFFF", width=2.4, rounded=1, arc=8)
    pad = 15
    component_box(
        x + pad,
        y + 22,
        w - 2 * pad,
        120,
        "Dense/<br>Router",
        BLUE,
        BLUE_FILL,
        "dense",
    )
    component_box(
        x + pad,
        y + 160,
        w - 2 * pad,
        112,
        "Experts",
        PURPLE,
        PURPLE_FILL,
        "experts",
    )
    if optimizer_solid or optimizer_dashed:
        component_box(
            x + pad,
            y + h - 128,
            w - 2 * pad,
            120,
            "Optimizer<br>State",
            ORANGE,
            ORANGE_FILL,
            "optimizer",
            dashed=optimizer_dashed,
        )


def check_badge(cx, cy, r=38):
    ellipse(cx - r, cy - r, r * 2, r * 2, GREEN, GREEN, 1)
    line(cx - 17, cy, cx - 4, cy + 15, "#FFFFFF", width=7)
    line(cx - 4, cy + 15, cx + 21, cy - 19, "#FFFFFF", width=7)


def training_pill(x, y):
    rect(x, y, 270, 75, GREEN, GREEN_FILL, width=1.5, rounded=1, arc=10)
    ellipse(x + 16, y + 15, 44, 44, GREEN, GREEN, 1)
    line(x + 28, y + 36, x + 38, y + 47, "#FFFFFF", width=5)
    line(x + 38, y + 47, x + 55, y + 25, "#FFFFFF", width=5)
    text("Training can resume", x + 68, y + 18, 190, 38, size=21, color=GREEN, bold=True, align="left")


def transition_arrow(x, y):
    add_cell(
        "",
        x,
        y,
        34,
        34,
        f"html=1;shape=mxgraph.arrows2.arrow;dy=0.32;dx=20;notch=0;fillColor={AXIS};strokeColor={AXIS};",
    )


# Background
rect(0, 0, WIDTH, HEIGHT, "#FFFFFF", "#FFFFFF", width=0)

# Stage headings
stage_badge(1, 60, 120, RED, "Failure", title_w=180)
stage_badge(2, 400, 120, BLUE, "Phase 1:<br>Weights First", title_w=250)
stage_badge(3, 827, 120, ORANGE, "Phase 2:<br>Optimizer Later", title_w=270)
stage_badge(4, 1202, 120, GREEN, "Fully<br>Recovered", title_w=150)

# Panel frames
panel(38, 223, 230, 644, RED, RED_FILL)
panel(318, 223, 425, 644, BLUE, BLUE_FILL)
panel(789, 223, 340, 644, ORANGE, ORANGE_FILL)
panel(1177, 223, 230, 644, GREEN, GREEN_FILL)

# Stage 1
text("Worker / Rank", 70, 258, 170, 42, size=27, color=INK, bold=True)
rank_icon(84, 327)
red_cross_badge(194, 539, 43)

# Transitions
transition_arrow(278, 484)
transition_arrow(750, 484)
transition_arrow(1136, 484)

# Stage 2
rect(338, 255, 160, 195, BLUE, "#FFFFFF", width=1.4, rounded=1, arc=7)
text("Peer", 358, 265, 120, 28, size=22, color=BLUE, bold=True)
peer_rank(354, 310)
peer_rank(401, 310)
peer_rank(448, 310)
text(". . .", 392, 395, 50, 34, size=26, color=INK, bold=True)
ellipse(479, 426, 52, 52, BLUE, BLUE, 1)
dense_icon(494, 440, "#FFFFFF", size=7, gap=4)
text("Dense/<br>Router", 465, 488, 90, 58, size=20, color=BLUE, bold=True)

replacement_worker(590, 339, 136, 302)
line(501, 323, 580, 356, BLUE, width=3.2, arrow="classic")
line(502, 361, 580, 392, BLUE, width=3.2, arrow="classic")
line(502, 399, 580, 427, BLUE, width=3.2, arrow="classic")

rect(340, 573, 136, 160, PURPLE, "#FFFFFF", width=1.5, rounded=1, arc=8)
text("Checkpoint", 354, 592, 108, 32, size=24, color=PURPLE, bold=True)
checkpoint(375, 633, PURPLE)
ellipse(487, 655, 50, 50, PURPLE, PURPLE, 1)
expert_icon(512, 682, "#FFFFFF")
text("Experts", 482, 718, 70, 30, size=18, color=PURPLE, bold=True)
line(480, 633, 582, 581, PURPLE, width=3.2, arrow="classic")
training_pill(438, 765)

# Stage 3
text("Checkpoint", 812, 336, 110, 34, size=20, color=ORANGE, bold=True)
checkpoint(827, 374, ORANGE)
line(860, 468, 860, 561, ORANGE, width=3, dashed=True)
line(860, 561, 893, 561, ORANGE, width=3, dashed=True)
line(893, 561, 893, 680, ORANGE, width=3, dashed=True)
line(893, 680, 960, 686, ORANGE, width=3, dashed=True, arrow="classic")
replacement_worker(971, 340, 138, 446, optimizer_dashed=True)

# Stage 4
replacement_worker(1211, 340, 166, 414, optimizer_solid=True)
check_badge(1293, 813, 36)

# Timeline
line(68, 927, 1380, 927, AXIS, width=3.5, arrow="classic")
for x, color in [(157, RED), (495, BLUE), (956, ORANGE), (1281, GREEN)]:
    ellipse(x - 14, 913, 28, 28, color, color, 1)
text("recovery time", 638, 958, 180, 40, size=24, color=INK, bold=True)


header = textwrap.dedent(
    f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <mxfile host="app.diagrams.net" agent="moeguard-generator" version="22.0.0">
      <diagram id="two-phase-recovery" name="MoEGuard Two-Phase Recovery">
        <mxGraphModel dx="{WIDTH}" dy="{HEIGHT}" grid="1" gridSize="10" guides="1"
                      tooltips="1" connect="1" arrows="1" fold="1" page="1"
                      pageScale="1" pageWidth="{WIDTH}" pageHeight="{HEIGHT}"
                      math="0" shadow="0">
          <root>
            <mxCell id="0"/>
            <mxCell id="1" parent="0"/>
    """
)

footer = textwrap.dedent(
    """\
          </root>
        </mxGraphModel>
      </diagram>
    </mxfile>
    """
)

xml = header + "\n".join("        " + c for c in cells) + "\n" + footer

out_dir = "/Users/zds/bsr/log_analysis/figures"
os.makedirs(out_dir, exist_ok=True)
for ext in (".drawio.xml", ".drawio"):
    path = os.path.join(out_dir, f"innovation2_two_phase_recovery{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"Wrote {path}")
print(f"Total cells: {len(cells)}")
