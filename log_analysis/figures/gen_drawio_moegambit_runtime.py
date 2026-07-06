#!/usr/bin/env python3
"""Generate an editable draw.io vector diagram for MoEGambit runtime architecture.

The generated file recreates the supplied PNG as native draw.io cells:
panels, labels, worker/device icons, checkpoint cylinders, arrows, policy boxes,
recovery state machines, and the bottom legend.
"""

import html
import os
import textwrap


WIDTH = 1491
HEIGHT = 1055

BLUE = "#2362AE"
BLUE_DARK = "#1C2B54"
BLUE_FILL = "#F3F7FC"
BLUE_SOFT = "#F3F7FC"
RED = "#E7322C"
RED_DARK = "#E60012"
RED_FILL = "#FDECEA"
GREEN = "#2F7E39"
GREEN_DARK = "#187038"
GREEN_FILL = "#F5FBF5"
GREEN_SOFT = "#F5FBF5"
ORANGE = "#EF7F33"
ORANGE_DARK = "#E94619"
ORANGE_FILL = "#FFF6EF"
ORANGE_SOFT = "#FFF6EF"
PURPLE = "#3A2A5F"
PURPLE_FILL = "#F7F4F9"
INK = "#231F20"
MUTED = "#5F5F5F"
GRAY = "#959494"
LIGHT = "#F7F7F7"

STAGE_GPU_COLORS = [ORANGE_DARK, BLUE, GREEN, PURPLE]
STAGE_GPU_FILLS = [ORANGE_FILL, BLUE_FILL, GREEN_FILL, PURPLE_FILL]
DAMAGED_GPU_COLOR = GREEN
DAMAGED_GPU_FILL = GREEN_FILL

cells = []
uid = 0


def esc(value):
    return html.escape(str(value), quote=True)


def cid():
    global uid
    uid += 1
    return f"g{uid}"


def add_cell(value, x, y, w, h, style, *, edge=False, points=None):
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


def font(size=16, color=INK, bold=False, align="center", valign="middle"):
    style = (
        "html=1;whiteSpace=wrap;fillColor=none;strokeColor=none;"
        f"fontSize={size};fontColor={color};align={align};verticalAlign={valign};"
        "fontFamily=Helvetica;spacing=0;"
    )
    if bold:
        style += "fontStyle=1;"
    return style


def text(value, x, y, w, h, size=16, color=INK, bold=False, align="center", valign="middle"):
    return add_cell(value, x, y, w, h, font(size, color, bold, align, valign))


def rect(
    x,
    y,
    w,
    h,
    stroke,
    fill="#FFFFFF",
    width=1.5,
    rounded=True,
    arc=8,
    dashed=False,
    dash="8 6",
):
    style = (
        "html=1;whiteSpace=wrap;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={width};"
    )
    if rounded:
        style += f"rounded=1;arcSize={arc};"
    if dashed:
        style += f"dashed=1;dashPattern={dash};"
    return add_cell("", x, y, w, h, style)


def ellipse(x, y, w, h, stroke, fill="#FFFFFF", width=1.5):
    return add_cell(
        "",
        x,
        y,
        w,
        h,
        "html=1;ellipse;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={width};",
    )


def line(x1, y1, x2, y2, color=INK, width=2, arrow="classic", dashed=False, points=None):
    style = (
        "html=1;rounded=0;"
        f"strokeColor={color};strokeWidth={width};endArrow={arrow};endSize=8;"
    )
    if dashed:
        style += "dashed=1;dashPattern=8 6;"
    return add_cell("", x1, y1, x2 - x1, y2 - y1, style, edge=True, points=points)


def right_arrow(x, y, w=31, h=24, color=INK):
    add_cell(
        "",
        x,
        y,
        w,
        h,
        f"html=1;shape=mxgraph.arrows2.arrow;dy=0.32;dx=18;notch=0;fillColor={color};strokeColor={color};strokeWidth=1;",
    )


def panel(x, y, w, h, stroke, fill="#FFFFFF"):
    rect(x, y, w, h, stroke, fill, width=1.15, rounded=True, arc=6)


def badge(x, y, label, color, size=34, font_size=19):
    ellipse(x, y, size, size, color, color, 1.2)
    text(label, x, y + 1, size, size - 2, size=font_size, color="#FFFFFF", bold=True)


def outline_badge(x, y, label, color, size=62, font_size=34):
    ellipse(x, y, size, size, color, "#FFFFFF", 4)
    text(label, x, y + 1, size, size - 2, size=font_size, color=color, bold=True)


def bullet(x, y, color=BLUE, r=3.5):
    ellipse(x - r, y - r, r * 2, r * 2, color, color, 1)


def tiny_text(value, x, y, w, h, size=12, color=INK, bold=False, align="center"):
    text(value, x, y, w, h, size=size, color=color, bold=bold, align=align)


def gpu_icon(x, y, scale=1.0, stroke=INK, fill="#FFFFFF", failed=False):
    card_w = 58 * scale
    card_h = 34 * scale
    color = stroke
    card_y = y + 15 * scale
    rect(x, card_y, card_w, card_h, color, fill, width=1.6, rounded=True, arc=12)

    def fan(cx, cy, r):
        ellipse(cx - r, cy - r, r * 2, r * 2, color, fill, 1.25)
        ellipse(cx - 3.8 * scale, cy - 3.8 * scale, 7.6 * scale, 7.6 * scale, color, fill, 1.2)
        for dx1, dy1, dx2, dy2 in [
            (-0.08, -0.96, 0.34, -0.18),
            (0.82, -0.48, 0.42, 0.22),
            (0.90, 0.28, 0.16, 0.46),
            (0.08, 0.96, -0.34, 0.18),
            (-0.82, 0.48, -0.42, -0.22),
            (-0.90, -0.28, -0.16, -0.46),
        ]:
            line(cx + dx1 * r, cy + dy1 * r, cx + dx2 * r, cy + dy2 * r, color, width=1.05 * scale, arrow="none")

    fan(x + 20.0 * scale, card_y + card_h / 2, 12.5 * scale)
    fan(x + 38.0 * scale, card_y + card_h / 2, 12.5 * scale)


def failed_mark(cx, cy, r=12):
    ellipse(cx - r, cy - r, r * 2, r * 2, RED, RED, 1)
    line(cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45, "#FFFFFF", width=3.5, arrow="none")
    line(cx + r * 0.45, cy - r * 0.45, cx - r * 0.45, cy + r * 0.45, "#FFFFFF", width=3.5, arrow="none")


def worker_grid_icon(x, y, color=BLUE, size=7, gap=4):
    for row in range(3):
        for col in range(3):
            rect(x + col * (size + gap), y + row * (size + gap), size, size, color, color, width=0.9, rounded=True, arc=3)


def shield_icon(x, y, color=BLUE, size=35):
    ellipse(x + size * 0.08, y + size * 0.08, size * 0.84, size * 0.84, color, BLUE_SOFT, 2)
    text("+", x + size * 0.22, y + size * 0.18, size * 0.56, size * 0.48, size=18, color=color, bold=True)


def cylinder(x, y, w=62, h=70, color=ORANGE, fill="#FFFFFF", width=2):
    add_cell(
        "",
        x,
        y,
        w,
        h,
        "html=1;shape=cylinder3;whiteSpace=wrap;boundedLbl=1;backgroundOutline=1;"
        f"size=14;fillColor={fill};strokeColor={color};strokeWidth={width};",
    )


def checkpoint_blocks(x, y, color=ORANGE):
    for i in range(3):
        rect(x + i * 17, y, 10, 28, color, color, width=0.6, rounded=False)


def document_icon(x, y, w=86, h=118, color=BLUE):
    add_cell(
        "",
        x,
        y,
        w,
        h,
        "html=1;shape=document;whiteSpace=wrap;boundedLbl=1;"
        f"fillColor=#FFFFFF;strokeColor={color};strokeWidth=4;",
    )
    for yy in (y + 46, y + 66, y + 86):
        line(x + 21, yy, x + w - 18, yy, color, width=4, arrow="none")


def magnifier_icon(x, y, color=BLUE):
    ellipse(x, y, 22, 22, color, "#FFFFFF", 2.4)
    line(x + 17, y + 18, x + 31, y + 34, color, width=2.4, arrow="none")


def gauge_icon(cx, cy, color=BLUE):
    ellipse(cx - 16, cy - 16, 32, 32, color, "#FFFFFF", 2.2)
    rect(cx - 18, cy - 19, 36, 19, "#FFFFFF", "#FFFFFF", width=0, rounded=False)
    line(cx, cy, cx + 10, cy - 10, color, width=2.2, arrow="none")
    bullet(cx, cy, color, r=2.5)


def trace_icon(x, y, color=BLUE):
    for px, py in ((x, y + 24), (x + 20, y + 8), (x + 40, y + 24)):
        ellipse(px - 3, py - 3, 6, 6, color, "#FFFFFF", 2)
    line(x, y + 24, x + 20, y + 8, color, width=1.7, arrow="none")
    line(x + 20, y + 8, x + 40, y + 24, color, width=1.7, arrow="none")
    line(x, y + 24, x, y + 44, color, width=1.7, arrow="none")
    line(x + 20, y + 8, x + 20, y + 44, color, width=1.7, arrow="none")
    line(x + 40, y + 24, x + 40, y + 44, color, width=1.7, arrow="none")


def bars_icon(x, y, color=BLUE):
    heights = [26, 38, 48]
    for i, h in enumerate(heights):
        rect(x + i * 11, y + 50 - h, 7, h, color, BLUE_SOFT, width=2, rounded=False)


def check_badge(cx, cy, r=17, color=GREEN):
    ellipse(cx - r, cy - r, r * 2, r * 2, color, color, 1)
    line(cx - r * 0.45, cy + r * 0.02, cx - r * 0.12, cy + r * 0.42, "#FFFFFF", width=4, arrow="none")
    line(cx - r * 0.12, cy + r * 0.42, cx + r * 0.55, cy - r * 0.48, "#FFFFFF", width=4, arrow="none")


def tiny_worker(x, y, color=DAMAGED_GPU_COLOR, fill=DAMAGED_GPU_FILL, failed=False, scale=0.75):
    gpu_icon(x, y, scale=scale, stroke=color, fill=fill, failed=failed)


def component_card(x, y, w, h, label, color, fill, kind):
    rect(x, y, w, h, color, fill, width=1.2, rounded=True, arc=8)
    if kind == "dense":
        worker_grid_icon(x + w / 2 - 14, y + 18, BLUE, size=6, gap=3.5)
    elif kind == "experts":
        expert_icon(x + w / 2, y + 38, PURPLE)
    elif kind == "opt":
        gauge_icon(x + w / 2, y + 50, ORANGE)
    text(label, x + 8, y + h - 48, w - 16, 40, size=15, color=color, bold=True)


def expert_icon(cx, cy, color=PURPLE):
    ellipse(cx - 6, cy - 20, 12, 12, color, "#FFFFFF", 2.5)
    ellipse(cx - 21, cy + 3, 12, 12, color, "#FFFFFF", 2.5)
    ellipse(cx + 9, cy + 3, 12, 12, color, "#FFFFFF", 2.5)


def replacement_rank_box(x, y):
    rect(x, y, 142, 78, GREEN_DARK, GREEN_FILL, width=1.5, rounded=True, arc=7)
    gpu_icon(x + 14, y + 17, scale=0.56, stroke=DAMAGED_GPU_COLOR, fill=DAMAGED_GPU_FILL)
    text("Replacement<br>Rank r′", x + 55, y + 16, 78, 42, size=14, color=INK, bold=True)


def state_box(x, y, w, label, sublabel, color, fill):
    rect(x, y, w, 60, color, fill, width=1.1, rounded=True, arc=5)
    text(label, x + 5, y + 8, w - 10, 20, size=13.5, color=color, bold=True)
    text(sublabel, x + 5, y + 29, w - 10, 22, size=11, color=INK)


def legend_arrow(x, y, color, label, dashed=False):
    line(x, y, x + 46, y, color, width=2.2, arrow="classic", dashed=dashed)
    text(label, x + 58, y - 11, 150, 24, size=13, color=INK, align="left")


def policy_formula(x, y):
    """Typeset the policy metric as separate vector text objects."""
    text("Φ′(t) =", x + 8, y + 29, 55, 20, size=15.5, color=INK, align="left")
    text("S(t)+", x + 66, y + 12, 50, 22, size=17, color=INK, align="left")
    line(x + 117, y + 16, x + 117, y + 33, INK, width=1.2, arrow="none")
    text("E", x + 122, y + 12, 14, 22, size=17, color=INK, align="left")
    text("new", x + 136, y + 25, 26, 11, size=9, color=INK, align="left")
    line(x + 157, y + 16, x + 157, y + 33, INK, width=1.2, arrow="none")
    text("·Δ", x + 162, y + 12, 23, 22, size=17, color=INK, align="left")
    line(x + 65, y + 43, x + 183, y + 43, INK, width=1.3, arrow="none")
    text("N", x + 88, y + 49, 16, 22, size=17, color=INK, align="left")
    text("expert", x + 104, y + 63, 40, 11, size=9, color=INK, align="left")
    text("·W", x + 148, y + 49, 30, 22, size=17, color=INK, align="left")


def subscript_item(x, y, main, sub, color=INK):
    text(main, x, y - 11, 26, 22, size=13.5, color=color, align="left")
    text(sub, x + 13, y + 1, 35, 10, size=8, color=color, align="left")


# Background and title
rect(0, 0, WIDTH, HEIGHT, "#FFFFFF", "#FFFFFF", width=0, rounded=False)
text("MoEGambit Runtime Architecture", 0, 17, WIDTH, 48, size=39, color="#000000", bold=True)

# Main panels
panel(18, 105, 285, 800, BLUE, "#FFFFFF")
panel(328, 105, 235, 800, BLUE, "#FFFFFF")
panel(588, 105, 210, 800, ORANGE, "#FFFFFF")
panel(822, 94, 402, 607, GREEN, "#FFFFFF")
panel(822, 720, 402, 185, RED, "#FFFFFF")
panel(1268, 107, 205, 800, BLUE, "#FFFFFF")

# Panel 1: training job
badge(36, 123, "1", BLUE, size=34, font_size=19)
text("Training Job", 80, 123, 170, 34, size=18, bold=True, align="left")
text("Sparse MoE Distributed<br>Training (3 DP × 4 Pipeline)", 72, 188, 180, 45, size=14, color=INK)
for idx, sx in enumerate(["S0", "S1", "S2", "S3"]):
    text(sx, 72 + idx * 62, 276, 44, 22, size=15, bold=True)
for ridx, dp in enumerate(["DP0", "DP1", "DP2"]):
    y = 312 + ridx * 137
    text(dp, 27, y + 22, 35, 22, size=15, color=BLUE_DARK, bold=True, align="left")
    for c in range(4):
        x = 68 + c * 62
        failed = ridx == 1 and c == 2
        tiny_worker(x, y, color=STAGE_GPU_COLORS[c], fill=STAGE_GPU_FILLS[c], failed=failed)
        if c < 3:
            line(x + 47, y + 28, x + 60, y + 28, INK, width=1.7, arrow="classic")
        if failed:
            failed_mark(x + 42, y + 45, r=11)
text("r (failed rank)", 169, 524, 103, 22, size=12, color=INK)
line(34, 700, 287, 700, GRAY, width=1, arrow="none", dashed=True)
tiny_worker(47, 729, color=DAMAGED_GPU_COLOR, fill="#FFFFFF", scale=0.68)
text("Healthy worker", 94, 727, 150, 28, size=13, color=INK, align="left")
tiny_worker(47, 797, color=DAMAGED_GPU_COLOR, fill="#FFFFFF", failed=True, scale=0.68)
failed_mark(89, 840, r=11)
text("Failed worker (rank r)", 94, 798, 160, 42, size=13, color=INK, align="left")

# Flow arrows between major panels
line(303, 466, 328, 466, INK, width=2.2, arrow="classic")
line(563, 466, 588, 466, INK, width=2.2, arrow="classic")
line(1224, 270, 1268, 270, INK, width=2.2, arrow="classic")
line(1224, 617, 1268, 617, INK, width=2.2, arrow="classic")
line(1224, 803, 1268, 803, INK, width=2.2, arrow="classic")

# Panel 2: failure detection and safe-point repair
badge(346, 123, "2", BLUE, size=34, font_size=19)
text("Failure Detection +<br>Safe-Point Repair (R1)", 388, 119, 155, 58, size=16, bold=True, align="left")
rect(382, 233, 136, 76, RED_DARK, RED_FILL, width=1.4, rounded=True, arc=8)
text("Failure Event<br><span style=\"font-size:14px\">event &lt;r, t, c&gt;</span>", 396, 247, 108, 42, size=16, color=RED_DARK, bold=False)
line(450, 309, 450, 381, INK, width=2.1, arrow="classic")
rect(346, 382, 200, 244, INK, "#FFFFFF", width=1.4, rounded=True, arc=7)
shield_icon(429, 393, BLUE, size=37)
text("Repair Controller<br>(Safe-Point Guard)", 372, 453, 150, 58, size=16, bold=True)
for yy, label in [
    (528, "mark r as RECOVERING"),
    (562, "block optimizer commit"),
    (596, "discard current iteration"),
]:
    bullet(364, yy, BLUE)
    text(label, 381, yy - 12, 146, 24, size=13, align="left")
rect(344, 705, 205, 125, BLUE, "#FFFFFF", width=1.1, rounded=True, arc=7, dashed=True)
for yy, label in [
    (735, "<b>r</b> : failed rank"),
    (770, "<b>t</b> : failure step (current)"),
    (805, "<b>c</b> : last checkpoint step"),
]:
    text(label, 377, yy - 11, 145, 22, size=13, align="left")

# Panel 3: guarded policy
badge(606, 123, "3", ORANGE_DARK, size=34, font_size=19)
text("Guarded Policy (R2)", 649, 126, 140, 30, size=16, color=INK, bold=True, align="left")
text("Policy Metric", 602, 192, 140, 26, size=14, color=ORANGE_DARK, bold=True, align="left")
rect(598, 219, 190, 86, ORANGE, ORANGE_FILL, width=1.1, rounded=True, arc=6)
policy_formula(598, 219)
text("Inputs", 602, 330, 112, 24, size=14, color=ORANGE_DARK, bold=True, align="left")
for i, label in enumerate(["Δ = t − c", "peerAvail", "S(t)"]):
    yy = 372 + i * 26
    bullet(604, yy, ORANGE_DARK)
    text(label, 618, yy - 11, 95, 22, size=13.5, align="left")
yy = 450
bullet(604, yy, ORANGE_DARK)
text("E", 618, yy - 11, 15, 22, size=13.5, align="left")
text("new", 631, yy + 1, 26, 10, size=8, align="left")
text("Thresholds", 602, 487, 112, 24, size=14, color=ORANGE_DARK, bold=True, align="left")
for i, (main, sub) in enumerate([("Δ", "min"), ("Δ", "max"), ("Φ", "max")]):
    yy = 524 + i * 28
    bullet(604, yy, ORANGE_DARK)
    subscript_item(618, yy, main, sub)
add_cell(
    "",
    598,
    615,
    145,
    145,
    f"html=1;rhombus;whiteSpace=wrap;fillColor={ORANGE_FILL};strokeColor={GRAY};strokeWidth=1.5;",
)
shield_icon(655, 625, BLUE, size=31)
text("guards pass?<br>HYBRID ?", 623, 674, 95, 44, size=14, color=INK, bold=True)
text("YES", 748, 656, 45, 22, size=14, color=GREEN, bold=True, align="left")
line(743, 682, 817, 682, GREEN, width=2.2, arrow="classic")
text("NO", 682, 789, 42, 20, size=14, color=RED_DARK, bold=True, align="left")
line(671, 760, 671, 821, RED, width=2.2, arrow="none")
line(671, 821, 817, 821, RED, width=2.2, arrow="classic")

# Panel 4a: hybrid recovery
badge(839, 107, "4a", GREEN, size=34, font_size=16)
text("Hybrid Recovery <span style=\"font-size:14px\">(fast path)</span>", 893, 106, 260, 34, size=18, color=GREEN_DARK, bold=True, align="left")
rect(834, 146, 180, 114, BLUE, BLUE_FILL, width=1.1, rounded=True, arc=6)
text("Healthy Peer(s) at step t", 849, 156, 150, 22, size=13, color=BLUE_DARK, bold=True)
for ix in range(3):
    tiny_worker(852 + ix * 50, 186, scale=0.74)
text("…", 994, 206, 18, 20, size=19, color=INK, bold=True)
rect(1036, 146, 176, 114, ORANGE, ORANGE_FILL, width=1.1, rounded=True, arc=6)
text("Checkpoint Shard (step c)", 1044, 156, 160, 22, size=13, color=ORANGE_DARK, bold=True)
cylinder(1099, 184, 58, 66, INK, "#FFFFFF", width=1.8)
checkpoint_blocks(1112, 221, ORANGE)
replacement_rank_box(947, 337)
line(920, 260, 978, 334, BLUE, width=2.2, arrow="classic", points=[(940, 284)])
line(1088, 260, 1018, 334, ORANGE, width=2.2, arrow="classic", points=[(1072, 286)])
line(1018, 415, 1018, 444, GREEN, width=2, arrow="classic")
text("Path P<br>(peer ➜ r′)", 838, 279, 95, 40, size=13, color=BLUE_DARK, bold=True)
for i, label in enumerate(["dense/shared", "router", "replicated opt state"]):
    bullet(831, 340 + i * 21, BLUE, r=2.5)
    text(label, 845, 330 + i * 21, 98, 21, size=12, color=BLUE_DARK, align="left")
text("Path C<br>(checkpoint ➜ r′)", 1095, 279, 105, 40, size=13, color=ORANGE_DARK, bold=True)
for i, label in enumerate(["local experts", "expert opt state"]):
    bullet(1102, 348 + i * 22, ORANGE, r=2.5)
    text(label, 1116, 338 + i * 22, 90, 21, size=12, color=ORANGE_DARK, align="left")

rect(830, 458, 382, 84, GREEN, "#FFFFFF", width=1.0, rounded=True, arc=6, dashed=True, dash="3 3")
rect(932, 445, 182, 24, "#FFFFFF", "#FFFFFF", width=0, rounded=False)
text("Two-Phase Recovery Protocol", 934, 446, 176, 24, size=13, color=GREEN_DARK, bold=True)
rect(842, 473, 99, 56, BLUE, BLUE_FILL, width=1.0, rounded=True, arc=5)
text("Phase A:<br>weights first", 853, 484, 77, 32, size=12, color=BLUE_DARK, bold=True)
line(941, 501, 966, 501, INK, width=1.8, arrow="classic")
rect(966, 473, 105, 56, GREEN_DARK, GREEN_FILL, width=1.0, rounded=True, arc=5)
text("resume training<br>under barrier", 973, 485, 91, 31, size=12.5, color=INK)
line(1071, 501, 1095, 501, INK, width=1.8, arrow="classic")
rect(1095, 473, 102, 56, ORANGE, ORANGE_FILL, width=1.0, rounded=True, arc=5)
text("Phase B:<br>optimizer later", 1105, 484, 82, 32, size=12, color=ORANGE_DARK, bold=True)

line(1018, 542, 1018, 571, GREEN, width=2, arrow="classic")
rect(830, 582, 382, 98, GREEN, "#FFFFFF", width=1.0, rounded=True, arc=6, dashed=True, dash="3 3")
rect(902, 571, 240, 24, "#FFFFFF", "#FFFFFF", width=0, rounded=False)
text("Reintegration (Guarded State Machine)", 905, 572, 234, 24, size=13, color=GREEN_DARK, bold=True)
state_box(838, 602, 85, "RECOVERING", "(recovering r′)", BLUE_DARK, BLUE_FILL)
line(923, 632, 940, 632, INK, width=1.7, arrow="classic")
state_box(940, 602, 78, "REPAIRED", "(state synced)", GREEN_DARK, GREEN_FILL)
line(1018, 632, 1036, 632, INK, width=1.7, arrow="classic")
state_box(1036, 602, 74, "BARRIER", "(global sync)", ORANGE_DARK, ORANGE_FILL)
line(1110, 632, 1127, 632, INK, width=1.7, arrow="classic")
state_box(1127, 602, 70, "HEALTHY", "(fully active)", GREEN_DARK, GREEN_FILL)
check_badge(1197, 645, r=12)

# Panel 4b: checkpoint restart
badge(835, 735, "4b", RED, size=34, font_size=16)
text("Checkpoint Restart <span style=\"font-size:14px\">(fallback)</span>", 886, 735, 250, 30, size=17, color=RED_DARK, bold=True, align="left")
cylinder(845, 778, 50, 62, INK, "#FFFFFF", width=1.6)
checkpoint_blocks(855, 813, ORANGE)
text("Full Checkpoint<br>(step c)", 829, 845, 83, 40, size=12, color=INK)
line(896, 807, 935, 807, RED, width=2, arrow="classic")
for ix in range(2):
    tiny_worker(949 + ix * 38, 776, scale=0.64)
text("Reload<br>All Ranks", 938, 845, 85, 40, size=12, color=INK)
line(1017, 807, 1050, 807, RED, width=2, arrow="classic")
ellipse(1061, 783, 55, 55, "#555555", "#FFFFFF", 1.6)
text("↻", 1062, 786, 53, 48, size=35, color="#555555", bold=True)
text("Replay<br>t − c", 1050, 845, 78, 40, size=12, color=INK)
line(1118, 807, 1150, 807, RED, width=2, arrow="classic")
ellipse(1150, 775, 60, 60, INK, "#FFFFFF", 1.5)
add_cell(
    "",
    1173,
    792,
    18,
    28,
    f"html=1;shape=triangle;direction=east;fillColor={INK};strokeColor={INK};strokeWidth=1;",
)
text("Resume<br>Training", 1139, 845, 84, 40, size=12, color=INK)

# Panel 5: observability
badge(1286, 124, "5", BLUE, size=34, font_size=19)
text("Observability +<br>Logs (R3)", 1328, 121, 115, 55, size=16, bold=True, align="left")
document_icon(1322, 204, 86, 120, BLUE)
text("Every recovery<br>decision is auditable.", 1318, 356, 110, 48, size=13, color=INK)
line(1281, 429, 1456, 429, GRAY, width=1, arrow="none", dashed=True)
for y, label, icon in [
    (457, "policy inputs + reason", "mag"),
    (548, "TTTR / TTTFR /<br>path latency", "gauge"),
    (640, "state-machine trace", "trace"),
    (732, "quality metrics<br>(loss, ppl,<br>load balance)", "bars"),
]:
    rect(1282, y, 174, 64 if y == 457 else 74, BLUE, BLUE_FILL, width=1.0, rounded=True, arc=6)
    if icon == "mag":
        magnifier_icon(1292, y + 16, BLUE)
    elif icon == "gauge":
        gauge_icon(1304, y + 33, BLUE)
    elif icon == "trace":
        trace_icon(1292, y + 13, BLUE)
    else:
        bars_icon(1293, y + 13, BLUE)
    text(label, 1340, y + 12, 102, 42 if y == 457 else 50, size=13, color=INK, align="left")

# Bottom legend
legend_y = 958
legend_arrow(75, legend_y, "#000000", "Control Flow")
legend_arrow(262, legend_y, BLUE, "Path P (Peer State)")
legend_arrow(475, legend_y, ORANGE, "Path C (Checkpoint State)")
legend_arrow(736, legend_y, GREEN, "Hybrid Path (Fast)")
legend_arrow(957, legend_y, RED, "Restart Path (Fallback)")
rect(1208, legend_y - 15, 48, 28, "#777777", "#FFFFFF", width=1.1, rounded=True, arc=6, dashed=True)
text("R1/R2/R3: MoEGambit Rules", 1264, legend_y - 11, 220, 24, size=13, color=INK, align="left")


header = textwrap.dedent(
    f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <mxfile host="app.diagrams.net" agent="moegambit-generator" version="30.0.4">
      <diagram id="moegambit-runtime" name="MoEGambit Runtime Architecture">
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
    path = os.path.join(out_dir, f"moegambit_runtime_architecture{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"Wrote {path}")
print(f"Total cells: {len(cells)}")
