from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("/Users/zds/bsr/tmp/patent_figures")
DOCX_TARGETS = [
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请.docx"),
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx"),
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = FONT_CANDIDATES
    if bold:
        names = ["/System/Library/Fonts/STHeiti Medium.ttc"] + FONT_CANDIDATES
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(44, True)
F_HEAD = font(31, True)
F_BODY = font(29)
F_SMALL = font(25)
F_TINY = font(22)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        line = ""
        for ch in raw:
            candidate = line + ch
            if text_size(draw, candidate, fnt)[0] <= max_w or not line:
                line = candidate
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
    return lines or [""]


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont = F_BODY,
    line_gap: int = 8,
):
    x1, y1, x2, y2 = box
    lines = wrap_text(draw, text, fnt, x2 - x1 - 28)
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, fnt)
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=fnt, fill="black")
        y += h + line_gap


def rect(draw, box, label, fnt=F_BODY, width=4):
    draw.rectangle(box, outline="black", width=width)
    centered_text(draw, box, label, fnt)


def diamond(draw, center, size, label, fnt=F_SMALL, width=4):
    cx, cy = center
    w, h = size
    points = [(cx, cy - h // 2), (cx + w // 2, cy), (cx, cy + h // 2), (cx - w // 2, cy)]
    draw.polygon(points, outline="black", fill="white")
    draw.line(points + [points[0]], fill="black", width=width)
    centered_text(draw, (cx - w // 2 + 15, cy - h // 2 + 10, cx + w // 2 - 15, cy + h // 2 - 10), label, fnt, 5)


def arrow(draw, p1, p2, width=4, dashed=False, label: str | None = None, label_offset=(0, -28)):
    x1, y1 = p1
    x2, y2 = p2
    if dashed:
        steps = max(1, int(math.hypot(x2 - x1, y2 - y1) // 22))
        for i in range(steps):
            if i % 2 == 0:
                xa = x1 + (x2 - x1) * i / steps
                ya = y1 + (y2 - y1) * i / steps
                xb = x1 + (x2 - x1) * (i + 1) / steps
                yb = y1 + (y2 - y1) * (i + 1) / steps
                draw.line((xa, ya, xb, yb), fill="black", width=width)
    else:
        draw.line((x1, y1, x2, y2), fill="black", width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 20
    left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
    right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), left, right], fill="black")
    if label:
        fnt = F_TINY
        w, h = text_size(draw, label, fnt)
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        draw.rectangle((mx - w / 2 - 6, my - 4, mx + w / 2 + 6, my + h + 4), fill="white")
        draw.text((mx - w / 2, my), label, font=fnt, fill="black")


def poly_arrow(draw, points, width=4, dashed=False, label=None):
    for a, b in zip(points, points[1:-1]):
        arrow(draw, a, b, width=width, dashed=dashed)
    arrow(draw, points[-2], points[-1], width=width, dashed=dashed, label=label)


def new_canvas(w: int, h: int):
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    return img, draw


def draw_figure1():
    img, d = new_canvas(2400, 1200)
    boxes = {
        "event": (120, 90, 430, 210),
        "ctrl": (560, 70, 930, 230),
        "judge": (1060, 70, 1510, 230),
        "diamond": (1690, 150),
        "restart": (1840, 260, 2220, 385),
        "class": (120, 540, 440, 720),
        "peer": (610, 470, 950, 610),
        "expert": (610, 710, 950, 850),
        "hybrid": (1080, 575, 1510, 735),
        "two": (1690, 520, 2070, 660),
        "log": (1690, 820, 2070, 980),
    }

    rect(d, boxes["event"], "失效rank事件\n⟨r,t,c⟩", F_SMALL)
    rect(d, boxes["ctrl"], "恢复控制器（101）\n安全点控制、提交保护", F_SMALL)
    rect(d, boxes["judge"], "恢复策略判定（102）\nPeerAvail、Δ、Φ′(t)", F_SMALL)
    diamond(d, boxes["diamond"], (170, 170), "混合\n或\n重启", F_TINY)
    rect(d, boxes["restart"], "检查点重启路径（109）\n全局一致恢复", F_SMALL)
    rect(d, boxes["class"], "状态分类模块（103）\n非专家复制态\n专家分片态\n运行时元数据", F_SMALL)
    rect(d, boxes["peer"], "状态来源A（104）\n健康dense-DP peer\n当前非专家状态", F_SMALL)
    rect(d, boxes["expert"], "状态来源B（105）\n检查点分片\n或专家peer", F_SMALL)
    rect(d, boxes["hybrid"], "混合状态恢复模块（106）\nPath P + Path C\n重构替换rank状态", F_SMALL)
    rect(d, boxes["two"], "两阶段恢复模块（107）\n权重优先\n优化器稍后", F_SMALL)
    rect(d, boxes["log"], "重集成与日志模块（108）\nRECOVERING→HEALTHY\n记录决策、阈值、时延", F_SMALL)

    arrow(d, (430, 150), (560, 150))
    arrow(d, (930, 150), (1060, 150))
    arrow(d, (1510, 150), (1605, 150))
    arrow(d, (1775, 150), (1840, 320), label="Restart", label_offset=(35, -5))
    arrow(d, (1688, 235), (1295, 575), dashed=True, label="Hybrid", label_offset=(10, -20))
    arrow(d, (275, 210), (275, 540), dashed=True, label="状态建模", label_offset=(-60, -10))
    arrow(d, (440, 630), (610, 540))
    arrow(d, (440, 630), (610, 780))
    arrow(d, (950, 540), (1080, 625), label="Path P")
    arrow(d, (950, 780), (1080, 690), label="Path C")
    arrow(d, (1510, 655), (1690, 590))
    arrow(d, (1880, 660), (1880, 820))
    img.save(OUT_DIR / "patent_fig1_system.png", dpi=(300, 300))


def draw_figure2():
    img, d = new_canvas(1900, 1900)
    left_x, box_w, box_h = 140, 420, 140
    y0, gap = 100, 210
    left_boxes = []
    labels = [
        "S1 接收故障事件\n⟨r,t,c⟩",
        "S2 建立安全点\n丢弃飞行中迭代",
        "S3 划分训练状态\n非专家/专家/元数据",
        "S4 计算恢复风险\nΔ、S(t)、Φ′(t)",
    ]
    for i, label in enumerate(labels):
        box = (left_x, y0 + i * gap, left_x + box_w, y0 + i * gap + box_h)
        rect(d, box, label, F_SMALL)
        left_boxes.append(box)
        if i:
            arrow(d, ((box[0] + box[2]) // 2, y0 + i * gap - gap + box_h), ((box[0] + box[2]) // 2, box[1]))

    dec_center = (900, 850)
    diamond(d, dec_center, (250, 250), "是否满足\nPeerAvail\n及阈值", F_TINY)
    arrow(d, (left_boxes[-1][2], (left_boxes[-1][1] + left_boxes[-1][3]) // 2), (775, 850))

    restart = (650, 1230, 1060, 1370)
    rect(d, restart, "S5c 检查点重启\n任一保护条件不满足", F_SMALL)
    arrow(d, (900, 975), (860, 1230), label="否", label_offset=(-35, 0))

    right_x = 1260
    right_boxes = [
        (right_x, 510, right_x + 470, 650, "S5a Path P\n从健康dense-DP peer\n拉取非专家状态"),
        (right_x, 810, right_x + 470, 950, "S5b Path C\n恢复专家状态\n检查点分片/专家peer"),
        (right_x, 1110, right_x + 470, 1250, "S6 两阶段恢复\n权重优先、优化器稍后"),
        (right_x, 1410, right_x + 470, 1550, "S7 重建运行时元数据\n通信组、专家目录、rank映射"),
        (right_x, 1660, right_x + 470, 1800, "S8 替换rank进入HEALTHY\n恢复训练"),
    ]
    for i, (x1, y1, x2, y2, label) in enumerate(right_boxes):
        rect(d, (x1, y1, x2, y2), label, F_SMALL)
        if i:
            arrow(d, ((x1 + x2) // 2, right_boxes[i - 1][3]), ((x1 + x2) // 2, y1))
    arrow(d, (1025, 850), (1260, 580), label="是", label_offset=(-10, -18))
    # Dotted conservative fallback edge.
    arrow(d, (1060, 1300), (1260, 1480), dashed=True, label="重集成", label_offset=(20, -15))
    img.save(OUT_DIR / "patent_fig2_flow.png", dpi=(300, 300))


def draw_lifeline(d, x, y1, y2, title):
    rect(d, (x - 140, y1, x + 140, y1 + 90), title, F_SMALL, width=4)
    d.line((x, y1 + 90, x, y2), fill="black", width=3)


def draw_figure3():
    img, d = new_canvas(2400, 1350)
    xs = [300, 820, 1340, 1860]
    titles = ["恢复控制器\n（101）", "替换rank", "状态来源\n（104/105）", "更新屏障\n（107）"]
    for x, title in zip(xs, titles):
        draw_lifeline(d, x, 80, 1200, title)

    events = [
        (220, xs[0], xs[1], "分配替换rank"),
        (330, xs[0], xs[3], "安装提交保护"),
        (460, xs[0], xs[2], "请求当前非专家状态"),
        (560, xs[2], xs[1], "返回非专家状态"),
        (680, xs[0], xs[2], "读取专家权重"),
        (780, xs[2], xs[1], "返回专家权重"),
        (900, xs[1], xs[3], "等待优化器状态"),
        (1020, xs[2], xs[1], "后台返回专家优化器状态"),
        (1130, xs[0], xs[3], "释放更新屏障并记录日志"),
    ]
    for y, x1, x2, label in events:
        arrow(d, (x1, y), (x2, y), width=4, label=label, label_offset=(0, -36))

    d.rectangle((190, 1220, 1970, 1305), outline="black", width=4)
    centered_text(d, (190, 1220, 1970, 1305), "结果：替换rank经 RECOVERING → REPAIRED → BARRIER → HEALTHY 后恢复训练", F_SMALL)
    img.save(OUT_DIR / "patent_fig3_sequence.png", dpi=(300, 300))


def replace_media(docx_path: Path):
    backup = docx_path.with_name(docx_path.stem + f"_附图替换前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(docx_path, backup)
    tmp = docx_path.with_suffix(".tmp.docx")
    replacements = {
        "word/media/image1.png": OUT_DIR / "patent_fig1_system.png",
        "word/media/image2.png": OUT_DIR / "patent_fig2_flow.png",
        "word/media/image3.png": OUT_DIR / "patent_fig3_sequence.png",
    }
    with ZipFile(docx_path, "r") as zin, ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename in replacements:
                data = replacements[info.filename].read_bytes()
            zout.writestr(info, data)
    tmp.replace(docx_path)
    return backup


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_figure1()
    draw_figure2()
    draw_figure3()
    for target in DOCX_TARGETS:
        print(replace_media(target))
    for png in sorted(OUT_DIR.glob("patent_fig*.png")):
        print(png, Image.open(png).size)
