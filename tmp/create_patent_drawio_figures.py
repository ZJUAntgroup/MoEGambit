#!/usr/bin/env python3
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path


OUT_DIR = Path("/Users/zds/Desktop/MoEGambit_patent_drawio")
COMBINED = OUT_DIR / "MoEGambit_专利附图_可编辑.drawio"
SINGLE_NAMES = [
    "图1_运行时混合恢复系统架构.drawio",
    "图2_运行时混合恢复方法流程.drawio",
    "图3_两阶段恢复时序.drawio",
]


def esc(value: str) -> str:
    return html.escape(value.replace("\n", "<br>"), quote=True)


def style_rect(font_size=15, bold=False) -> str:
    return (
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        "strokeColor=#000000;strokeWidth=2;fontFamily=Arial;"
        f"fontSize={font_size};fontColor=#000000;align=center;"
        "verticalAlign=middle;spacing=8;"
        f"fontStyle={1 if bold else 0};"
    )


def style_text(font_size=13) -> str:
    return (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Arial;"
        f"fontSize={font_size};fontColor=#000000;"
    )


def style_diamond(font_size=14) -> str:
    return (
        "rhombus;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        "strokeColor=#000000;strokeWidth=2;fontFamily=Arial;"
        f"fontSize={font_size};fontColor=#000000;align=center;"
        "verticalAlign=middle;spacing=8;"
    )


def style_edge(dashed=False) -> str:
    dashed_part = "dashed=1;dashPattern=8 6;" if dashed else ""
    return (
        "endArrow=block;endFill=1;html=1;rounded=0;"
        "strokeColor=#000000;strokeWidth=2;fontFamily=Arial;fontSize=12;"
        f"{dashed_part}"
    )


class Diagram:
    def __init__(self, name: str, page_w: int, page_h: int):
        self.name = name
        self.page_w = page_w
        self.page_h = page_h
        self.cells: list[str] = []
        self.n = 0

    def cid(self, prefix="c") -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def rect(self, id_: str, value: str, x, y, w, h, *, bold=False, font_size=15):
        self.cells.append(
            f'<mxCell id="{id_}" value="{esc(value)}" style="{style_rect(font_size, bold)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return id_

    def diamond(self, id_: str, value: str, x, y, w, h, *, font_size=14):
        self.cells.append(
            f'<mxCell id="{id_}" value="{esc(value)}" style="{style_diamond(font_size)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return id_

    def label(self, value: str, x, y, w, h, *, font_size=13):
        id_ = self.cid("t")
        self.cells.append(
            f'<mxCell id="{id_}" value="{esc(value)}" style="{style_text(font_size)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return id_

    def edge(self, src, dst, *, label="", dashed=False, points=None):
        id_ = self.cid("e")
        pts = ""
        if points:
            pts = '<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points) + "</Array>"
        self.cells.append(
            f'<mxCell id="{id_}" value="{esc(label)}" style="{style_edge(dashed)}" edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>'
        )
        return id_

    def coord_edge(self, x1, y1, x2, y2, *, label="", dashed=False, points=None):
        id_ = self.cid("e")
        pts = ""
        if points:
            pts = '<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points) + "</Array>"
        self.cells.append(
            f'<mxCell id="{id_}" value="{esc(label)}" style="{style_edge(dashed)}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>{pts}<mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
            f'</mxGeometry></mxCell>'
        )
        return id_

    def mxgraph(self) -> str:
        root = ['<root><mxCell id="0"/><mxCell id="1" parent="0"/>']
        root.extend(self.cells)
        root.append("</root>")
        return (
            f'<mxGraphModel dx="{self.page_w}" dy="{self.page_h}" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{self.page_w}" pageHeight="{self.page_h}" math="0" shadow="0">'
            + "".join(root)
            + "</mxGraphModel>"
        )

    def diagram_xml(self, diagram_id: str) -> str:
        return f'<diagram id="{diagram_id}" name="{esc(self.name)}">{self.mxgraph()}</diagram>'


def fig1() -> Diagram:
    d = Diagram("图1 运行时混合恢复系统架构", 1500, 850)
    d.rect("f1_event", "失效rank事件\n⟨r,t,c⟩", 50, 70, 180, 75)
    d.rect("f1_ctrl", "恢复控制器（101）\n安全点控制、提交保护", 310, 60, 260, 95, bold=True)
    d.rect("f1_policy", "恢复策略判定（102）\nPeerAvail、Δ、Φ′(t)", 660, 60, 310, 95, bold=True)
    d.diamond("f1_decision", "混合\n或\n重启", 1070, 62, 120, 110)
    d.rect("f1_restart", "检查点重启路径（109）\n全局一致恢复", 1240, 210, 210, 85)

    d.rect("f1_class", "状态分类模块（103）\n非专家复制态\n专家分片态\n运行时元数据", 60, 455, 230, 140)
    d.rect("f1_peer", "状态来源A（104）\n健康dense-DP peer\n当前非专家状态", 420, 360, 250, 100)
    d.rect("f1_expert", "状态来源B（105）\n检查点分片\n或专家peer", 420, 545, 250, 100)
    d.rect("f1_hybrid", "混合状态恢复模块（106）\nPath P + Path C\n重构替换rank状态", 770, 445, 270, 115, bold=True)
    d.rect("f1_two", "两阶段恢复模块（107）\n权重优先\n优化器稍后", 1150, 400, 240, 100)
    d.rect("f1_log", "重集成与日志模块（108）\nRECOVERING→HEALTHY\n记录决策、阈值、时延", 1150, 610, 260, 115)

    d.edge("f1_event", "f1_ctrl")
    d.edge("f1_ctrl", "f1_policy")
    d.edge("f1_policy", "f1_decision")
    d.edge("f1_decision", "f1_restart", label="Restart")
    d.coord_edge(140, 145, 175, 455, label="状态建模", dashed=True, points=[(140, 260), (175, 260)])
    d.edge("f1_class", "f1_peer")
    d.edge("f1_class", "f1_expert")
    d.edge("f1_peer", "f1_hybrid", label="Path P")
    d.edge("f1_expert", "f1_hybrid", label="Path C")
    d.edge("f1_hybrid", "f1_two")
    d.edge("f1_two", "f1_log")
    d.coord_edge(1130, 172, 905, 445, label="Hybrid", dashed=True, points=[(1130, 260), (905, 260)])
    return d


def fig2() -> Diagram:
    d = Diagram("图2 运行时混合恢复方法流程", 1200, 1300)
    x_left, w, h = 90, 290, 95
    ys = [80, 235, 390, 545]
    labels = [
        "S1 接收故障事件\n⟨r,t,c⟩",
        "S2 建立安全点\n丢弃在途迭代",
        "S3 划分训练状态\n非专家/专家/元数据",
        "S4 计算恢复风险\nΔ、S(t)、Φ′(t)",
    ]
    ids = []
    for i, (y, label) in enumerate(zip(ys, labels), 1):
        ids.append(d.rect(f"f2_s{i}", label, x_left, y, w, h))
        if i > 1:
            d.edge(ids[i - 2], ids[i - 1])
    d.diamond("f2_decide", "是否满足\nPeerAvail\n及阈值", 560, 560, 145, 145)
    d.edge("f2_s4", "f2_decide")

    d.rect("f2_restart", "S5c 检查点重启\n任一保护条件不满足", 460, 900, 310, 95)
    d.edge("f2_decide", "f2_restart", label="否")
    d.rect("f2_p", "S5a Path P\n从健康dense-DP peer\n拉取非专家状态", 870, 420, 300, 105)
    d.rect("f2_c", "S5b Path C\n恢复专家状态\n检查点分片/专家peer", 870, 610, 300, 105)
    d.rect("f2_phase", "S6 两阶段恢复\n权重优先、优化器稍后", 870, 800, 300, 105)
    d.rect("f2_meta", "S7 重建运行时元数据\n通信组、专家目录、rank映射", 870, 990, 300, 105)
    d.rect("f2_done", "S8 替换rank进入HEALTHY\n恢复训练", 870, 1170, 300, 95)
    d.edge("f2_decide", "f2_p", label="是")
    d.edge("f2_p", "f2_c")
    d.edge("f2_c", "f2_phase")
    d.edge("f2_phase", "f2_meta")
    d.edge("f2_meta", "f2_done")
    d.edge("f2_restart", "f2_meta", label="重集成", dashed=True)
    return d


def fig3() -> Diagram:
    d = Diagram("图3 混合恢复与两阶段协议时序", 1500, 1000)
    xs = [170, 520, 880, 1230]
    titles = ["恢复控制器\n（101）", "替换rank", "状态来源\n（104/105）", "更新屏障\n（107）"]
    for i, (x, title) in enumerate(zip(xs, titles), 1):
        d.rect(f"f3_head{i}", title, x - 85, 55, 170, 60)
        d.coord_edge(x, 115, x, 860, dashed=False)
    arrows = [
        (160, 170, 520, "分配替换rank"),
        (250, 170, 1230, "安装提交保护"),
        (345, 170, 880, "请求当前非专家状态"),
        (435, 880, 520, "返回非专家状态"),
        (530, 170, 880, "读取专家权重"),
        (620, 880, 520, "返回专家权重"),
        (710, 520, 1230, "等待优化器状态"),
        (790, 880, 520, "后台返回专家优化器状态"),
        (855, 170, 1230, "释放更新屏障并记录日志"),
    ]
    for y, x1, x2, label in arrows:
        d.coord_edge(x1, y, x2, y, label=label)
    d.rect("f3_result", "结果：替换rank经 RECOVERING → REPAIRED → BARRIER → HEALTHY 后恢复训练", 120, 920, 1260, 55)
    return d


def mxfile(diagrams: list[Diagram]) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    body = "\n".join(diagram.diagram_xml(f"moegambit_patent_{i}") for i, diagram in enumerate(diagrams, 1))
    return (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f'<mxfile host="app.diagrams.net" modified="{now}" agent="Codex" version="30.2.6" type="device">\n'
        f"{body}\n"
        "</mxfile>\n"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagrams = [fig1(), fig2(), fig3()]
    COMBINED.write_text(mxfile(diagrams), encoding="utf-8")
    for diagram, name in zip(diagrams, SINGLE_NAMES):
        (OUT_DIR / name).write_text(mxfile([diagram]), encoding="utf-8")
    print(COMBINED)
    for name in SINGLE_NAMES:
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
