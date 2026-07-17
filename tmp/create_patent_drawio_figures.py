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


def style_rect(font_size=18, bold=False) -> str:
    return (
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        "strokeColor=#000000;strokeWidth=1.5;fontFamily=PingFang SC;"
        f"fontSize={font_size};fontColor=#000000;align=center;"
        "verticalAlign=middle;spacing=6;"
        f"fontStyle={1 if bold else 0};"
    )


def style_text(font_size=16) -> str:
    return (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=PingFang SC;"
        f"fontSize={font_size};fontColor=#000000;"
    )


def style_diamond(font_size=17) -> str:
    return (
        "rhombus;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        "strokeColor=#000000;strokeWidth=1.5;fontFamily=PingFang SC;"
        f"fontSize={font_size};fontColor=#000000;align=center;"
        "verticalAlign=middle;spacing=5;"
    )


def style_edge(dashed=False) -> str:
    dashed_part = "dashed=1;dashPattern=6 5;" if dashed else ""
    return (
        "endArrow=block;endFill=1;html=1;rounded=0;"
        "strokeColor=#000000;strokeWidth=1.5;fontFamily=PingFang SC;fontSize=16;"
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

    def rect(self, id_: str, value: str, x, y, w, h, *, bold=False, font_size=18):
        self.cells.append(
            f'<mxCell id="{id_}" value="{esc(value)}" style="{style_rect(font_size, bold)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return id_

    def diamond(self, id_: str, value: str, x, y, w, h, *, font_size=17):
        self.cells.append(
            f'<mxCell id="{id_}" value="{esc(value)}" style="{style_diamond(font_size)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return id_

    def label(self, value: str, x, y, w, h, *, font_size=16):
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
    d = Diagram("图1 运行时混合恢复系统架构", 1600, 800)
    d.rect("f1_event", "失效进程事件\n⟨r,t,c⟩", 35, 65, 190, 80)
    d.rect("f1_ctrl", "恢复控制器（101）\n安全点与替换进程管理", 275, 52, 280, 105, bold=True)
    d.rect("f1_policy", "恢复策略判定模块（102）\nPeerAvail、Δ、Φ′(t)", 615, 52, 315, 105, bold=True)
    d.diamond("f1_decision", "允许\n混合恢复？", 1000, 48, 145, 120)
    d.rect("f1_restart", "检查点重启路径（109）\n恢复全局一致状态", 1260, 65, 270, 85)

    d.rect("f1_class", "状态分类模块（103）\n非专家状态\n专家状态\n运行时元数据", 55, 420, 245, 145)
    d.rect("f1_peer", "非专家状态来源（104）\n健康非专家层数据并行对等进程\n当前安全点状态", 375, 305, 335, 125)
    d.rect("f1_expert", "专家状态来源（105）\n健康专家对等进程优先\n不可用时读取检查点分片", 375, 585, 335, 130)
    d.rect("f1_hybrid", "混合状态恢复模块（106）\n路径P：非专家状态\n路径C：专家状态", 790, 430, 305, 130, bold=True)
    d.rect("f1_two", "两阶段恢复模块（107）\n权重优先恢复\n优化器状态后台恢复", 1195, 350, 300, 120)
    d.rect("f1_log", "重集成与结构化日志模块（108）\n更新屏障、状态迁移\n记录决策、阈值与时延", 1195, 590, 300, 125)

    d.edge("f1_event", "f1_ctrl")
    d.edge("f1_ctrl", "f1_policy")
    d.edge("f1_policy", "f1_decision")
    d.edge("f1_decision", "f1_restart", label="否")
    d.coord_edge(130, 145, 175, 420, label="状态建模", dashed=True, points=[(130, 245), (175, 245)])
    d.edge("f1_class", "f1_peer")
    d.edge("f1_class", "f1_expert")
    d.edge("f1_peer", "f1_hybrid", label="路径P")
    d.edge("f1_expert", "f1_hybrid", label="路径C")
    d.edge("f1_hybrid", "f1_two")
    d.edge("f1_two", "f1_log")
    d.coord_edge(1072, 168, 942, 430, label="是", dashed=True, points=[(1072, 245), (942, 245)])
    return d


def fig2() -> Diagram:
    d = Diagram("图2 运行时混合恢复方法流程", 1200, 1200)
    x_left, w, h = 455, 290, 82
    ys = [35, 150, 265, 380]
    labels = [
        "S1 接收故障事件\n⟨r,t,c⟩",
        "S2 分配替换进程并建立安全点\n丢弃在途迭代",
        "S3 分类状态并确定恢复来源\n形成 E_ckpt(t)",
        "S4 计算专家陈旧暴露\nΔ、S(t)、Φ′(t)",
    ]
    ids = []
    for i, (y, label) in enumerate(zip(ys, labels), 1):
        ids.append(d.rect(f"f2_s{i}", label, x_left, y, w, h))
        if i > 1:
            d.edge(ids[i - 2], ids[i - 1])
    d.diamond("f2_decide", "PeerAvail为真\n且满足阈值？", 520, 505, 160, 135)
    d.edge("f2_s4", "f2_decide")

    d.rect("f2_restart", "S5c 检查点重启\n任一运行时条件不满足", 845, 520, 290, 95)
    d.edge("f2_decide", "f2_restart", label="否")
    d.rect("f2_p", "S5a 非专家状态路径P\n从健康非专家层数据并行\n对等进程同步当前状态", 105, 700, 385, 115)
    d.rect("f2_c", "S5b 专家状态路径C\n健康专家对等进程优先\n不可用时读取检查点分片", 710, 700, 385, 115)
    d.rect("f2_phase", "S6 两阶段恢复\n权重优先，优化器状态稍后", 455, 875, 290, 90)
    d.rect("f2_meta", "S7 重建运行时元数据并处理暂存梯度\n通信组、专家目录、进程映射", 390, 1000, 420, 95)
    d.rect("f2_done", "S8 释放更新屏障并恢复训练\n记录结构化日志", 455, 1130, 290, 65)
    d.edge("f2_decide", "f2_p", label="是")
    d.edge("f2_decide", "f2_c")
    d.edge("f2_p", "f2_phase")
    d.edge("f2_c", "f2_phase")
    d.edge("f2_phase", "f2_meta")
    d.edge("f2_meta", "f2_done")
    return d


def fig3() -> Diagram:
    d = Diagram("图3 混合恢复与两阶段协议时序", 1350, 900)
    xs = [220, 515, 825, 1120]
    x_ctrl, x_replacement, x_source, x_barrier = xs
    titles = [
        "恢复控制器\n（101）",
        "替换逻辑计算进程",
        "状态来源（104/105）\n健康对等进程/检查点",
        "更新屏障与日志\n（107/108）",
    ]
    for i, (x, title) in enumerate(zip(xs, titles), 1):
        d.rect(f"f3_head{i}", title, x - 108, 25, 216, 70)
        d.coord_edge(x, 95, x, 835, dashed=False)
    d.label("阶段一：权重优先", 5, 315, 165, 55, font_size=18)
    d.label("阶段二：优化器稍后", 5, 610, 165, 55, font_size=18)
    arrows = [
        (145, x_ctrl, x_replacement, "分配替换进程"),
        (220, x_ctrl, x_barrier, "安装安全点和提交保护"),
        (315, x_replacement, x_source, "请求非专家状态和专家权重"),
        (405, x_source, x_replacement, "返回状态和权重"),
        (495, x_replacement, x_barrier, "权重就绪，进入已修复状态"),
        (570, x_barrier, x_replacement, "允许前向/反向；暂存专家梯度"),
        (650, x_replacement, x_source, "后台请求专家优化器状态"),
        (720, x_source, x_replacement, "返回专家优化器状态"),
        (780, x_replacement, x_barrier, "按训练步顺序处理暂存梯度"),
        (820, x_ctrl, x_barrier, "释放更新屏障并记录日志"),
    ]
    for y, x1, x2, label in arrows:
        d.coord_edge(x1, y, x2, y, label=label)
    d.rect(
        "f3_result",
        "状态迁移：恢复中（RECOVERING）→ 已修复（REPAIRED）→ 屏障等待（BARRIER）→ 健康（HEALTHY）",
        145,
        845,
        1060,
        45,
        font_size=16,
    )
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
