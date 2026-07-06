from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import time

from lxml import etree


TARGETS = [
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请.docx"),
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx"),
]

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
W = f"{{{NS['w']}}}"
M = f"{{{NS['m']}}}"


INSERT_AFTER = {
    "本发明属于大模型分布式训练、混合专家模型训练和运行时容错恢复技术领域，具体涉及一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统，尤其适用于多机多卡GPU集群中采用流水线并行、密集张量并行、专家并行和专家张量并行进行大规模预训练或继续训练的场景。":
        "为便于描述，本文中的rank是指分布式训练作业中的逻辑计算进程或设备编号；peer是指在相同并行坐标下持有相应复制状态的健康对等rank；dense-DP是指非专家层所在的密集数据并行维度。上述术语仅用于说明训练系统中的状态来源关系，不限定具体训练框架或通信库实现。"
}

PARA_REPLACEMENTS = {
    "12. 一种实现权利要求1-11任一项所述方法的面向稀疏混合专家模型训练的运行时混合恢复系统，其特征在于，包括：故障检测与替换rank管理模块，用于接收或生成失效rank事件并分配替换rank；安全点控制模块，用于标记恢复中状态并安装优化器提交保护；状态分类模块，用于区分非专家状态、专家状态和运行时元数据；恢复策略判定模块，用于计算Δ、S(t)和Φ'(t)，并基于peer可用性和阈值约束选择混合恢复路径或检查点重启路径；状态恢复模块，用于从健康peer和检查点分别恢复对应状态；两阶段恢复模块，用于执行权重优先和优化器稍后的恢复过程；重集成与日志模块，用于重建元数据、释放更新屏障并生成结构化恢复日志。":
        "12. 一种实现权利要求1-11任一项所述方法的面向稀疏混合专家模型训练的运行时混合恢复系统，其特征在于，包括：故障检测与替换rank管理模块，用于接收或生成失效rank事件并分配替换rank；安全点控制模块，用于标记恢复中状态并安装优化器提交保护；状态分类模块，用于区分非专家状态、专家状态和运行时元数据；恢复策略判定模块，用于计算Δ、S(t)和Φ'(t)，并基于peer可用性和阈值约束选择混合恢复路径或检查点重启路径；状态恢复模块，用于从健康peer、专家peer或检查点恢复对应状态；两阶段恢复模块，用于执行权重优先和优化器稍后的恢复过程；重集成与结构化日志模块，用于重建元数据、释放更新屏障并生成结构化恢复日志。",
    "14. 根据权利要求12所述的系统，其特征在于，所述状态恢复模块包括peer拉取子模块和分片读取子模块；所述peer拉取子模块用于在相同流水线并行坐标和密集张量并行坐标下从健康dense-DP peer同步当前非专家状态；所述分片读取子模块用于从检查点读取失效rank对应的专家分片。":
        "14. 根据权利要求12所述的系统，其特征在于，所述状态恢复模块包括peer拉取子模块、检查点分片读取子模块和专家peer读取子模块；所述peer拉取子模块用于在相同流水线并行坐标和密集张量并行坐标下从健康dense-DP peer同步当前非专家状态；所述检查点分片读取子模块用于在专家状态无健康副本时从检查点读取失效rank对应的专家分片；所述专家peer读取子模块用于在专家peer可用时从健康专家peer同步专家状态。",
    "本发明还提供一种实现上述方法的运行时混合恢复系统，包括故障检测与替换rank管理模块、安全点控制模块、状态分类模块、恢复策略判定模块、状态恢复模块、两阶段恢复模块以及重集成与日志模块。":
        "本发明还提供一种实现上述方法的运行时混合恢复系统，包括故障检测与替换rank管理模块、安全点控制模块、状态分类模块、恢复策略判定模块、状态恢复模块、两阶段恢复模块以及重集成与结构化日志模块。",
}

INLINE_REPLACEMENTS = {
    "本申请实施例": "本发明实施例",
    "fallback原因": "回退原因",
    "合规审计": "运行审计",
}


def paragraph_text(p):
    parts = []
    for node in p.iter():
        if node.tag in {W + "t", M + "t"} and node.text:
            parts.append(node.text)
    return "".join(parts)


def replace_text_only_para(p, text):
    ppr = p.find("w:pPr", namespaces=NS)
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    r = etree.SubElement(p, W + "r")
    rpr = etree.SubElement(r, W + "rPr")
    rfonts = etree.SubElement(rpr, W + "rFonts")
    rfonts.set(W + "ascii", "Times New Roman")
    rfonts.set(W + "hAnsi", "Times New Roman")
    rfonts.set(W + "eastAsia", "宋体")
    etree.SubElement(rpr, W + "b").set(W + "val", "0")
    etree.SubElement(rpr, W + "sz").set(W + "val", "24")
    t = etree.SubElement(r, W + "t")
    t.text = text


def make_para_like(reference, text):
    p = etree.Element(W + "p")
    ppr = reference.find("w:pPr", namespaces=NS)
    if ppr is not None:
        p.append(deepcopy(ppr))
    r = etree.SubElement(p, W + "r")
    rpr = etree.SubElement(r, W + "rPr")
    rfonts = etree.SubElement(rpr, W + "rFonts")
    rfonts.set(W + "ascii", "Times New Roman")
    rfonts.set(W + "hAnsi", "Times New Roman")
    rfonts.set(W + "eastAsia", "宋体")
    etree.SubElement(rpr, W + "b").set(W + "val", "0")
    etree.SubElement(rpr, W + "sz").set(W + "val", "24")
    t = etree.SubElement(r, W + "t")
    t.text = text
    return p


def patch(path: Path):
    backup = path.with_name(path.stem + f"_内容一致性修改前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")
    with ZipFile(path, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        body = root.find("w:body", namespaces=NS)

        for p in list(body.findall("w:p", namespaces=NS)):
            text = paragraph_text(p)
            if text in INSERT_AFTER:
                p.addnext(make_para_like(p, INSERT_AFTER[text]))
            if text in PARA_REPLACEMENTS:
                replace_text_only_para(p, PARA_REPLACEMENTS[text])

        for wt in root.findall(".//w:t", namespaces=NS):
            if not wt.text:
                continue
            for old, new in INLINE_REPLACEMENTS.items():
                wt.text = wt.text.replace(old, new)

        new_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        with ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "word/document.xml":
                    data = new_xml
                zout.writestr(info, data)
    tmp.replace(path)
    return backup


if __name__ == "__main__":
    for target in TARGETS:
        print(patch(target))
