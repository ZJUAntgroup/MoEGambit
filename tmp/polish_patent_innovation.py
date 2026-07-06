from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import time

from lxml import etree


MAIN = Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请.docx")
FORMULA = Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx")

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
W = f"{{{NS['w']}}}"
M = f"{{{NS['m']}}}"


ABSTRACT_NEW = (
    "本发明公开了一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统，"
    "属于大模型分布式训练与容错恢复技术领域。该方法在rank失效时建立安全点并阻止不完整优化器提交，"
    "将训练状态划分为非专家复制态、专家分片态和运行时元数据；根据检查点间隔、专家陈旧债务和专家加权陈旧密度判定恢复路径。"
    "满足阈值时，从健康dense-DP peer拉取当前非专家状态，并从检查点或专家peer恢复专家状态；"
    "再通过权重优先、优化器稍后的两阶段协议恢复训练。该方案减少全局回滚、检查点读写和重放迭代开销，"
    "并生成结构化日志以支持恢复决策审计。"
)

REPLACEMENTS = {
    "针对上述问题，本发明提出一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统，旨在解决现有检查点重启恢复开销大、密集peer恢复无法覆盖EDP=1专家分片、以及部分恢复缺乏质量风险约束和审计机制的问题。":
        "针对上述问题，本发明提出一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统，所要解决的技术问题包括：现有检查点重启将检查点步到故障步之间已经完成的GPU计算整体丢弃；密集peer恢复要求失效rank的全部状态均存在健康副本，不能覆盖EDP=1时rank本地专家分片无副本的情形；直接进行部分恢复又缺乏对专家状态陈旧暴露的运行时约束和审计记录。",
    "本发明的核心思想在于：将混合专家训练状态按照恢复来源划分为非专家复制态、专家分片态和运行时元数据；在rank失效时优先从健康密集数据并行peer恢复当前非专家状态，仅对缺乏健康副本的专家状态执行检查点分片恢复；同时通过专家加权陈旧密度Φ'(t)约束专家状态陈旧暴露，并通过安全点控制、两阶段恢复和结构化日志保证恢复语义正确和可审计。":
        "本发明的区别性技术方案在于：首先按照状态来源将训练状态划分为非专家复制态、专家分片态和运行时元数据；其次对失效事件建立安全点并安装优化器提交保护；再次根据健康peer可用性、检查点间隔和专家加权陈旧密度选择混合恢复或检查点重启；最后通过权重优先、优化器稍后的两阶段协议与结构化日志完成重集成。",
    "与现有技术相比，本发明至少具有以下技术效果：第一，避免将全部训练状态整体回滚，减少检查点加载和重放迭代开销；第二，针对EDP=1下专家状态无peer副本的特殊性，允许非专家状态peer恢复与专家状态检查点恢复并存；第三，通过专家加权陈旧密度对混合恢复风险进行运行时约束，避免无界累积陈旧专家状态；第四，通过安全点控制和两阶段恢复避免不完整优化器提交；第五，通过结构化日志提高恢复过程的可审计性和可复现性。":
        "与现有技术相比，本发明至少具有以下技术效果：第一，以状态来源为依据拆分恢复路径，在允许时避免从检查点恢复全部状态，减少检查点加载和重放迭代开销；第二，在EDP=1时仍可恢复非专家复制态，并仅对无健康副本的专家分片使用检查点，扩大peer恢复在混合专家训练中的适用范围；第三，以专家加权陈旧密度限制窗口内专家陈旧暴露，使混合恢复具有可检查的运行时准入条件；第四，通过安全点控制和两阶段恢复避免不完整优化器提交；第五，通过结构化日志记录阈值输入、路径决策和状态迁移，提高恢复过程的可审计性。",
}

INSERT_AFTER = {
    "本发明的区别性技术方案在于：首先按照状态来源将训练状态划分为非专家复制态、专家分片态和运行时元数据；其次对失效事件建立安全点并安装优化器提交保护；再次根据健康peer可用性、检查点间隔和专家加权陈旧密度选择混合恢复或检查点重启；最后通过权重优先、优化器稍后的两阶段协议与结构化日志完成重集成。":
        "在该方案中，EDP=1并非作为不可恢复条件，而是作为专家状态来源选择条件：非专家状态仍从健康dense-DP peer取得，专家状态从检查点分片取得；当EDP>1且专家peer可用时，专家状态也可以从健康专家peer取得。由此，本发明同时覆盖密集模型可peer恢复场景和混合专家模型EDP=1场景。",
    "图3：混合恢复与两阶段协议时序图，展示替换rank、健康dense-DP peer、检查点存储和全局更新屏障之间的交互关系。":
        "附图标记说明：101、恢复控制器；102、恢复策略判定模块；103、状态分类模块；104、健康dense-DP peer；105、专家状态来源；106、混合状态恢复模块；107、两阶段恢复模块；108、重集成与结构化日志模块；109、检查点重启路径。图中相同或相似标号表示相同或相似模块。",
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


def patch_docx(path):
    backup = path.with_name(path.stem + f"_完善前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")

    with ZipFile(path, "r") as zin:
        xml = zin.read("word/document.xml")
        root = etree.fromstring(xml)
        body = root.find("w:body", namespaces=NS)
        paras = body.findall("w:p", namespaces=NS)

        # Abstract follows the centered "摘要" heading.
        for i, p in enumerate(paras[:-1]):
            if paragraph_text(p) == "摘要":
                replace_text_only_para(paras[i + 1], ABSTRACT_NEW)
                break

        # Replace prose paragraphs while leaving equation-bearing paragraphs untouched.
        paras = body.findall("w:p", namespaces=NS)
        for p in paras:
            text = paragraph_text(p)
            if text in REPLACEMENTS:
                replace_text_only_para(p, REPLACEMENTS[text])

        # Insert clarifying patent paragraphs.
        paras = body.findall("w:p", namespaces=NS)
        for p in list(paras):
            text = paragraph_text(p)
            if text in INSERT_AFTER:
                p.addnext(make_para_like(p, INSERT_AFTER[text]))

        # Clean residual LaTeX command names that should not be visible inside Word equations.
        for mt in root.findall(".//m:t", namespaces=NS):
            if mt.text:
                mt.text = (
                    mt.text.replace("\\min", "min")
                    .replace("\\max", "max")
                    .replace("\\mathrm", "")
                )

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
    backups = []
    for target in (MAIN, FORMULA):
        backups.append(patch_docx(target))
    print(f"abstract_chars={len(ABSTRACT_NEW)}")
    for backup in backups:
        print(f"backup={backup}")
