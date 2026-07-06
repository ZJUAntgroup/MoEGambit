from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import time

from lxml import etree


TARGETS = [
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请.docx"),
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx"),
]

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

REPLACEMENTS = {
    "本申请的一些实施例": "本发明的一些实施例",
    "并输出Hybrid或Restart决策": "并输出混合恢复或检查点重启决策",
    "任一rank到达optimizer.step()之前均检查该标记": "任一rank到达优化器提交点之前均检查该标记",
    "从而保证恢复策略具有保守fallback": "从而保证恢复策略具有保守回退机制",
}


def patch(path: Path):
    backup = path.with_name(path.stem + f"_语言尾项修改前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")
    with ZipFile(path, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        for wt in root.findall(".//w:t", namespaces=NS):
            if not wt.text:
                continue
            for old, new in REPLACEMENTS.items():
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
