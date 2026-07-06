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
W = f"{{{NS['w']}}}"

TEXT_REPLACEMENTS = {
    "不能覆盖EDP=1时rank本地专家分片无副本的情形": "不能覆盖专家数据并行度为一时rank本地专家分片无副本的情形",
    "EDP=1并非作为不可恢复条件": "专家数据并行度为一并非作为不可恢复条件",
    "当EDP>1且专家peer可用时": "当专家数据并行度大于一且专家peer可用时",
    "混合专家模型EDP=1场景": "混合专家模型专家数据并行度为一的场景",
    "在EDP=1时仍可恢复非专家复制态": "在专家数据并行度为一时仍可恢复非专家复制态",
}


def patch(path: Path):
    backup = path.with_name(path.stem + f"_公式文本规范化前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")
    with ZipFile(path, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        for wt in root.findall(".//w:t", namespaces=NS):
            if not wt.text:
                continue
            for old, new in TEXT_REPLACEMENTS.items():
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
