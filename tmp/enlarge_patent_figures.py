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
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

# Use nearly the full A4 text width while keeping each figure's aspect ratio.
TARGET_CX = 6_300_000
ASPECTS = [
    (2400, 1200),  # Fig. 1
    (1900, 1900),  # Fig. 2
    (2400, 1350),  # Fig. 3
]


def patch(path: Path):
    backup = path.with_name(path.stem + f"_附图放大前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")
    with ZipFile(path, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        drawings = root.findall(".//w:drawing", namespaces=NS)
        if len(drawings) != 3:
            raise RuntimeError(f"expected 3 drawings, found {len(drawings)}")
        for drawing, (w, h) in zip(drawings, ASPECTS):
            cy = int(TARGET_CX * h / w)
            for extent in drawing.findall(".//wp:extent", namespaces=NS):
                extent.set("cx", str(TARGET_CX))
                extent.set("cy", str(cy))
            for ext in drawing.findall(".//a:ext", namespaces=NS):
                ext.set("cx", str(TARGET_CX))
                ext.set("cy", str(cy))
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
