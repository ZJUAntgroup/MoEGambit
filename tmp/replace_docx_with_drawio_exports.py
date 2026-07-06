from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import time

from lxml import etree
from PIL import Image


TARGETS = [
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请.docx"),
    Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx"),
]

EXPORT_DIR = Path("/Users/zds/Desktop/MoEGambit_patent_drawio/exported")
PNG_MAP = {
    "word/media/image1.png": EXPORT_DIR / "图1_运行时混合恢复系统架构.png",
    "word/media/image2.png": EXPORT_DIR / "图2_运行时混合恢复方法流程.png",
    "word/media/image3.png": EXPORT_DIR / "图3_两阶段恢复时序.png",
}

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

TARGET_CX = 6_300_000


def patch(path: Path):
    backup = path.with_name(path.stem + f"_drawio附图替换前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")
    sizes = []
    for media_path in ("word/media/image1.png", "word/media/image2.png", "word/media/image3.png"):
        with Image.open(PNG_MAP[media_path]) as im:
            w, h = im.size
        sizes.append((TARGET_CX, int(TARGET_CX * h / w)))

    with ZipFile(path, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        drawings = root.findall(".//w:drawing", namespaces=NS)
        if len(drawings) != 3:
            raise RuntimeError(f"expected 3 drawings, found {len(drawings)}")
        for drawing, (cx, cy) in zip(drawings, sizes):
            for extent in drawing.findall(".//wp:extent", namespaces=NS):
                extent.set("cx", str(cx))
                extent.set("cy", str(cy))
            for ext in drawing.findall(".//a:ext", namespaces=NS):
                ext.set("cx", str(cx))
                ext.set("cy", str(cy))
        new_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")

        with ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "word/document.xml":
                    data = new_xml
                elif info.filename in PNG_MAP:
                    data = PNG_MAP[info.filename].read_bytes()
                zout.writestr(info, data)

    tmp.replace(path)
    return backup


if __name__ == "__main__":
    for target in TARGETS:
        print(patch(target))
