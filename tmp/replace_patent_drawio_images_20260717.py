#!/usr/bin/env python3
from __future__ import annotations

import shutil
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree
from PIL import Image


TARGET = Path(
    "/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx"
)
EXPORT_DIR = Path("/Users/zds/Desktop/MoEGambit_patent_drawio/exported")
IMAGES = [
    EXPORT_DIR / "图1_运行时混合恢复系统架构.png",
    EXPORT_DIR / "图2_运行时混合恢复方法流程.png",
    EXPORT_DIR / "图3_两阶段恢复时序.png",
]
MEDIA_NAMES = [
    "word/media/image1.png",
    "word/media/image2.png",
    "word/media/image3.png",
]
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

# Keep figures inside the A4 text area while preserving each exported aspect ratio.
TARGET_CX = 6_050_000


def replace_images() -> Path:
    for image in IMAGES:
        if not image.exists():
            raise FileNotFoundError(image)

    backup = TARGET.with_name(
        TARGET.stem + f"_附图全面修改前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx"
    )
    shutil.copy2(TARGET, backup)
    tmp = TARGET.with_suffix(".tmp.docx")

    dimensions = []
    for image in IMAGES:
        with Image.open(image) as raster:
            dimensions.append(raster.size)

    with ZipFile(TARGET, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        drawings = root.findall(".//w:drawing", namespaces=NS)
        if len(drawings) != len(IMAGES):
            raise RuntimeError(f"expected {len(IMAGES)} drawings, found {len(drawings)}")

        for drawing, (width, height) in zip(drawings, dimensions):
            target_cy = round(TARGET_CX * height / width)
            for extent in drawing.findall(".//wp:extent", namespaces=NS):
                extent.set("cx", str(TARGET_CX))
                extent.set("cy", str(target_cy))
            for extent in drawing.findall(".//a:ext", namespaces=NS):
                extent.set("cx", str(TARGET_CX))
                extent.set("cy", str(target_cy))

        document_xml = etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone="yes",
        )
        replacements = dict(zip(MEDIA_NAMES, IMAGES))
        with ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "word/document.xml":
                    data = document_xml
                elif info.filename in replacements:
                    data = replacements[info.filename].read_bytes()
                zout.writestr(info, data)

    tmp.replace(TARGET)
    return backup


if __name__ == "__main__":
    print(replace_images())
