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


def p_text(p):
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def has_only_page_break(p):
    text = p_text(p).strip()
    drawings = p.xpath(".//w:drawing", namespaces=NS)
    brs = p.xpath(".//w:br[@w:type='page']", namespaces=NS)
    return not text and not drawings and bool(brs)


def ensure_page_break_before(p):
    ppr = p.find("w:pPr", namespaces=NS)
    if ppr is None:
        ppr = etree.Element(W + "pPr")
        p.insert(0, ppr)
    if ppr.find("w:pageBreakBefore", namespaces=NS) is None:
        etree.SubElement(ppr, W + "pageBreakBefore")


def patch(path: Path):
    backup = path.with_name(path.stem + f"_图页分页修复前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")
    with ZipFile(path, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        body = root.find("w:body", namespaces=NS)
        paras = body.findall("w:p", namespaces=NS)

        # Remove standalone page-break paragraphs immediately before figure headings.
        for p in list(paras):
            nxt = p.getnext()
            if nxt is not None and p_text(nxt) in {"图1", "图2", "图3"} and has_only_page_break(p):
                body.remove(p)

        for p in body.findall("w:p", namespaces=NS):
            if p_text(p) in {"图1", "图2", "图3"}:
                ensure_page_break_before(p)

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
