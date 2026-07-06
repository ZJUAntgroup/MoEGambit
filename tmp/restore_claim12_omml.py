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


def para_text(p):
    return "".join(t for t in p.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))


def math_text(m):
    return "".join(t for t in m.xpath(".//m:t/text()", namespaces=NS))


def make_run(text):
    r = etree.Element(W + "r")
    rpr = etree.SubElement(r, W + "rPr")
    rfonts = etree.SubElement(rpr, W + "rFonts")
    rfonts.set(W + "ascii", "Times New Roman")
    rfonts.set(W + "hAnsi", "Times New Roman")
    rfonts.set(W + "eastAsia", "宋体")
    etree.SubElement(rpr, W + "b").set(W + "val", "0")
    etree.SubElement(rpr, W + "sz").set(W + "val", "24")
    t = etree.SubElement(r, W + "t")
    t.text = text
    return r


def patch(path: Path):
    backup = path.with_name(path.stem + f"_权利要求12公式修复前备份_{time.strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(path, backup)
    tmp = path.with_suffix(".tmp.docx")
    with ZipFile(path, "r") as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        samples = {}
        for m in root.findall(".//m:oMath", namespaces=NS):
            txt = math_text(m)
            if txt in {"Δ", "S(t)", "Φ'(t)"} and txt not in samples:
                samples[txt] = deepcopy(m)
        missing = {"Δ", "S(t)", "Φ'(t)"} - set(samples)
        if missing:
            raise RuntimeError(f"missing formula samples: {missing}")

        for p in root.findall(".//w:body/w:p", namespaces=NS):
            text = para_text(p)
            if not text.startswith("12. 一种实现权利要求1-11任一项所述方法"):
                continue
            ppr = p.find("w:pPr", namespaces=NS)
            for child in list(p):
                if child is not ppr:
                    p.remove(child)
            prefix, rest = text.split("Δ、S(t)和Φ'(t)", 1)
            p.append(make_run(prefix))
            p.append(deepcopy(samples["Δ"]))
            p.append(make_run("、"))
            p.append(deepcopy(samples["S(t)"]))
            p.append(make_run("和"))
            p.append(deepcopy(samples["Φ'(t)"]))
            p.append(make_run(rest))
            break

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
