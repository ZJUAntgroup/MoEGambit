from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree


SRC = Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请.docx")
OUT = Path("/Users/zds/Desktop/一种面向稀疏混合专家模型训练的运行时混合恢复方法及系统_专利申请_公式版.docx")
XSL = Path("/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl")
WORK = Path("/Users/zds/bsr/tmp/patent_formula_omml_work")

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
MATHML_NS = "http://www.w3.org/1998/Math/MathML"


class LatexParser:
    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def parse(self):
        return self.parse_until(None)

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def consume(self) -> str:
        ch = self.peek()
        self.i += 1
        return ch

    def parse_until(self, end: str | None):
        nodes = []
        while self.i < len(self.text):
            if end is not None and self.peek() == end:
                self.i += 1
                break
            nodes.append(self.parse_atom())
        return mrow(nodes)

    def parse_group(self):
        self.skip_spaces()
        if self.peek() == "{":
            self.consume()
            return self.parse_until("}")
        return self.parse_atom()

    def parse_atom(self):
        self.skip_spaces()
        ch = self.peek()
        if ch == "\\":
            node = self.parse_command()
        elif ch == "{":
            self.consume()
            node = self.parse_until("}")
        elif ch.isalpha():
            node = mi(self.consume())
        elif ch.isdigit():
            node = mn(self.consume())
        else:
            node = mo(self.consume())

        while self.peek() in ("_", "^"):
            op = self.consume()
            script = self.parse_group()
            if op == "_":
                node = wrap_script("msub", node, script)
            else:
                node = wrap_script("msup", node, script)
        return node

    def parse_command(self):
        assert self.consume() == "\\"
        name = []
        while self.peek().isalpha():
            name.append(self.consume())
        cmd = "".join(name)

        if cmd == "frac":
            num = self.parse_group()
            den = self.parse_group()
            return elem("mfrac", num, den)
        if cmd == "mathrm":
            group_text = self.read_group_text()
            return mi(group_text, normal=True)
        if cmd == "text":
            group_text = self.read_group_text()
            return mtext(group_text)
        if cmd in {"Delta", "Phi"}:
            return mi({"Delta": "Δ", "Phi": "Φ"}[cmd])
        if cmd == "sum":
            return mo("∑")
        if cmd == "times":
            return mo("×")
        if cmd == "le":
            return mo("≤")
        if cmd == "ge":
            return mo("≥")
        if cmd == "lt":
            return mo("<")
        if cmd == "gt":
            return mo(">")
        if cmd == "langle":
            return mo("⟨")
        if cmd == "rangle":
            return mo("⟩")
        if cmd == "in":
            return mo("∈")
        if cmd == "cdot":
            return mo("·")
        # Fallback: keep unknown command visibly, but inside a formula object.
        return mi("\\" + cmd)

    def read_group_text(self) -> str:
        self.skip_spaces()
        if self.peek() != "{":
            return ""
        self.consume()
        depth = 1
        start = self.i
        while self.i < len(self.text) and depth:
            ch = self.consume()
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        return self.text[start : self.i - 1]

    def skip_spaces(self):
        while self.peek().isspace():
            self.consume()


def elem(tag: str, *children):
    e = etree.Element(f"{{{MATHML_NS}}}{tag}")
    for c in children:
        if isinstance(c, list):
            for item in c:
                e.append(item)
        else:
            e.append(c)
    return e


def token(tag: str, text: str, normal: bool = False):
    e = etree.Element(f"{{{MATHML_NS}}}{tag}")
    if normal:
        e.set("mathvariant", "normal")
    e.text = text
    return e


def mi(text: str, normal: bool = False):
    return token("mi", text, normal=normal)


def mn(text: str):
    return token("mn", text)


def mo(text: str):
    return token("mo", text)


def mtext(text: str):
    return token("mtext", text)


def mrow(nodes):
    # A single child need not be wrapped, but wrapping keeps XSLT output stable.
    return elem("mrow", nodes)


def wrap_script(kind: str, base, script):
    return elem(kind, base, script)


def latex_to_mathml(latex: str):
    math = etree.Element(f"{{{MATHML_NS}}}math", nsmap={None: MATHML_NS})
    math.append(LatexParser(latex).parse())
    return math


def build_transform():
    if not XSL.exists():
        raise FileNotFoundError(f"Cannot find Word mathml2omml.xsl: {XSL}")
    return etree.XSLT(etree.parse(str(XSL)))


def latex_to_omml(latex: str, transform):
    mathml = latex_to_mathml(latex)
    result = transform(mathml)
    root = result.getroot()
    # Word's XSLT usually returns m:oMath; keep the first math element.
    if etree.QName(root).localname in {"oMath", "oMathPara"}:
        omml = root
    else:
        found = root.find(".//m:oMath", namespaces=NS)
        if found is None:
            raise ValueError(f"No OMML produced for {latex!r}")
        omml = found
    return etree.fromstring(etree.tostring(omml))


FORMULA_RE = re.compile(r"\\\((.*?)\\\)")


def paragraph_text(p):
    return "".join(t.text or "" for t in p.findall(".//w:t", namespaces=NS))


def make_text_run(text: str):
    r = etree.Element(f"{{{NS['w']}}}r")
    t = etree.SubElement(r, f"{{{NS['w']}}}t")
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def replace_paragraph_content(p, text: str, transform):
    # Preserve paragraph properties, then rebuild its inline content.
    ppr = p.find("w:pPr", namespaces=NS)
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    last = 0
    for m in FORMULA_RE.finditer(text):
        if m.start() > last:
            p.append(make_text_run(text[last : m.start()]))
        p.append(latex_to_omml(m.group(1), transform))
        last = m.end()
    if last < len(text):
        p.append(make_text_run(text[last:]))


def patch_docx():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(SRC) as z:
        z.extractall(WORK)

    document_xml = WORK / "word" / "document.xml"
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(document_xml), parser)
    transform = build_transform()

    changed = 0
    for p in tree.findall(".//w:p", namespaces=NS):
        text = paragraph_text(p)
        if FORMULA_RE.search(text):
            replace_paragraph_content(p, text, transform)
            changed += 1

    tree.write(str(document_xml), xml_declaration=True, encoding="UTF-8", standalone=True)

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for path in WORK.rglob("*"):
            if path.is_file():
                zout.write(path, path.relative_to(WORK))

    return changed


if __name__ == "__main__":
    changed = patch_docx()
    print(f"patched_paragraphs={changed}")
    print(OUT)
