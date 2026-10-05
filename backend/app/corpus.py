"""Turn eCFR section files into citable passages.

Each passage carries the exact citation a reviewer would write, e.g.
"21 CFR 211.84(d)(2)". A paragraph is kept whole unless it is long and has
numbered sub-paragraphs of sentence length, in which case each sub-paragraph
becomes a passage and inherits the parent's lead-in sentence as context.
"""
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from . import config

HEADING = re.compile(r"^#+\s*§\s*([\d.]+)\s+(.*)$")
MARKER = re.compile(r"^\((\w{1,5})\)\s+")
EDITORIAL = re.compile(r"^(\\\[.*\\\]|\[.*\]|Link to an amendment.*)$")
ROMAN = re.compile(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)$")
SPLIT_OVER = 900          # characters; longer paragraphs with numbered children are split
MIN_KID_CHARS = 150       # ...but only if the children are real sentences, not list items like "(1) Dates;"


@dataclass
class Passage:
    id: str               # "211.84(d)(2)"
    citation: str         # "21 CFR 211.84(d)(2)"
    part: str
    section: str          # "211.84"
    section_title: str
    subpart: str
    text: str             # the regulation's own words, verbatim
    context: str          # lead-in inherited from the parent paragraph, may be ""

    @property
    def search_text(self):
        return f"{self.section_title}. {self.context} {self.text}"


def _level(tok, state):
    """1 = (a), 2 = (1), 3 = (i). '(i)' is a letter only straight after '(h)'."""
    if tok.isdigit():
        return 2
    if ROMAN.match(tok) and state["have2"] and not (tok == "i" and state["last1"] == "h" and not state["in2"]):
        return 3
    return 1 if tok.isalpha() and len(tok) == 1 else 3


def parse_section(path: Path):
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l and not EDITORIAL.match(l)]
    m = HEADING.match(lines[0])
    if not m:
        raise ValueError(f"no section heading in {path}")
    section, title = m.group(1), m.group(2).rstrip(".")
    part = section.split(".")[0]
    subpart = path.parent.name.replace("-", " ") if path.parent.name.startswith("subpart") else ""

    intro, paras = [], []          # paras: {tok, lines:[...], kids:[{tok, lines}]}
    state = {"have2": False, "last1": "", "in2": False}
    for line in lines[1:]:
        mk = MARKER.match(line)
        level = _level(mk.group(1), state) if mk else 0
        if level == 1:
            paras.append({"tok": mk.group(1), "lines": [line], "kids": []})
            state.update(last1=mk.group(1), have2=False, in2=False)
        elif level == 2 and paras:
            paras[-1]["kids"].append({"tok": mk.group(1), "lines": [line]})
            state.update(have2=True, in2=True)
        elif level == 2:                       # numbered list straight under the intro
            paras.append({"tok": mk.group(1), "lines": [line], "kids": []})
        elif paras:
            (paras[-1]["kids"][-1] if paras[-1]["kids"] else paras[-1])["lines"].append(line)
        else:
            intro.append(line)

    def mk(pid, text, context=""):
        return Passage(id=pid, citation=f"21 CFR {pid}", part=part, section=section,
                       section_title=title, subpart=subpart, text=text, context=context)

    intro_text = " ".join(intro)
    if not paras:
        return [mk(section, intro_text)]
    out = []
    for p in paras:
        head = " ".join(p["lines"])
        whole = " ".join([head] + [" ".join(k["lines"]) for k in p["kids"]])
        pid = f"{section}({p['tok']})"
        kid_len = sum(len(" ".join(k["lines"])) for k in p["kids"]) / max(len(p["kids"]), 1)
        if p["kids"] and len(whole) > SPLIT_OVER and kid_len >= MIN_KID_CHARS:
            out.append(mk(pid, head, intro_text))
            for k in p["kids"]:
                out.append(mk(f"{pid}({k['tok']})", " ".join(k["lines"]), head))
        else:
            out.append(mk(pid, whole, intro_text))
    return out


def _sort_key(p: Passage):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.()]+", p.id) if x]


@lru_cache(maxsize=1)
def load_corpus():
    passages = []
    for path in sorted(config.REG_DIR.rglob("section-*.md")):
        passages.extend(parse_section(path))
    ids = [p.id for p in passages]
    assert len(ids) == len(set(ids)), "duplicate passage ids"
    return sorted(passages, key=lambda p: (int(p.part), [str(k).zfill(6) for k in _sort_key(p)]))


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"{len(corpus)} passages from {len({p.section for p in corpus})} sections")
    for pid in ("211.84(d)(2)", "11.10(e)", "211.192", "210.3(b)(4)"):
        p = next(x for x in corpus if x.id == pid)
        print(json.dumps(asdict(p), indent=1)[:600], "\n")
