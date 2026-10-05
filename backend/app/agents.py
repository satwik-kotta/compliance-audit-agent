"""The audit pipeline: five small agents, each with one job.

  ClauseExtractor   proposal text            -> individual commitments
  RetrievalAgent    clause                   -> candidate regulation passages
  AnalystAgent      clause + passages        -> verdict, citation, verbatim quote
  CitationVerifier  finding                  -> checks the quote really is in the cited passage
  ReportWriter      verified findings        -> audit report (markdown)

The analyst is the only step that uses an LLM. The verifier is plain code on
purpose: a model can argue itself into a citation, a string comparison cannot.
"""
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from . import config
from .corpus import Passage
from .llm import describe
from .retrieval import HybridRetriever, tokenize

Verdict = Literal["non_compliant", "needs_review", "compliant", "no_flag", "not_addressed"]


@dataclass
class Finding:
    clause_no: int
    clause: str
    verdict: Verdict
    citation: Optional[str] = None           # "21 CFR 211.165(a)"
    passage_id: Optional[str] = None
    quote: Optional[str] = None              # verbatim regulation text
    rationale: str = ""
    recommendation: str = ""
    citation_verified: bool = False
    verifier_note: str = ""
    retrieved: list = field(default_factory=list)   # [{id, citation, score}]


# ---------------------------------------------------------------- 1. extractor
class ClauseExtractor:
    NUMBERED = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•]|\([a-z0-9]{1,3}\))\s+")
    HEADING = re.compile(r"^(#+\s|[A-Z][A-Za-z ,&/-]{0,60}:?$)")

    def run(self, text: str):
        text = text.replace("\r", "")[: config.MAX_PROPOSAL_CHARS]
        blocks, current, numbered = [], [], False      # blocks: (is_numbered, text)

        def flush():
            if current:
                blocks.append((numbered, " ".join(current)))
            current.clear()

        for line in text.split("\n"):
            if not line.strip():
                flush()
            elif self.NUMBERED.match(line):
                flush()
                numbered = True
                current.append(self.NUMBERED.sub("", line).strip())
            else:
                if not current:
                    numbered = False
                current.append(line.strip())
        flush()
        # A document written as a numbered list: the list items are the commitments,
        # and unnumbered text around them is title, preamble or sign-off.
        if sum(n for n, _ in blocks) >= 3:
            blocks = [b for b in blocks if b[0]]
        clauses = []
        for _, b in blocks:
            b = re.sub(r"\s+", " ", b).strip()
            if len(b) < 40 or (self.HEADING.match(b) and len(b) < 80):
                continue                       # titles and stray fragments carry no commitment
            if len(b) > 700:                   # an unnumbered wall of text: fall back to sentences
                clauses.extend(x.strip() for x in re.split(r"(?<=[.;])\s+(?=[A-Z])", b) if len(x.strip()) >= 40)
            else:
                clauses.append(b)
        return clauses[: config.MAX_CLAUSES]


# ---------------------------------------------------------------- 2. retrieval
class RetrievalAgent:
    def __init__(self, retriever: HybridRetriever, k=config.TOP_K):
        self.retriever, self.k = retriever, k

    def run(self, clause: str):
        return self.retriever.search(clause, k=self.k)


# ---------------------------------------------------------------- 3. analyst
class Judgement(BaseModel):
    verdict: Verdict = Field(description="non_compliant | needs_review | compliant | not_addressed")  # no_flag is baseline-only
    passage_id: Optional[str] = Field(None, description="id of the single most relevant passage, exactly as given")
    quote: Optional[str] = Field(None, description="one sentence copied word for word from that passage")
    rationale: str = Field("", description="one or two sentences, plain language")
    recommendation: str = Field("", description="what to change; empty if compliant")


SYSTEM = """You are a GMP compliance analyst reviewing one clause of an internal company proposal \
against U.S. FDA regulations (21 CFR Parts 11, 210, 211).

Judge the clause ONLY against the passages provided. Do not rely on regulations you remember.

Verdicts:
- non_compliant: the clause states something a passage prohibits, or omits/waives something a passage requires.
- needs_review: the clause touches a requirement but is too vague to tell, or depends on facts not stated.
- compliant: the clause is consistent with the passages.
- not_addressed: none of the passages bear on this clause.

For every verdict except not_addressed, give passage_id exactly as shown and a quote copied word \
for word from that passage (one sentence, no paraphrase, no ellipsis). The quote is checked by \
software; a quote that is not in the passage is discarded.

{format_instructions}"""

HUMAN = """Clause {n}:
{clause}

Passages:
{passages}"""


class LLMAnalyst:
    name = "llm"

    def __init__(self, model):
        self.parser = PydanticOutputParser(pydantic_object=Judgement)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
        self.chain = prompt.partial(format_instructions=self.parser.get_format_instructions()) | model | self.parser
        self.label = describe(model)

    def run(self, n, clause, hits):
        passages = "\n\n".join(
            f"[{p.id}] {p.citation} - {p.section_title}\n"
            + (f"(lead-in: {p.context})\n" if p.context else "") + p.text for p, _ in hits)
        try:
            j = self.chain.invoke({"n": n, "clause": clause, "passages": passages})
        except Exception as exc:               # malformed output or API error: never guess
            return Judgement(verdict="needs_review", rationale=f"Analyst output could not be used ({type(exc).__name__}).")
        return j


class BaselineAnalyst:
    """No-LLM fallback. It cannot judge compliance; it can only notice that a
    clause uses waiver/negation language near a passage that imposes a duty.
    Everything it raises is 'needs_review', never 'non_compliant'."""
    name = "baseline"
    label = describe(None)
    CUES = re.compile(
        r"\b(no|not|never|without|waive[sd]?|skip(?:ped|s)?|omit(?:ted|s)?|optional|deferred?|"
        r"instead of|rather than|in lieu of|only if|unless|need not|no longer|discard(?:ed)?|"
        r"destroy(?:ed)?|overwrit\w+|share[sd]?|informal(?:ly)?|verbal(?:ly)?|at the discretion|"
        r"if budget allows|when (?:they|he|she) (?:judge|think)s?)\b", re.I)
    DUTY = re.compile(r"\b(shall|must|required)\b", re.I)

    def run(self, n, clause, hits):
        if not hits:
            return Judgement(verdict="not_addressed", rationale="No relevant passage retrieved.")
        cue = self.CUES.search(clause)
        best, sentence = None, None
        ctoks = set(tokenize(clause))
        for p, _ in hits:
            for s in split_sentences(p.text):
                if not self.DUTY.search(s):
                    continue
                overlap = len(ctoks & set(tokenize(s)))
                if best is None or overlap > best[0]:
                    best, sentence = (overlap, p), s
        if best is None or best[0] < 2:
            return Judgement(verdict="not_addressed", rationale="No retrieved requirement overlaps this clause.")
        p = best[1]
        if cue:
            return Judgement(verdict="needs_review", passage_id=p.id, quote=sentence,
                             rationale=f"The clause uses limiting language (\"{cue.group(0)}\") next to a stated requirement. "
                                       "A reviewer should confirm whether the requirement is still met.",
                             recommendation="Have a qualified reviewer compare this clause with the cited requirement.")
        return Judgement(verdict="no_flag", passage_id=p.id, quote=sentence,
                         rationale="No waiver or negation language found near the closest requirement. "
                                   "This is a keyword check, not a finding of compliance.")


def split_sentences(text):
    text = re.sub(r"^\(\w{1,5}\)\s+", "", text.strip())
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text) if len(s.strip()) >= config.MIN_QUOTE_CHARS]


# ---------------------------------------------------------------- 4. verifier
def _norm(s):
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().strip('"').strip().lower()


class CitationVerifier:
    """A finding may only cite a passage that was actually retrieved, and its
    quote must appear word for word in that passage. Otherwise the citation is
    stripped and a non-compliance call is downgraded to needs_review."""

    def run(self, finding: Finding, hits):
        by_id = {p.id: p for p, _ in hits}
        if finding.verdict == "not_addressed":
            finding.citation = finding.passage_id = finding.quote = None
            return finding
        problem = None
        p = by_id.get((finding.passage_id or "").strip().strip("[]"))
        if p is None:
            problem = "cited a passage that was not among those retrieved"
        elif not finding.quote or len(_norm(finding.quote)) < config.MIN_QUOTE_CHARS:
            problem = "gave no usable quote"
        elif _norm(finding.quote).rstrip(".;:") not in _norm(f"{p.context} {p.text}"):
            problem = "quote does not appear in the cited passage"
        if problem:
            finding.verifier_note = f"Citation removed: analyst {problem}."
            finding.citation = finding.passage_id = finding.quote = None
            finding.citation_verified = False
            if finding.verdict in ("non_compliant", "compliant", "no_flag"):
                finding.verdict = "needs_review"
            return finding
        finding.passage_id, finding.citation, finding.citation_verified = p.id, p.citation, True
        return finding


# ---------------------------------------------------------------- 5. report
LABEL = {"non_compliant": "Non-compliant", "needs_review": "Needs review",
         "compliant": "Compliant", "no_flag": "No flag raised (keyword check only)",
         "not_addressed": "Not addressed by corpus"}


class ReportWriter:
    def run(self, findings, analyst_label, title="Proposal"):
        counts = {k: sum(f.verdict == k for f in findings) for k in LABEL}
        lines = [f"# Compliance audit: {title}", "",
                 f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Analyst: {analyst_label}.",
                 "Scope: 21 CFR Parts 11, 210 and 211. This is a screening aid, not legal advice.", "",
                 "## Summary", "",
                 f"- Clauses reviewed: {len(findings)}"] + [f"- {LABEL[k]}: {v}" for k, v in counts.items() if v]
        for key in ("non_compliant", "needs_review"):
            group = [f for f in findings if f.verdict == key]
            if not group:
                continue
            lines += ["", f"## {LABEL[key]}", ""]
            for f in group:
                lines += [f"### Clause {f.clause_no}", "", f"> {f.clause}", ""]
                if f.citation:
                    lines += [f"**{f.citation}**: \"{f.quote}\"", ""]
                if f.rationale:
                    lines += [f.rationale, ""]
                if f.recommendation:
                    lines += [f"Recommendation: {f.recommendation}", ""]
                if f.verifier_note:
                    lines += [f"_{f.verifier_note}_", ""]
        for key in ("compliant", "no_flag"):
            ok = [f for f in findings if f.verdict == key]
            if ok:
                lines += ["", f"## {LABEL[key]}", ""] + [f"- Clause {f.clause_no}: closest requirement {f.citation}" for f in ok]
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- orchestrator
def run_audit(text, retriever, model=None, title="Proposal"):
    analyst = LLMAnalyst(model) if model is not None else BaselineAnalyst()
    retrieve, verify = RetrievalAgent(retriever), CitationVerifier()
    findings = []
    for n, clause in enumerate(ClauseExtractor().run(text), start=1):
        hits = retrieve.run(clause)
        j = analyst.run(n, clause, hits)
        f = Finding(clause_no=n, clause=clause, verdict=j.verdict, passage_id=j.passage_id,
                    quote=j.quote, rationale=j.rationale, recommendation=j.recommendation,
                    retrieved=[{"id": p.id, "citation": p.citation, "score": round(s, 4)} for p, s in hits])
        findings.append(verify.run(f, hits))
    counts = {k: sum(f.verdict == k for f in findings) for k in LABEL}
    return {"title": title, "analyst": analyst.label, "analyst_kind": analyst.name,
            "counts": counts, "findings": [asdict(f) for f in findings],
            "citations_removed": sum(bool(f.verifier_note) for f in findings),
            "report_markdown": ReportWriter().run(findings, analyst.label, title)}
