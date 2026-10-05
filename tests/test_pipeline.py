import json

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.app.agents import CitationVerifier, ClauseExtractor, Finding, run_audit
from backend.app.corpus import load_corpus
from backend.app.main import app
from backend.app.retrieval import get_retriever

CLAUSE = ("Batch production records will be kept for six months after the batch is released "
          "and then destroyed to reduce storage costs.")
REAL_QUOTE = ("Any production, control, or distribution record that is required to be maintained in compliance "
              "with this part and is specifically associated with a batch of a drug product shall be retained "
              "for at least 1 year after the expiration date of the batch")


def by_id(pid):
    return next(p for p in load_corpus() if p.id == pid)


def test_corpus_citations_are_unique_and_verbatim():
    corpus = load_corpus()
    assert len(corpus) > 250 and {p.part for p in corpus} == {"11", "210", "211"}
    p = by_id("211.84(d)(2)")
    assert p.citation == "21 CFR 211.84(d)(2)"
    assert "at least one specific identity test is conducted" in p.text
    assert p.context.startswith("(d) Samples shall be examined")
    assert by_id("11.10(e)").text.startswith("(e) Use of secure, computer-generated, time-stamped audit trails")
    assert not any("FR " in p.text and p.text.startswith("\\[") for p in corpus)   # editorial notes stripped


def test_letter_i_is_a_paragraph_not_a_roman_numeral():
    assert by_id("11.10(i)").text.startswith("(i) Determination that persons who develop")


def test_extractor_handles_numbered_and_prose():
    numbered = "Title\n\nPreamble that is long enough to look like a clause but is not one.\n\n" + \
               "\n".join(f"{i}. Commitment number {i} is written out here as a full sentence." for i in range(1, 5))
    assert len(ClauseExtractor().run(numbered)) == 4
    prose = "We will keep records for six months and then destroy them all.\n\nTraining is optional and offered once per year only."
    assert len(ClauseExtractor().run(prose)) == 2


def test_retrieval_finds_the_governing_section():
    hits = get_retriever().search(CLAUSE, k=5)
    assert "211.180" in [p.section for p, _ in hits]


def _audit_with(reply):
    model = FakeListChatModel(responses=[json.dumps(reply)])
    return run_audit("1. " + CLAUSE, get_retriever(), model=model)["findings"][0]


def test_llm_finding_with_real_quote_is_kept():
    f = _audit_with({"verdict": "non_compliant", "passage_id": "211.180(a)", "quote": REAL_QUOTE,
                     "rationale": "Six months is shorter than required.", "recommendation": "Retain for 1 year past expiry."})
    assert f["verdict"] == "non_compliant" and f["citation"] == "21 CFR 211.180(a)" and f["citation_verified"]


def test_fabricated_quote_is_stripped_and_verdict_downgraded():
    f = _audit_with({"verdict": "non_compliant", "passage_id": "211.180(a)",
                     "quote": "Batch records shall be retained for a minimum of seven years after release.",
                     "rationale": "x", "recommendation": "y"})
    assert f["verdict"] == "needs_review" and f["citation"] is None and "quote does not appear" in f["verifier_note"]


def test_citation_outside_retrieved_set_is_stripped():
    f = _audit_with({"verdict": "non_compliant", "passage_id": "312.62(c)", "quote": REAL_QUOTE,
                     "rationale": "x", "recommendation": "y"})
    assert f["verdict"] == "needs_review" and f["citation"] is None and "not among those retrieved" in f["verifier_note"]


def test_malformed_llm_output_never_becomes_a_finding():
    model = FakeListChatModel(responses=["Sure! This clause looks fine to me."])
    f = run_audit("1. " + CLAUSE, get_retriever(), model=model)["findings"][0]
    assert f["verdict"] == "needs_review" and f["citation"] is None


def test_verifier_tolerates_whitespace_and_quote_marks():
    hits = [(by_id("211.180(a)"), 1.0)]
    f = Finding(1, CLAUSE, "non_compliant", passage_id="[211.180(a)]", quote='"' + REAL_QUOTE.replace(" ", "  ", 3) + '"')
    assert CitationVerifier().run(f, hits).citation_verified


def test_api_end_to_end_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    assert client.get("/api/health").json()["passages"] > 250
    samples = client.get("/api/samples").json()
    assert len(samples) == 3
    r = client.post("/api/audit", json={"text": samples[0]["text"], "title": samples[0]["title"]})
    body = r.json()
    assert r.status_code == 200 and len(body["findings"]) == 12 and body["analyst_kind"] == "baseline"
    assert body["counts"]["non_compliant"] == 0          # the baseline never asserts non-compliance
    assert all(f["citation_verified"] for f in body["findings"] if f["citation"])
    assert client.get("/api/passages/211.192").json()["citation"] == "21 CFR 211.192"
    assert client.post("/api/audit", json={"text": "too short"}).status_code == 422
