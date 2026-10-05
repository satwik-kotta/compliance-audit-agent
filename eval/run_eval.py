"""Score the pipeline against 36 hand-labelled clauses (eval/gold.json).

    python -m eval.run_eval            # retrieval + whichever analyst is configured

Retrieval and the baseline analyst need no API key. With ANTHROPIC_API_KEY or
OPENAI_API_KEY set, the LLM analyst is scored on the same clauses.
"""
import json
from pathlib import Path

from backend.app import config
from backend.app.agents import run_audit
from backend.app.llm import describe, get_chat_model
from backend.app.retrieval import get_retriever

HERE = Path(__file__).parent


def section_of(pid):
    return pid.split("(")[0] if pid else None


def retrieval_scores(gold, retriever):
    out = {}
    for mode in ("bm25", "dense", "hybrid"):
        ranks = []
        for g in gold:
            secs = [section_of(p.id) for p, _ in retriever.search(g["clause"], k=10, mode=mode)]
            ranks.append(next((i + 1 for i, s in enumerate(secs) if s in g["gold_sections"]), None))
        n = len(ranks)
        out[mode] = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 3) for k in (1, 3, 5, 10)}
        out[mode]["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    return out


def analyst_scores(gold, retriever, model):
    rows = []
    for slug in dict.fromkeys(g["proposal"] for g in gold):
        text = (config.PROPOSAL_DIR / f"{slug}.txt").read_text()
        result = run_audit(text, retriever, model=model, title=slug)
        for g, f in zip([g for g in gold if g["proposal"] == slug], result["findings"]):
            assert g["clause"] == f["clause"]
            rows.append({**g, "verdict": f["verdict"], "cited_section": section_of(f["passage_id"]),
                         "verified": f["citation_verified"], "note": f["verifier_note"]})
    flagged = lambda r: r["verdict"] in ("non_compliant", "needs_review")
    bad = [r for r in rows if r["gold"] == "non_compliant"]
    good = [r for r in rows if r["gold"] == "compliant"]
    tp, fp = sum(map(flagged, bad)), sum(map(flagged, good))
    cited = [r for r in rows if r["cited_section"]]
    flagged_bad = [r for r in bad if flagged(r)]
    return {
        "analyst": describe(model), "clauses": len(rows),
        "violations": len(bad), "violations_flagged": tp,
        "recall": round(tp / len(bad), 3),
        "compliant_clauses": len(good), "false_alarms": fp,
        "precision": round(tp / max(tp + fp, 1), 3),
        "called_non_compliant": sum(r["verdict"] == "non_compliant" for r in rows),
        "citations_given": len(cited),
        "citation_section_correct": round(sum(r["cited_section"] in r["gold_sections"] for r in cited) / max(len(cited), 1), 3),
        "flagged_violations_with_correct_section": f"{sum(r['cited_section'] in r['gold_sections'] for r in flagged_bad)}/{len(flagged_bad)}",
        "citations_removed_by_verifier": sum(bool(r["note"]) for r in rows),
    }, rows


def main():
    gold = json.loads((HERE / "gold.json").read_text())
    retriever = get_retriever()
    results = {"passages": len(retriever.passages), "retrieval": retrieval_scores(gold, retriever)}
    results["baseline"], rows = analyst_scores(gold, retriever, None)
    model = get_chat_model()
    if model is not None:
        results["llm"], rows = analyst_scores(gold, retriever, model)
        (HERE / "llm_rows.json").write_text(json.dumps(rows, indent=1))
    else:
        results["llm"] = "not run: no API key in this environment"
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
