"""Pre-compute audits of the sample proposals for the static (GitHub Pages) demo.

    python -m backend.app.export_static

The hosted demo has no server, so it replays these results. It always uses the
keyword baseline analyst, never an LLM, so the published page is reproducible
without a key.
"""
import json
from dataclasses import asdict

from . import config
from .agents import run_audit
from .corpus import load_corpus
from .llm import describe
from .main import _samples
from .retrieval import get_retriever


def main():
    corpus = load_corpus()
    samples = _samples()
    audits = {s["id"]: run_audit(s["text"], get_retriever(), model=None, title=s["title"]) for s in samples}
    used = {f["passage_id"] for a in audits.values() for f in a["findings"] if f["passage_id"]}
    data = {
        "health": {"status": "ok", "passages": len(corpus), "sections": len({p.section for p in corpus}),
                   "parts": sorted({p.part for p in corpus}, key=int), "analyst": describe(None)},
        "samples": samples, "audits": audits,
        "passages": {p.id: asdict(p) for p in corpus if p.id in used},
    }
    out = config.ROOT / "frontend" / "public" / "static.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out.relative_to(config.ROOT)}: {len(audits)} audits, {len(used)} passages")


if __name__ == "__main__":
    main()
