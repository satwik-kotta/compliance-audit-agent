"""Settings, all overridable from the environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG_DIR = ROOT / "data" / "regulations"
PROPOSAL_DIR = ROOT / "data" / "proposals"
FRONTEND_DIST = ROOT / "frontend" / "dist"

TOP_K = int(os.environ.get("TOP_K", "5"))            # passages shown to the analyst per clause
LSA_DIM = 128                                        # dense vector size
RRF_K = 60                                           # reciprocal-rank-fusion constant
MIN_QUOTE_CHARS = 25                                 # a citation needs a real quote, not two words
MAX_CLAUSES = int(os.environ.get("MAX_CLAUSES", "60"))
MAX_PROPOSAL_CHARS = 40_000

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
