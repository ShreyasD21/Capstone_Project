"""Parse comments into lambeq string diagrams, with failures recorded, never hidden.

Two parsers are supported, chosen in the config (`text.parser`):

grammar  the project's hand-written pregroup grammar (text/grammar.py). Default.
bobcat   lambeq's Bobcat parser, loaded from a local model folder (`text.bobcat_dir`).
         lambeq's download server no longer exists, so a model must be supplied
         by hand, e.g. a copy of someone's ~/.cache/lambeq/bobcat/bobcat.

There is deliberately no fallback between them: if the configured parser is
unavailable the pipeline stops.
"""

import hashlib
import pickle
import re
from collections.abc import Callable
from pathlib import Path

import lambeq
from lambeq import AtomicType
from lambeq.backend.grammar import Diagram

from chess_qnlp.text import grammar


def _bobcat(model_dir: str | Path | None) -> Callable[[str], Diagram]:
    if not model_dir or not Path(model_dir).is_dir():
        raise RuntimeError(
            "text.parser is 'bobcat' but text.bobcat_dir does not point to a Bobcat model folder. "
            "lambeq can no longer download it (qnlp.cambridgequantum.com is gone). "
            "Supply a local copy or use text.parser: grammar."
        )
    parser = lambeq.BobcatParser(model_name_or_path=str(model_dir), verbose="suppress")

    def parse(sentence: str) -> Diagram:
        diagram = parser.sentence2diagram(re.findall(r"[a-z0-9]+", sentence.lower()), tokenised=True)
        if diagram is None:
            raise ValueError("Bobcat returned no parse")
        return diagram

    return parse


def get_parser(text_cfg: dict) -> Callable[[str], Diagram]:
    name = text_cfg.get("parser", "grammar")
    if name == "grammar":
        return grammar.parse
    if name == "bobcat":
        return _bobcat(text_cfg.get("bobcat_dir"))
    raise ValueError(f"Unknown parser {name!r}; use 'grammar' or 'bobcat'")


def parse_all(sentences, text_cfg: dict, cache_dir: Path) -> tuple[dict, dict]:
    """Parse each distinct sentence once. Returns ({sentence: diagram}, {sentence: error})."""
    name = text_cfg.get("parser", "grammar")
    if name == "grammar":  # editing the grammar must invalidate cached parses
        name += "-" + hashlib.sha256(Path(grammar.__file__).read_bytes()).hexdigest()[:10]
    cache_path = cache_dir / f"diagrams_{name}_lambeq-{lambeq.__version__}.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    todo = sorted(set(sentences) - set(cache))
    if todo:
        parse = get_parser(text_cfg)
        for sentence in todo:
            try:
                diagram = parse(sentence)
                if diagram.cod != AtomicType.SENTENCE:
                    raise ValueError(f"parse has type {diagram.cod}, not a sentence")
                cache[sentence] = diagram
            except Exception as e:  # recorded per sentence and reported by the caller
                cache[sentence] = f"{type(e).__name__}: {e}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cache))
    diagrams = {s: cache[s] for s in set(sentences) if not isinstance(cache[s], str)}
    errors = {s: cache[s] for s in set(sentences) if isinstance(cache[s], str)}
    return diagrams, errors
