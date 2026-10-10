"""Convert chess comments into lambeq diagrams.

The Bobcat model is used when it is already available locally.  Some older
lambeq releases check a retired model-version endpoint during initialisation;
the small patch below keeps an existing local model usable offline.  It does
not download a missing model.
"""

from collections import Counter

import pandas as pd
import lambeq.text2diagram.model_based_reader.model_downloader as md
from lambeq import BobcatParser, cups_reader


# Avoid the retired qnlp.cambridgequantum.com version endpoint.  This is kept
# local to the parsing module and only affects lambeq's version lookup.
md.ModelDownloader.get_latest_remote_version = lambda self, *args, **kwargs: "0.1.0"


def get_working_reader():
    """Return a locally cached Bobcat parser, or a no-download reader."""
    try:
        return BobcatParser(force_download=False, verbose="suppress")
    except Exception:
        # ``LinearReader`` needs a combining diagram in lambeq 0.5, so do
        # not instantiate it directly. ``cups_reader`` is a ready-made,
        # local word-sequence reader and does not download a parser model.
        return cups_reader


def parse_to_diagrams(comments: list[str]):
    """Parse comments and preserve their text for dataset alignment.

    Returns ``[(comment, diagram), ...]`` plus failures, matching the input
    contract required by ``dataset_builder.assemble_triples``.
    """
    reader = get_working_reader()
    diagrams, failed = [], []

    for index, comment in enumerate(comments):
        try:
            diagrams.append((comment, reader.sentence2diagram(comment)))
        except Exception as error:
            failed.append((index, comment, str(error)))

    return diagrams, failed


def build_type_inventory(diagrams: list) -> pd.Series:
    """Count diagram output types for inspection before circuit compilation."""
    counts = Counter()
    for item in diagrams:
        diagram = item[1] if isinstance(item, tuple) else item
        for box in diagram.boxes:
            counts[str(box.cod)] += 1
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)
