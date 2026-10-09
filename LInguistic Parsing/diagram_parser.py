from collections import Counter

import pandas as pd


def parse_to_diagrams(sentences: list[str]):
    """
    Parses each UNIQUE sentence once (template data repeats the same sentence
    hundreds of times, so this is much faster). Returns (sentence, diagram)
    pairs; assemble_triples looks diagrams up by sentence.
    """
    from lambeq import BobcatParser

    unique = list(dict.fromkeys(sentences))
    parser = BobcatParser(verbose="suppress")
    results, failed = [], []
    for sent in unique:
        try:
            results.append((sent, parser.sentence2diagram(sent)))
        except Exception as e:  # parsing can fail on odd sentences
            failed.append((sent, str(e)))

    print(f"Parsed: {len(results)} / {len(unique)} unique sentences "
          f"(failed: {len(failed)})")
    for sent, err in failed[:5]:
        print("  failed:", sent, "->", err[:80])
    return results, failed


def build_type_inventory(diagrams: list) -> pd.Series:
    counts = Counter()
    for _, diagram in diagrams:
        for box in diagram.boxes:
            counts[str(box.cod)] += 1
    return pd.Series(counts).sort_values(ascending=False)
