from collections import Counter 
import pandas as pd 
 
def parse_to_diagrams(sentences: list[str]):   
    from lambeq import BobcatParser
    parser = BobcatParser(verbose="suppress")
    results, failed = [], []
    for sent in sentences:
        try:
            results.append((sent, parser.sentence2diagram(sent)))
        except Exception as e:  
            failed.append((sent, str(e)))
 
    print(f"Parsed: {len(results)} / {len(sentences)}  (failed: {len(failed)})")
    return results, failed
 
def build_type_inventory(diagrams: list) -> pd.Series:
    counts = Counter()
    for _, diagram in diagrams:
        for box in diagram.boxes:
            counts[str(box.cod)] += 1
    return pd.Series(counts).sort_values(ascending=False)
 
