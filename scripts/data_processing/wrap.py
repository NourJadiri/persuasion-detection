import re

def remap_spans_in_wrapped_article(wrapped_text, token_label_mapping):
    """
    Given a wrapped article and the token-label mapping (from get_token_label_mapping_from_labels_file),
    returns a list of dicts with token, label, new_start, new_end (span of the inner text, not including tokens).

    Args:
        wrapped_text (str): The text of the wrapped (possibly translated) article.
        token_label_mapping (list): List of dicts for this article, each with 'token' and 'label'.

    Returns:
        List[dict]: Each dict has 'token', 'label', 'new_start', 'new_end'.
    """
    # Build a mapping from token to label
    token_to_label = {d['token']: d['label'] for d in token_label_mapping}
    results = []
    # Regex to find all <<S_n>>...<</S_n>>
    pattern = re.compile(r"<<(?P<token>S_\d+)>>(.*?)<</\\1>>", re.DOTALL)
    for match in pattern.finditer(wrapped_text):
        token = match.group('token')
        inner_text = match.group(2)
        # The span of the inner text (excluding tokens)
        # match.start(2) and match.end(2) are the offsets of the inner text
        new_start = match.start(2)
        new_end = match.end(2)
        label = token_to_label.get(token, None)
        results.append({
            'token': token,
            'label': label,
            'new_start': new_start,
            'new_end': new_end
        })
    return results

def print_span(article_number, start_offset, end_offset, lang="en", base_path="data/raw"):
    article_path = f"{base_path}/{lang}/train-articles-subtask-3/article{article_number}.txt"
    with open(article_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    span = text[start_offset:end_offset]
    print(span)

import re
import os

def clean_existing_tokens(text):
    # Remove anything like <<S_23>> or <</S_23>>
    return re.sub(r"<<\/?S_\d+>>", "", text)

def wrap_annotated_spans(
    article_path: str,
    spans: list[tuple[int, int]],
    token_prefix: str = "S",
) -> str:
    """Wrap every (start,end) pair in <<S_n>> … <</S_n>> without breaking when
    spans share boundaries or nest.
    """
    with open(article_path, "rb") as f:
        text = clean_existing_tokens(f.read().decode("utf-8", "ignore"))

    text_len = len(text)
    # keep only valid spans and remember their original index (for the S_n id)
    valid = [(s, e, i + 1)
             for i, (s, e) in enumerate(spans)
             if 0 <= s < e <= text_len]
    if not valid:
        return text

    # build “open” and “close” event tables
    opens, closes = {}, {}
    for s, e, idx in valid:
        # we store span length too, used only for sorting at the same position
        length = e - s
        opens.setdefault(s, []).append((length, idx))
        closes.setdefault(e, []).append((length, idx))

    # enforce the right order at identical positions
    for pos in opens:
        # longer spans open first → proper nesting
        opens[pos].sort(key=lambda x: -x[0])
    for pos in closes:
        # shorter spans close first → proper nesting
        closes[pos].sort(key=lambda x: x[0])

    # single left-to-right pass
    out = []
    for i in range(text_len + 1):
        if i in closes:
            for _, idx in closes[i]:
                out.append(f"<</{token_prefix}_{idx}>>")
        if i in opens:
            for _, idx in opens[i]:
                out.append(f"<<{token_prefix}_{idx}>>")
        if i < text_len:
            out.append(text[i])

    return "".join(out)

def wrap_spans_from_file(labels_file, articles_folder, output_folder, lang="en", token_prefix="S"):
    """
    Reads a span annotation file and wraps the spans in the corresponding article files.
    Args:
        labels_file (str): Path to the span annotation file (e.g., train-labels-subtask-3-spans.txt).
        articles_folder (str): Path to the folder containing article text files.
        output_folder (str): Path to the folder to write wrapped articles.
        lang (str): Language code (default: 'en').
        token_prefix (str): Prefix for the span tokens (default: 'S').
    """
    import os
    from collections import defaultdict
    
    # Collect spans per article
    article_spans = defaultdict(list)
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            article_id, label, start, end = parts[:4]
            article_spans[article_id].append((int(start), int(end)))

    os.makedirs(output_folder, exist_ok=True)

    # For each article, wrap spans and write output
    for article_id, spans in article_spans.items():
        article_path = os.path.join(articles_folder, f"article{article_id}.txt")
        if not os.path.exists(article_path):
            print(f"Warning: Article file not found: {article_path}")
            continue
        wrapped_text = wrap_annotated_spans(article_path, spans, token_prefix=token_prefix)
        output_path = os.path.join(output_folder, f"article{article_id}.txt")
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(wrapped_text)

def get_token_label_mapping_from_labels_file(labels_file):
    """
    Reads a span annotation file and returns a mapping:
    {
        article_id: [
            {"token": "S_1", "label": label, "orig_start": start, "orig_end": end},
            ...
        ],
        ...
    }
    The order of tokens matches the reverse order used in wrap_annotated_spans.
    """
    from collections import defaultdict
    mapping = defaultdict(list)
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            article_id, label, start, end = parts[:4]
            mapping[article_id].append({
                "label": label,
                "orig_start": int(start),
                "orig_end": int(end)
            })
    # Now, for each article, sort spans in reverse order and assign token numbers
    for article_id, spans in mapping.items():
        spans_sorted = sorted(spans, key=lambda x: x["orig_start"], reverse=True)
        for idx, span in enumerate(spans_sorted, 1):
            span["token"] = f"S_{idx}"
        mapping[article_id] = spans_sorted
    return mapping

def write_remapped_spans_to_file(remapped_spans, output_file):
    """
    Writes the remapped spans to a file, one per line:
    token<TAB>label<TAB>new_start<TAB>new_end
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for span in remapped_spans:
            f.write(f"{span['token']}\t{span['label']}\t{span['new_start']}\t{span['new_end']}\n")

if __name__ == "__main__":
    # wrap articles in the russian folder
    wrap_spans_from_file(
        labels_file="data/raw/fr/train-labels-subtask-3-spans.txt",
        articles_folder="data/raw/fr/train-articles-subtask-3",
        output_folder="data/processed/fr/wrapped-articles",
        lang="fr",
        token_prefix="S"
    )