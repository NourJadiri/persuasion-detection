import re

def print_span(article_number, start_offset, end_offset, lang="en", base_path="data/raw"):
    article_path = f"{base_path}/{lang}/train-articles-subtask-3/article{article_number}.txt"
    with open(article_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    span = text[start_offset:end_offset]
    print(span)

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
        with open(output_path, "wb") as out_f:
            out_f.write(wrapped_text.encode("utf-8"))
            
def map_new_spans(filename, text, target_folder, origin_folder):
    src_lang = filename.split("_")[0]
    # Extract the article number from the filename (e.g., fr_article1111.txt -> 1111)
    match = re.search(r'article(\d+)\.txt', filename)
    if match:
        article_number = match.group(1)
    else:
        raise ValueError(f"Could not extract article number from filename: {filename}")
    
    