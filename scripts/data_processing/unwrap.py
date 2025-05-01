from collections import defaultdict
import os
import re
from typing import Dict, Iterable, List, Tuple


def build_label_lookup(label_file: str) -> Dict[str, Dict[int, str]]:
    """
    Returns: {article_id: {token_id: label, …}, …}
    """
    lookup: Dict[str, Dict[int, str]] = defaultdict(dict)
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            art, label, *_ = line.rstrip("\n").split("\t")
            token_id = len(lookup[art]) + 1          # 1-based, keeps order
            lookup[art][token_id] = label
    return lookup

_TOKEN_RE = re.compile(r"<<(/?)S_(\d+)>>")

def extract_spans(text: str) -> Iterable[Tuple[int, int, int]]:
    """
    Yields (token_id, start, end) where start/end are offsets **after** tokens
    have been stripped.  Works with nesting and shared boundaries.
    """
    stack: List[Tuple[int, int]] = []          # [(token_id, clean_start), …]
    clean_idx = 0                              # offset in token-free text
    i = 0
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if m:                                  # marker found
            closing, tok = m.groups()
            token_id = int(tok)
            if closing:                        # </S_k>
                # pop matching opener
                for j in range(len(stack) - 1, -1, -1):
                    if stack[j][0] == token_id:
                        start = stack[j][1]
                        yield token_id, start, clean_idx
                        stack.pop(j)
                        break
            else:                              # <S_k>
                stack.append((token_id, clean_idx))
            i += m.end() - m.start()           # skip marker
        else:                                  # normal char
            clean_idx += 1
            i += 1
            
def remap_article(
    filepath: str,
    lookup: Dict[str, Dict[int, str]],
    error_log: List[str],
) -> List[str]:
    """
    Returns the remapped label lines for this file.
    Missing labels are logged and skipped.
    """
    # id is the digits after “article” in the filename
    m = re.search(r'article(\d+)\.txt$', os.path.basename(filepath))
    if not m:
        error_log.append(f"Cannot parse article id from {filepath}")
        return []
    art_id = m.group(1)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    lines = []
    for tok_id, start, end in extract_spans(text):
        label = lookup.get(art_id, {}).get(tok_id)
        if label is None:
            error_log.append(
                f"[{art_id}] token S_{tok_id} has no label – skipped")
            continue
        lines.append(f"{art_id}\t{label}\t{start}\t{end}")
    return lines


def remap_folder(
    translated_dir: str,
    src_lang: str,
    lookup_file: str,
    out_label_file: str,
    error_file: str = "remap_errors.log",
):
    lookup = build_label_lookup(lookup_file)
    errors: List[str] = []
    output_lines: List[str] = []

    for name in os.listdir(translated_dir):
        if not name.startswith(f"{src_lang}_article") or not name.endswith(".txt"):
            continue
        path = os.path.join(translated_dir, name)
        output_lines.extend(remap_article(path, lookup, errors))

    # write new label file
    with open(out_label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    # write errors (if any)
    if errors:
        with open(error_file, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))
        print(f"Finished with {len(errors)} issue(s) – see {error_file}")
    else:
        print("Finished with no errors.")

# remap_folder(
#     translated_dir="data/processed/ru/wrapped-articles",
#     src_lang="en",                    # prefix in filenames
#     lookup_file="data/processed/en/train-labels-subtask-3-spans.txt",
#     out_label_file="data/processed/ru/train-labels-subtask-3-spans-en.txt",
# )

def unwrap_articles(lang_dir: str):
    """
    Removes all <<S_NUMBER>> and <</S_NUMBER>> tokens from articles in lang_dir,
    and saves the cleaned articles to a new 'unwrapped-articles' folder.
    Shows a simple progression indicator.
    """
    base_dir = '../data/processed'
    in_dir = os.path.join(base_dir, lang_dir, "wrapped-articles")
    out_dir = os.path.join(base_dir, lang_dir, "unwrapped-articles")
    os.makedirs(out_dir, exist_ok=True)
    token_re = re.compile(r"<<\/?S_\d+>>")
    files = [fname for fname in os.listdir(in_dir) if fname.endswith(".txt")]
    total = len(files)
    for idx, fname in enumerate(files, 1):
        with open(os.path.join(in_dir, fname), encoding="utf-8") as fin:
            text = fin.read()
        cleaned = token_re.sub("", text)
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fout:
            fout.write(cleaned)
        print(f"\rProcessing {idx}/{total} files...", end="", flush=True)
    print("\nDone.")
