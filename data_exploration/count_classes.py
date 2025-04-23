import os
import pandas as pd
from collections import Counter

def find_label_files(root_dir):
    label_files = []
    for lang in os.listdir(root_dir):
        lang_path = os.path.join(root_dir, lang)
        if os.path.isdir(lang_path):
            file_path = os.path.join(lang_path, "train-labels-subtask-3-spans.txt")
            if os.path.isfile(file_path):
                label_files.append(file_path)
    return label_files

def count_classes(label_files):
    class_counter = Counter()
    for file in label_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    class_counter[parts[1]] += 1
    return class_counter

if __name__ == "__main__":
    root = "data/raw"
    label_files = find_label_files(root)
    class_counts = count_classes(label_files)
    df = pd.DataFrame(class_counts.items(), columns=["Class", "Count"]).sort_values("Count", ascending=False)
    print(df)