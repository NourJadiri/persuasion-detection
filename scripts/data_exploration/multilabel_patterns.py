import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def compute_cooccurrence_matrix(df):
    """
    Computes a co-occurrence matrix for labels in overlapping (including nested) spans.
    Expects a pandas DataFrame with columns: ['doc_id', 'label', 'start', 'end'].
    Returns a pandas DataFrame with the co-occurrence counts.
    """
    all_labels = sorted(set(df['label']))
    co_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)

    for doc_id, group in df.groupby('doc_id'):
        spans = group[['label', 'start', 'end']].values.tolist()
        n = len(spans)
        for i in range(n):
            label_i, start_i, end_i = spans[i]
            for j in range(i+1, n):
                label_j, start_j, end_j = spans[j]
                # Check for overlap (including nested)
                if not (end_i <= start_j or end_j <= start_i):
                    co_matrix.loc[label_i, label_j] += 1
                    co_matrix.loc[label_j, label_i] += 1  # symmetric
    return co_matrix


def plot_cooccurrence_matrix(co_matrix):
    """
    Plots the co-occurrence matrix using seaborn heatmap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Co-occurrence Matrix')
    plt.xlabel('Labels')
    plt.ylabel('Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
