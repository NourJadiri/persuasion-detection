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
    Plots the co-occurrence matrix using seaborn heatmap with red shades only.
    """
    plt.figure(figsize=(16, 14))  # Increased figure size
    sns.heatmap(
        co_matrix,
        annot=True,
        fmt='d',
        cmap='Reds',  # Red shades only
        cbar=True,
        annot_kws={"size": 10}
    )
    plt.title('Co-occurrence Matrix', fontsize=18)
    plt.xlabel('Labels', fontsize=14)
    plt.ylabel('Labels', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()
