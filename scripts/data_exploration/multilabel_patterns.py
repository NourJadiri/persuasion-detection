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

def compute_exact_span_cooccurrence_matrix(df):
    """
    Computes a co-occurrence matrix for labels assigned to the exact same span (same doc_id, start, end).
    Expects a pandas DataFrame with columns: ['doc_id', 'label', 'start', 'end'].
    Returns a pandas DataFrame with the co-occurrence counts.
    """
    all_labels = sorted(set(df['label']))
    co_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)

    # Group by exact span
    grouped = df.groupby(['doc_id', 'start', 'end'])
    for _, group in grouped:
        labels = group['label'].unique()
        if len(labels) > 1:
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    co_matrix.loc[labels[i], labels[j]] += 1
                    co_matrix.loc[labels[j], labels[i]] += 1  # symmetric
    return co_matrix

def plot_cooccurrence_matrix(co_matrix, title='Co-occurrence Matrix', ax=None):
    """
    Plots the co-occurrence matrix using seaborn heatmap with red shades only.
    If ax is provided, plot on that axes; otherwise, create a new figure.
    """
    if ax is None:
        plt.figure(figsize=(16, 14))
        ax = plt.gca()
    sns.heatmap(
        co_matrix,
        annot=True,
        fmt='d',
        cmap='Reds',
        cbar=True,
        annot_kws={"size": 10},
        ax=ax
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Labels', fontsize=14)
    ax.set_ylabel('Labels', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    if ax is None:
        plt.show()
