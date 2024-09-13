
"""
Author: Zhengyuan Dong
Created Date: 2024-02-01
Last Edited Date: 2024-08-27
Description: 
    Plot the comparison of the retriever accuracy results for different libraries and models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# Load the CSV data
df = pd.read_csv('output/retriever_accuracy_results.csv')

# Set up the default Seaborn style
#sns.set(style="white") # grid

# Define the colors and labels
color_palette = ['#3a4068', '#3b6d94', '#45979c', '#71c6ab']
#color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # color1
legend_labels = ['Synthetic instruction', 'Annotated instruction',
                 'Synthetic instruction w/ ambiguity removal', 'Annotated instruction w/ ambiguity removal']

rename_map = {
    'scanpy': 'Scanpy',
    'squidpy': 'Squidpy',
    'ehrapy': 'Ehrapy',
    'snapatac2': 'SnapATAC2'
}
# Define the function for plotting
def create_subplot_bar_graphs(df, ax, lib, colors, display_test=False):
    x_labels = [
        'BM25',
        'SENTENCE-BERT \nw/o \nfine-tuning',
        'SENTENCE-BERT \nw/ \nfine-tuning'
    ]
    indices = np.arange(len(x_labels))
    bar_width = 0.2  # Adjusted bar width to ensure no gaps between bars

    # Position the bars for each condition in the middle of the category
    for i, col_prefix in enumerate(['BM25', 'Un-Finetuned', 'Finetuned']):
        val_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Val'].values[0]
        test_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Test'].values[0]
        val_am_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Ambiguous Val'].values[0]
        test_am_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Ambiguous Test'].values[0]

        # Shift bars to align correctly without gaps
        ax.bar(indices[i] - bar_width, val_acc, bar_width, color=colors[0])
        if display_test:
            ax.bar(indices[i], test_acc, bar_width, color=colors[1])
        ax.bar(indices[i] + bar_width, val_am_acc, bar_width, color=colors[2])
        if display_test:
            ax.bar(indices[i] + 2 * bar_width, test_am_acc, bar_width, color=colors[3])

        ax.text(indices[i] - bar_width, val_acc, f'{val_acc:.2f}', ha='center', va='bottom', fontsize=6.5)
        if display_test:
            ax.text(indices[i], test_acc, f'{test_acc:.2f}', ha='center', va='bottom', fontsize=6.5)
        ax.text(indices[i] + bar_width, val_am_acc, f'{val_am_acc:.2f}', ha='center', va='bottom', fontsize=6.5)
        if display_test:
            ax.text(indices[i] + 2 * bar_width, test_am_acc, f'{test_am_acc:.2f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_title(f'{rename_map.get(lib, lib)}', fontsize=12)
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, fontsize=8)
    #ax.set_ylabel('Prediction Accuracy', fontsize=10)

libs = df['LIB'].unique()

# Set figsize with each subplot as a square
fig, axs = plt.subplots(2, 2, figsize=(8,9), sharey=False)

# Flatten the axs array for easy iteration
axs = axs.ravel()

# Generate subplots
for i, lib in enumerate(libs):
    create_subplot_bar_graphs(df, axs[i], lib, color_palette, display_test=True)

# Remove empty subplots if any
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Add a global legend at the bottom, aligned horizontally
handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_palette[:len(legend_labels)]]
fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2)

fig.text(0.01, 0.5, 'Prediction Accuracy', va='center', ha='center', rotation='vertical', fontsize=12)

# Adjust the space between subplots
plt.subplots_adjust(hspace=0.35, wspace=0.35)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig('output/retriever_performance_compare.pdf')
plt.show()