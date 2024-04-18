
"""
Author: Zhengyuan Dong
Created Date: 2024-02-01
Last Edited Date: 2024-04-08
Description: 
    Plot the comparison of the retriever accuracy results for different libraries and models.
    Automatically adjust the number of subplots based on the number of libraries.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the CSV data
df = pd.read_csv('output/retriever_accuracy_results.csv')
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_b = default_colors[:4]
legend_labels = ['Synthetic instruction', 'Annotated instruction',
                 'Synthetic instruction w/ ambiguity removal', 'Annotated instruction w/ ambiguity removal']
# Define the function for plotting
def create_subplot_bar_graphs(df, ax, lib, title, colors, display_test=False):
    cols = [f'{lib} Val', f'{lib} Test', f'{lib} Ambiguous Val', f'{lib} Ambiguous Test']
    x_labels = ['BM25', 'SENTENCE-BERT \nw/o \nfine-tuning', 'SENTENCE-BERT \nw/ \nfine-tuning']
    bar_width = 0.15
    indices = np.arange(len(x_labels))
    for i, col_prefix in enumerate(['BM25', 'Un-Finetuned', 'Finetuned']):
        val_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Val'].values[0]
        test_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Test'].values[0]
        val_am_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Ambiguous Val'].values[0]
        test_am_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Ambiguous Test'].values[0]
        ax.bar(indices[i] - 3*bar_width/2, val_acc, bar_width, color=colors[0])
        if display_test:
            ax.bar(indices[i] - bar_width/2, test_acc, bar_width, color=colors[1])
        ax.bar(indices[i] + bar_width/2, val_am_acc, bar_width, color=colors[2])
        if display_test:
            ax.bar(indices[i] + 3*bar_width/2, test_am_acc, bar_width, color=colors[3])
        ax.text(indices[i] - 3*bar_width/2, val_acc, f'{val_acc:.2f}', ha='center', va='bottom', fontsize=8)
        if display_test:
            ax.text(indices[i] - bar_width/2, test_acc, f'{test_acc:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(indices[i] + bar_width/2, val_am_acc, f'{val_am_acc:.2f}', ha='center', va='bottom', fontsize=8)
        if display_test:
            ax.text(indices[i] + 3*bar_width/2, test_am_acc, f'{test_am_acc:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(f'{lib} {title}')
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels)
libs = df['LIB'].unique()
num_libs = len(libs)
# Calculate the number of rows and columns for the subplots
num_cols = 2
num_rows = math.ceil(num_libs / num_cols)
if num_libs <= 2:
    # Adjust subplot layout for 1 or 2 libs
    fig, axs = plt.subplots(1, num_libs, figsize=(6 * num_libs, 5), constrained_layout=True, sharey=True)
    if num_libs == 1:
        axs = np.array([axs])
else:
    # Calculate the number of rows and columns for the subplots for more than 2 libs
    num_cols = 2
    num_rows = math.ceil(num_libs / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6, num_rows * 5), constrained_layout=True, sharey=True)
    axs = axs.ravel()
for i, lib in enumerate(libs):
    ax = axs[i]
    create_subplot_bar_graphs(df, axs[i], lib, 'Prediction Accuracy', color_b, display_test=True)
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
# Add a global legend at the bottom
handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_b[:len(legend_labels)]]
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.subplots_adjust(hspace=0.4)  # Adjust this value to make space for the legend
plt.savefig('output/step5_analysis_compare_retriever.pdf')
plt.show()
