
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

df = pd.read_csv('./output/retriever_topk_results.csv')
columns_to_keep = ['retrieved_api_nums', 'Validation Accuracy', 'Test Accuracy', 'val ambiguous Accuracy', 'test ambiguous Accuracy']
labels = ['Synthetic instruction', 'Annotation instruction',
          'Synthetic instruction w/ ambiguity removal', 'Annotation instruction w/ ambiguity removal']
libs = df['LIB'].unique()
num_libs = len(libs)
if num_libs <= 2:
    fig, axs = plt.subplots(1, num_libs, figsize=(6 * num_libs, 5), sharey=True)
    if num_libs == 1:
        axs = np.array([axs])
else:
    num_rows = np.ceil(num_libs / 2).astype(int)
    fig, axs = plt.subplots(num_rows, 2, figsize=(6, 5 * num_rows), sharey=True)
    axs = axs.flatten()
for index, lib in enumerate(libs):
    ax = axs[index]
    lib_df = df[df['LIB'] == lib]
    for col, label in zip(columns_to_keep[1:], labels):
        ax.plot(lib_df['retrieved_api_nums'], lib_df[col], label=label)
    ax.set_title(f'Fine-tuned Retriever Accuracy v.s. Topk for {lib}')
    ax.set_xlabel('Topk')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
if num_libs > 1:
    for ax in axs[num_libs:]:
        ax.set_visible(False)
handles, labels = axs.flatten()[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.subplots_adjust(hspace=0.4)
plt_path = "./output/step4_analysis_retriever_acc_lib.pdf"
plt.savefig(plt_path)
plt.show()
plt_path