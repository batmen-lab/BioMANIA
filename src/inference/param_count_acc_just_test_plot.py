"""
Author: Zhengyuan Dong
Created Date: 2024-08-27
Last Edited Date: 2024-08-27
Description: 
    Plot the comparison of the parameters prediction results for different libraries.
"""

from .param_count_acc_just_test import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Rename libraries and process data
rename_map = {
    'scanpy': 'Scanpy',
    'squidpy': 'Squidpy',
    'ehrapy': 'Ehrapy',
    'snapatac2': 'SnapATAC2'
}


def create_bar_plot_for_metrics(metrics_dict):
    data = {
        'Library': [],
        'True Positive': [],
        'False Negative': []
    }

    for lib_name, metrics in metrics_dict.items():
        renamed_lib = rename_map.get(lib_name, lib_name)  # Use the rename map
        data['Library'].append(renamed_lib)
        data['True Positive'].append(metrics['true_positive_ratio'] * 100)
        data['False Negative'].append(metrics['false_positive_ratio_really'] * 100)

    # Reverse the order of the libraries
    data['Library'].reverse()
    data['True Positive'].reverse()
    data['False Negative'].reverse()

    df = pd.DataFrame(data)
    colors = ['#71c6ab', '#45979c', '#3b6d94', '#3a4068']  # Reversed color order

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(7,6))
    plot_grouped_bar_with_hatch(df, ax, colors)

    ax.set_ylabel('Library', fontsize=12)
    fig.text(0.65, 0.04, 'Percentage (%)', ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0.1, 0.05, 1, 0.95])
    plt.savefig('parameters_performance_comparison.pdf')
    plt.show()

def plot_grouped_bar_with_hatch(df, ax, colors):
    y_labels = df['Library']
    bar_width = 0.4
    index = np.arange(len(y_labels))

    bars_fn = ax.barh(index, df['False Negative'], bar_width, color=colors, label='False Negative', hatch='////')
    bars_tp = ax.barh(index + bar_width, df['True Positive'], bar_width, color=colors, label='True Positive')

    for bars in [bars_fn, bars_tp]:
        for bar in bars:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}', va='center', ha='left')

    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(y_labels)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right')

    ax.set_xlim(0, df[['True Positive', 'False Negative']].max().max() * 1.05)

# Continue with the data processing and calling the plot function as before.

LIB_list = ['scanpy', 'squidpy', 'ehrapy', 'snapatac2']
final_metrics_json = {}
for LIB in LIB_list:
    api_data = load_json(f"./data/standard_process/{LIB}/API_init.json")
    uncategorized_item = load_json(f'api_parameters_{LIB}_final_results.json')
    metrics = process_results(uncategorized_item, api_data)
    final_metrics = calculate_final_metrics_confusion(metrics)
    final_metrics_json[rename_map[LIB]] = final_metrics  # Use the renamed library in the metrics dict

create_bar_plot_for_metrics(final_metrics_json)