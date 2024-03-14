import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Based on the provided data structure and the requirements for subplots, let's create the desired bar graph.

# Load the CSV data
df = pd.read_csv('output/retriever_accuracy_results.csv')

color_b = ['blue', 'orange', 'green', 'red']

# Define the function for plotting
def create_subplot_bar_graphs(df, ax, lib, title, colors, display_test=False):
    # Columns for BM25, Un-Finetuned, and Finetuned (Val and Test)
    cols = [f'{lib} Val', f'{lib} Test', f'{lib} Ambiguous Val', f'{lib} Ambiguous Test']
    
    x_labels = ['BM25', 'SENTENCE-BERT \nw/o \nfine-tuning', 'SENTENCE-BERT \nw/ \nfine-tuning']

    # The width of the bars
    bar_width = 0.15
    
    # Set the positions of the bars
    indices = np.arange(len(x_labels))
    
    # Plot bars
    for i, col_prefix in enumerate(['BM25', 'Un-Finetuned', 'Finetuned']):
        #print('----------', col_prefix)
        val_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Val'].values[0]
        test_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Test'].values[0]
        #if col_prefix!='BM25':
        val_am_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Ambiguous Val'].values[0]
        test_am_acc = df.loc[df['LIB'] == lib, f'{col_prefix} Ambiguous Test'].values[0]
        
        ax.bar(indices[i] - 3*bar_width/2, val_acc, bar_width, label='Synthetic instruction' if i==0 else "", color=colors[0])
        if display_test:
            ax.bar(indices[i] - bar_width/2, test_acc, bar_width, label='Annotated instruction' if i==0 else "", color=colors[1])
        #if col_prefix!='BM25':
        ax.bar(indices[i] + bar_width/2, val_am_acc, bar_width, label='Synthetic Ambiguous instruction' if i==1 else "", color=colors[2])
        if display_test:
            ax.bar(indices[i] + 3*bar_width/2, test_am_acc, bar_width, label='Annotated Ambiguous instruction' if i==1 else "", color=colors[3])
        
        # Add data labels
        ax.text(indices[i] - 3*bar_width/2, val_acc, f'{val_acc:.2f}', ha='center', va='bottom')
        if display_test:
            ax.text(indices[i] - bar_width/2, test_acc, f'{test_acc:.2f}', ha='center', va='bottom')
        #if col_prefix!='BM25':
        ax.text(indices[i] + bar_width/2, val_am_acc, f'{val_am_acc:.2f}', ha='center', va='bottom')
        if display_test:
            ax.text(indices[i] + 3*bar_width/2, test_am_acc, f'{test_am_acc:.2f}', ha='center', va='bottom')

    # Set the title, x-ticks, and labels
    ax.set_title(f'{lib} {title}')
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels)
    
    # Set the legend only for the first subplot
    if lib == df['LIB'].unique()[0]:
        ax.legend()

# Set up the subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()  # Flatten the array of axes for easier indexing

# Plot each LIB as a separate subplot
for i, lib in enumerate(df['LIB'].unique()):
    create_subplot_bar_graphs(df, axs[i], lib, 'Prediction Accuracy', color_b)


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('./output/step5_analysis_compare_retriever.jpg')
plt.show()
