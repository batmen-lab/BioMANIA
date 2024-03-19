import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('plot/retriever_topk_results.csv')

columns_to_keep = ['retrieved_api_nums', 'Validation Accuracy', 'Test Accuracy', 'val ambiguous Accuracy', 'test ambiguous Accuracy'] # 'Training Accuracy', 'Training ambiguous Accuracy', 

labels = [
    'Synthetic instruction', 'Annotation instruction',
    'Synthetic instruction w/ ambiguity removal', 'Annotation instruction w/ ambiguity removal'
]

num_rows = 2
num_cols = 2

libs = df['LIB'].unique()[:num_rows * num_cols]

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
plt.subplots_adjust(bottom=0.25)  # 调整底部空间

for index, lib in enumerate(libs):
    ax = axs.flatten()[index]
    lib_df = df[df['LIB'] == lib][columns_to_keep]
    for col, label in zip(columns_to_keep[1:], labels):
        ax.plot(lib_df['retrieved_api_nums'], lib_df[col], label=label)
    ax.set_title(f'Fine-tuned Retriever Accuracy v.s. Topk for {lib}')
    ax.set_xlabel('Topk')
    ax.set_ylabel('Accuracy')

# 用flatten()将axs从2D数组转化为1D数组以便简单索引
handles, labels = axs.flatten()[0].get_legend_handles_labels()

# 在整个Figure的底部绘制一个共享的图例
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4)

# Tighten the layout with the rect parameter to fit the new legend
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('./plot/retriever_acc_lib_4.jpg')
plt.show()

