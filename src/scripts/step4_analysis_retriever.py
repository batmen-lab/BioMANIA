import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('plot/retriever_topk_results.csv')

columns_to_keep = ['retrieved_api_nums', 'Training Accuracy', 'Validation Accuracy', 'Training ambiguous Accuracy', 'Validation ambiguous Accuracy']

num_rows = 2
num_cols = 2

libs = df['LIB'].unique()[:num_rows * num_cols]

plt.figure(figsize=(15, 10))

for index, lib in enumerate(libs):
    plt.subplot(num_rows, num_cols, index + 1)
    lib_df = df[df['LIB'] == lib][columns_to_keep]
    for col in columns_to_keep[1:]:
        plt.plot(lib_df['retrieved_api_nums'], lib_df[col], label=col)
    plt.legend()
    plt.title(f'Fine-tuned Retriever Accuracy v.s. Topk for {lib}')
    plt.xlabel('Topk')
    plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('./plot/retriever_acc_lib_4.jpg')
plt.show()

