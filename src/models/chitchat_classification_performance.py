"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: This script contains chitchat classification performance evaluation functions.
"""
import os
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from .chitchat_classification import process_chitchat, process_apiquery, sampledata_combine, calculate_centroid, calculate_accuracy, load_json
from ..inference.utils import sentence_transformer_embed, predict_by_similarity

def collect_test_accuracy(lib_data_path:str, LIB: str, ratio_1_to_3: float, ratio_2_to_3: float, embed_method: str, device: torch.device) -> float:
    """
    Collects the test accuracy for a specified library.

    Parameters
    ----------
    lib_data_path: str
        The path to the library data.
    LIB : str
        The library name to process.
    ratio_1_to_3 : float
        The ratio of the first dataset to the third dataset in training.
    ratio_2_to_3 : float
        The ratio of the second dataset to the third dataset in training.
    embed_method : str
        Specifies the embedding method to use.
    device : torch.device
        The device configuration for PyTorch operations.

    Returns
    -------
    float
        The test accuracy for the specified library.
    """
    # Process the data
    process_chitchat()
    tmp_data = process_apiquery(lib_data_path)
    json_data = load_json(os.path.join(lib_data_path, 'API_inquiry.json'))
    max_query_id = max(entry['query_id'] for entry in json_data)
    test_data3 = process_apiquery(lib_data_path, 'API_inquiry_annotate.json', start_id=max_query_id + 1, index_save=False)
    
    # Load other datasets
    data1 = pd.read_csv('./data/others-data/dialogue_questions.csv')
    data2 = pd.read_csv('./data/others-data/combined_data.csv')
    data3 = pd.read_csv(os.path.join(lib_data_path, 'api_data.csv'))

    # Sample data for training and testing
    min_length_train = len(data3)
    min_length_test = len(test_data3)
    train_sample_num1 = int(min_length_train * ratio_1_to_3)
    train_sample_num2 = int(min_length_train * ratio_2_to_3)
    test_sample_num1 = int(min_length_test * ratio_1_to_3)
    test_sample_num2 = int(min_length_test * ratio_2_to_3)

    sampledata_combine(data1, data2, data3, test_data3, train_count_data1=train_sample_num1, train_count_data2=train_sample_num2, train_count_data3=min_length_train, test_count_data1=test_sample_num1, test_count_data2=test_sample_num2, test_count_data3=min_length_test)
    
    # Load the sampled train and test data
    train_data = pd.read_csv('./data/others-data/train_data.csv')
    test_data = pd.read_csv('./data/others-data/test_data.csv')
    
    # Load the model
    if embed_method == "st_untrained":
        model_chosen = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    elif embed_method == "st_trained":
        model_chosen = SentenceTransformer(f"./hugging_models/retriever_model_finetuned/{LIB}/assigned", device=device)
    else:
        raise ValueError("Invalid embedding method")
    
    # Calculate centroids
    centroid1 = calculate_centroid(train_data[train_data['Source'] == 'chitchat-data']['Question'], model_chosen)
    centroid2 = calculate_centroid(train_data[train_data['Source'] == 'topical-chat']['Question'], model_chosen)
    centroid3 = calculate_centroid(train_data[train_data['Source'] == 'api-query']['Question'], model_chosen)
    centroids = [centroid1, centroid2, centroid3]
    labels = ['chitchat-data', 'topical-chat', 'api-query']
    
    # Calculate test accuracy
    test_accuracy, _, _ = calculate_accuracy(test_data, centroids, labels, model_chosen)
    
    c3_accuracy, correct_predictions_c3, total_predictions = calculate_accuracy(test_data, centroids, labels, model_chosen)
    print(f"Accuracy on test data on 3 clusters: {c3_accuracy:.2f}")
    
    correct_predictions_c2 = 0
    correct_predictions_api_c2 = 0
    correct_predictions_nonapi_c2 = 0
    for index, row in test_data.iterrows():
        user_query_vector = np.array(sentence_transformer_embed(model_chosen, [row['Question']])).flatten().reshape(1,-1)
        predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
        actual_label = row['Source']
        if predicted_label == 'api-query' and actual_label == 'api-query':
            correct_predictions_api_c2 += 1
            correct_predictions_c2+=1
        elif predicted_label != 'api-query' and actual_label != 'api-query':
            correct_predictions_nonapi_c2 += 1
            correct_predictions_c2+=1
    total_api_c2 = len(test_data[test_data['Source'] == 'api-query'])
    total_nonapi_c2 = len(test_data[(test_data['Source'] == 'chitchat-data') | (test_data['Source'] == 'topical-chat')])
    assert total_api_c2+total_nonapi_c2==len(test_data)
    accuracy_c2 = correct_predictions_c2 / len(test_data) * 100
    return accuracy_c2

def plot_test_accuracies(accuracies, colors):
    """
    Plots a bar chart of test accuracies for each library.

    Parameters
    ----------
    accuracies : dict
        Dictionary containing the test accuracy for each library.
    colors : list
        List of colors to use for the bars.
    """
    # Sort the libraries and corresponding accuracies by the specified order
    sorted_libraries = ['scanpy', 'squidpy', 'ehrapy', 'snapatac2']
    sorted_accuracies = {lib: accuracies[lib] for lib in sorted_libraries}
    
    # Convert the sorted accuracies dictionary to a DataFrame for plotting
    df = pd.DataFrame(list(sorted_accuracies.items()), columns=['Library', 'Test accuracy'])
    
    # Plotting
    sns.set(style="white")  # No grid
    plt.figure(figsize=(8, 8))  # Make the plot square-shaped
    ax = sns.barplot(x='Test accuracy', y='Library', data=df, palette=colors)

    # Add data labels (without % sign)
    for index, value in enumerate(df['Test accuracy']):
        ax.text(value + 0.5, index, f'{value:.2f}', color='black', ha="center", va="center")

    plt.title('Chitchat classification test accuracy', fontsize=14)
    plt.xlabel('Test accuracy (%)', fontsize=12)  # % sign in the xlabel
    plt.ylabel('Library', fontsize=12)
    plt.tight_layout()
    plt.savefig('plot_test_accuracies.pdf')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')
    
    # Libraries to process
    libraries = ["scanpy", "squidpy", "ehrapy", "snapatac2"]
    lib_data_path = "./data/standard_process"
    ratio_1_to_3 = 1.0
    ratio_2_to_3 = 1.0
    embed_method = "st_untrained"  # or "st_trained"
    
    # Collect accuracy for each library
    accuracies = {}
    for lib in libraries:
        accuracies[lib] = collect_test_accuracy(lib_data_path+'/'+lib, lib, ratio_1_to_3, ratio_2_to_3, embed_method, device)
    
    # Define the colors you want to use for the plot
    colors = ['#3a4068', '#3b6d94', '#45979c', '#71c6ab']  # Colors in the same order as the sorted libraries
    
    # Plot the test accuracies
    plot_test_accuracies(accuracies, colors)
