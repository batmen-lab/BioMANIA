import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def process_topicalchat():
    import json
    import pandas as pd
    with open('./data/conversations/train.json', 'r') as file:
        data = json.load(file)
    questions = []
    for conversation_id, conversation_data in data.items():
        content = conversation_data["content"]
        for message_data in content:
            message = message_data["message"]
            questions.append(message)
    df = pd.DataFrame({"Question": questions, "Source": "topical-chat"})
    df.to_csv("./data/others-data/dialogue_questions.csv", index=False)
def process_chitchat():
    import pandas as pd
    import glob
    file_paths = glob.glob("./data/others-data/*.tsv")
    combined_data = pd.DataFrame(columns=["Question", "Source"])
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\t')
        questions = df["Question"]
        #source = file_path.split("_", 1)[-1].split(".")[0]
        temp_df = pd.DataFrame({"Question": questions, "Source": "chitchat-data"})
        combined_data = pd.concat([combined_data, temp_df], ignore_index=True)
    combined_data.to_csv("./data/others-data/combined_data.csv", sep=',', index=False)
    print("Data has been combined and saved as combined_data.csv")
def process_apiquery(lib_name, filename="API_inquiry.json", start_id=0, index_save=True):
    import json
    import pandas as pd
    with open(f'./data/standard_process/{lib_name}/{filename}', 'r') as file:
        json_data = json.load(file)
    filtered_data = [entry for entry in json_data if entry['query_id'] >= start_id]
    questions = [entry['query'] for entry in filtered_data]
    df = pd.DataFrame({'Question': questions, 'Source': 'api-query'})
    if index_save:
        df.to_csv(f'./data/standard_process/{lib_name}/api_data.csv', index=False)
    return df

def sampledata_combine(data1, data2, data3, test_data3, train_count_data1=1000, train_count_data2=1000, train_count_data3=1000,
                       test_count_data1=500, test_count_data2=500, test_count_data3=500):
    train_data1 = data1.sample(n=min(train_count_data1, len(data1)), random_state=42)
    train_data2 = data2.sample(n=min(train_count_data2, len(data2)), random_state=42)
    #train_data3 = data3.sample(n=min(train_count_data3, len(data3)), random_state=42)
    train_data3 = data3
    remaining_data1 = data1.drop(train_data1.index)
    remaining_data2 = data2.drop(train_data2.index)
    #remaining_data3 = data3.drop(train_data3.index)
    test_data1 = remaining_data1.sample(n=min(test_count_data1, len(remaining_data1)), random_state=1)
    test_data2 = remaining_data2.sample(n=min(test_count_data2, len(remaining_data2)), random_state=1)
    #test_data3 = remaining_data3.sample(n=min(test_count_data3, len(remaining_data3)), random_state=1)

    train_data = pd.concat([train_data1, train_data2, train_data3], ignore_index=True)
    test_data = pd.concat([test_data1, test_data2, test_data3], ignore_index=True)
    train_data.to_csv('./data/others-data/train_data.csv', index=False)
    test_data.to_csv('./data/others-data/test_data.csv', index=False)
    print("Train data and test data have been saved.")

def calculate_centroid(tfidf_matrix):
    return np.mean(tfidf_matrix, axis=0).A[0]  # Here we convert the matrix to array

def predict_by_similarity(user_query_vector, centroids, labels):
    similarities = [cosine_similarity(user_query_vector, centroid.reshape(1, -1)) for centroid in centroids]
    return labels[np.argmax(similarities)]

def plot_tsne_distribution_modified(lib_name, train_data, test_data, vectorizer, labels, c2_accuracy):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Combine train and test data for t-SNE
    combined_data = pd.concat([train_data['Question'], test_data['Question']], ignore_index=True)
    tfidf_matrix_combined = vectorizer.transform(combined_data)
    
    tsne = TSNE(n_components=2, random_state=40, init='random')
    reduced_data_combined = tsne.fit_transform(tfidf_matrix_combined)  # Fit and transform combined data
    
    # Split the reduced data back into train and test portions
    reduced_data_train = reduced_data_combined[:len(train_data)]
    reduced_data_test = reduced_data_combined[len(train_data):]
    
    # Define a list of colors for each class
    colors = ['#8a2be2', '#8a2be2', '#fad02e']  # Modify the colors list as per your requirement
    markers_train = ['o', 'D', 'o']
    markers_test = ['D', 'D', 'D']
    
    # Create a figure for train data
    plt.figure(figsize=(12, 12))
    
    # Plot training data points
    for i, label in enumerate(labels):
        mask = train_data['Source'] == label
        plt.scatter(reduced_data_train[mask, 0], reduced_data_train[mask, 1], 
                    label=f"Train-{label}", color=colors[i], marker=markers_train[i], alpha=0.5)
    formatted_accuracy = "{:.2f}".format(c2_accuracy)
    plt.title(f't-SNE visualization of train data with test accuracy for api/non-api {formatted_accuracy}%')
    plt.legend()
    plt.savefig(f'./plot/{lib_name}/chitchat_train_tsne_modified.png')
    plt.clf()  # Clear the current figure
    
    # Create a figure for test data
    plt.figure(figsize=(12, 8))
    
    # Plot test data points with different markers
    for i, label in enumerate(labels):
        mask = test_data['Source'] == label
        plt.scatter(reduced_data_test[mask, 0], reduced_data_test[mask, 1], 
                    label=f"Test-{label}", color=colors[i], marker=markers_test[i], alpha=0.5)
    
    plt.title('t-SNE visualization of test data')
    plt.legend()
    plt.savefig(f'./plot/{lib_name}/chitchat_test_tsne_modified.png')

def main():
    parser = argparse.ArgumentParser(description="Process data with a specified library.")
    parser.add_argument("--LIB", type=str, default="scanpy", required=True, help="Library to use for data processing.")
    parser.add_argument("--ratio_1_to_3", type=float, default=4.0, help="Ratio of data1 to data3.")
    parser.add_argument("--ratio_2_to_3", type=float, default=4.0, help="Ratio of data2 to data3.")
    args = parser.parse_args()
    process_topicalchat()
    process_chitchat()
    tmp_data = process_apiquery(args.LIB)
    import json
    with open(f'./data/standard_process/{args.LIB}/API_inquiry.json', 'r') as file:
        json_data = json.load(file)
    max_query_id = max(entry['query_id'] for entry in json_data)
    test_data3 = process_apiquery(args.LIB, 'API_inquiry_annotate.json', start_id=max_query_id + 1, index_save=False)
    with open(f'./data/standard_process/{args.LIB}/API_inquiry_annotate.json', 'r') as file:
        json_data = json.load(file)
    print(max_query_id)
    assert len(test_data3)+len(tmp_data)==len(json_data)
    data1 = pd.read_csv('./data/others-data/dialogue_questions.csv')
    data2 = pd.read_csv('./data/others-data/combined_data.csv')
    data3 = pd.read_csv(f'./data/standard_process/{args.LIB}/api_data.csv')
    #min_length = min(len(data1), len(data2), len(data3))
    #train_sample_num = int(min_length * 0.8)
    #test_sample_num = int(min_length * 0.2)

    min_length_train = len(data3)
    min_length_test = len(test_data3)
    #train_ratio = 0.8 # no need to split now
    train_sample_num1 = int(min_length_train*args.ratio_1_to_3)
    train_sample_num2 = int(min_length_train*args.ratio_2_to_3)
    test_sample_num1 = int(min_length_test*args.ratio_1_to_3)
    test_sample_num2 = int(min_length_test*args.ratio_2_to_3)
    
    sampledata_combine(data1, data2, data3, test_data3, train_count_data1=train_sample_num1, train_count_data2=train_sample_num2, train_count_data3=min_length_train, test_count_data1=test_sample_num1, test_count_data2=test_sample_num2, test_count_data3=min_length_test)
    train_data = pd.read_csv('./data/others-data/train_data.csv')
    test_data = pd.read_csv('./data/others-data/test_data.csv')
    train_data1 = train_data[train_data['Source'] == 'chitchat-data']
    train_data2 = train_data[train_data['Source'] == 'topical-chat']
    train_data3 = train_data[train_data['Source'] == 'api-query']
    print('length of train_data1, train_data2, train_data3: ', len(train_data1), len(train_data2), len(train_data3))
    print('The real ratio for data1, data2 based on API data is: ', len(train_data1)/len(train_data3), len(train_data2)/len(train_data3))
    vectorizer = TfidfVectorizer()
    all_data = pd.concat([train_data1, train_data2, train_data3], ignore_index=True)
    vectorizer.fit(all_data['Question'])
    tfidf_matrix1 = vectorizer.transform(train_data1['Question'])
    tfidf_matrix2 = vectorizer.transform(train_data2['Question'])
    tfidf_matrix3 = vectorizer.transform(train_data3['Question'])
    centroid1 = calculate_centroid(tfidf_matrix1)
    centroid2 = calculate_centroid(tfidf_matrix2)
    centroid3 = calculate_centroid(tfidf_matrix3)
    centroids = [centroid1, centroid2, centroid3]
    labels = ['chitchat-data', 'topical-chat', 'api-query']
    def calculate_accuracy(test_data, vectorizer, centroids, labels):
        correct_predictions = 0
        for index, row in test_data.iterrows():
            user_query_vector = vectorizer.transform([row['Question']])
            predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
            actual_label = row['Source']
            if predicted_label == actual_label:
                correct_predictions += 1
        return correct_predictions / len(test_data) * 100 if len(test_data) > 0 else 0, correct_predictions, len(test_data)
    test_data_api = test_data[test_data['Source'] == 'api-query']
    test_data_chitchat = test_data[test_data['Source'] == 'chitchat-data']
    test_data_topical = test_data[test_data['Source'] == 'topical-chat']
    #test_data_chitchat = test_data[(test_data['Source'] == 'chitchat-data') | (test_data['Source'] == 'topical-chat')]
    c_api_accuracy, correct_predictions_api, total_predictions_api = calculate_accuracy(test_data_api, vectorizer, centroids, labels)
    print(f"Accuracy on test data (API queries): {c_api_accuracy:.2f}%")
    # Calculate accuracy for Chitchat data
    c_chitchat_accuracy, correct_predictions_chitchat, total_predictions_chitchat = calculate_accuracy(test_data_chitchat, vectorizer, centroids, labels)
    print(f"Accuracy on test data (Chitchat queries): {c_chitchat_accuracy:.2f}%")
    c_topical_accuracy, correct_predictions_topical, total_predictions_topical = calculate_accuracy(test_data_topical, vectorizer, centroids, labels)
    print(f"Accuracy on test data (Topical queries): {c_topical_accuracy:.2f}%")
    c3_accuracy, correct_predictions_c3, total_predictions = calculate_accuracy(test_data, vectorizer, centroids, labels)
    print(f"Accuracy on test data on 3 clusters: {c3_accuracy:.2f}")
    
    assert correct_predictions_api + correct_predictions_chitchat + correct_predictions_topical == correct_predictions_c3, "Sum of correct predictions in subsets does not equal total correct predictions"
    assert total_predictions_api + total_predictions_chitchat + total_predictions_topical == total_predictions, "Sum of item counts in subsets does not equal total item count in test data"

    correct_predictions = 0
    for index, row in test_data.iterrows():
        user_query_vector = vectorizer.transform([row['Question']])
        predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
        actual_label = row['Source']
        if (actual_label=='api-query' and predicted_label=='api-query') or (actual_label!='api-query' and predicted_label!='api-query'):
            correct_predictions += 1
    c2_accuracy = correct_predictions / len(test_data) * 100
    print(f"Accuracy on test data on 2 clusters: {c2_accuracy:.2f}")

    import time
    start_time = time.time()
    import pickle
    with open(f'./data/standard_process/{args.LIB}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved. Time taken: {time.time() - start_time:.2f} seconds")

    with open(f'./data/standard_process/{args.LIB}/centroids.pkl', 'wb') as f:
        pickle.dump(centroids, f)
    start_time = time.time()
    os.makedirs(f"./plot/{args.LIB}", exist_ok=True)
    print(f"Centroids saved. Time taken: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    # Call the modified function to plot
    #plot_tsne_distribution_modified(args.LIB, train_data, test_data, vectorizer, labels, c2_accuracy)
    #print(f"t-SNE plot created. Time taken: {time.time() - start_time:.2f} seconds")

if __name__=='__main__':
    main()
