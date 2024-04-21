# main_multi_libraries.py

import argparse
from models.chitchat_classification import sampledata_combine, calculate_centroid, predict_by_similarity, plot_tsne_distribution_modified, process_topicalchat, process_chitchat
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gpt.utils import load_json

def process_multi_libraries(libraries):
    global combined_api_data
    combined_api_data = pd.DataFrame(columns=['Question', 'Source'])
    
    def process_apiquery(lib_name):
        global combined_api_data
        json_data = load_json(f'./data/standard_process/{lib_name}/API_inquiry_annotate.json')
        questions = [entry['query'] for entry in json_data]
        df = pd.DataFrame({'Question': questions, 'Source': 'api-query'})
        combined_api_data = pd.concat([combined_api_data, df], ignore_index=True)
    
    for lib_name in libraries:
        process_apiquery(lib_name)
    os.makedirs('./data/standard_process/multicorpus', exist_ok=True)
    combined_api_data.to_csv('./data/standard_process/multicorpus/api_data.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="Process data with multiple libraries.")
    parser.add_argument("--libraries", type=str, nargs='+', default=['scanpy', 'squidpy', 'pyopenms', 'qiime2', 'pyteomics', 'biopython', 'biotite', 'deap', 'eletoolkit', 'scenicplus', 'scikit-bio', 'scvi-tools'], help="List of libraries to process.")
    args = parser.parse_args()
    
    process_topicalchat()
    process_chitchat()
    process_multi_libraries(args.libraries)
    
    data1 = pd.read_csv('./data/others-data/dialogue_questions.csv')
    data2 = pd.read_csv('./data/others-data/combined_data.csv')
    data3 = pd.read_csv(f'./data/standard_process/multicorpus/api_data.csv')
    min_length = min(len(data1), len(data2), len(data3))
    train_sample_num = int(min_length * 0.8)
    test_sample_num = int(min_length * 0.2)
    sampledata_combine(data1, data2, data3,train_count_data1=train_sample_num, train_count_data2=train_sample_num, train_count_data3=train_sample_num, test_count_data1=test_sample_num, test_count_data2=test_sample_num, test_count_data3=test_sample_num)
    train_data = pd.read_csv('./data/others-data/train_data.csv')
    test_data = pd.read_csv('./data/others-data/test_data.csv')
    train_data1 = train_data[train_data['Source'] == 'chitchat-data']
    train_data2 = train_data[train_data['Source'] == 'topical-chat']
    train_data3 = train_data[train_data['Source'] == 'api-query']
    print('length of train_data1, train_data2, train_data3: ', len(train_data1), len(train_data2), len(train_data3))
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
    correct_predictions = 0
    for index, row in test_data.iterrows():
        user_query_vector = vectorizer.transform([row['Question']])
        predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
        actual_label = row['Source']
        if predicted_label == actual_label:
            correct_predictions += 1
    c3_accuracy = correct_predictions / len(test_data)*100
    print(f"Accuracy on test data on 3 clusters: {c3_accuracy:.2f}")
    correct_predictions = 0
    for index, row in test_data.iterrows():
        user_query_vector = vectorizer.transform([row['Question']])
        predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
        actual_label = row['Source']
        if (actual_label=='api-query' and predicted_label=='api-query') or (actual_label!='api-query' and predicted_label!='api-query'):
            correct_predictions += 1
    c2_accuracy = correct_predictions / len(test_data) * 100
    print(f"Accuracy on test data on 2 clusters: {c2_accuracy:.2f}")

    import pickle
    with open(f'./data/standard_process/multicorpus/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'./data/standard_process/multicorpus/centroids.pkl', 'wb') as f:
        pickle.dump(centroids, f)
    os.makedirs(f"./plot/multicorpus", exist_ok=True)

    # Call the modified function to plot
    plot_tsne_distribution_modified("multicorpus", train_data, test_data, vectorizer, labels, c2_accuracy)

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == '__main__':
    main()
