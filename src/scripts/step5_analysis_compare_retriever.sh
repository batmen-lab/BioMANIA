#!/bin/bash

# Usage: bash -x scripts/step5_analysis_compare_retriever.sh

export HUGGINGPATH=./hugging_models
libs=("scanpy" "squidpy" "ehrapy" "snapatac2") # scanpy_subset
csv_file="output/retriever_accuracy_results.csv"

# Header for CSV file
echo "LIB,BM25 Train,BM25 Val,BM25 Test,Un-Finetuned Train,Un-Finetuned Val,Un-Finetuned Test,Finetuned Train,Finetuned Val,Finetuned Test,BM25 Ambiguous Train,BM25 Ambiguous Val,BM25 Ambiguous Test,Un-Finetuned Ambiguous Train,Un-Finetuned Ambiguous Val,Un-Finetuned Ambiguous Test,Finetuned Ambiguous Train,Finetuned Ambiguous Val,Finetuned Ambiguous Test" > $csv_file

# Loop through each library
for LIB in "${libs[@]}"; do
    # Run BM25 inference
    bm25_output=$(python inference/retriever_bm25_inference.py --LIB ${LIB} --top_k 3)

    # Parse BM25 output
    bm25_train=$(echo "$bm25_output" | grep "train retriever top-3 accuracy rate" | grep -oP "\d+\.\d+")
    bm25_val=$(echo "$bm25_output" | grep "val retriever top-3 accuracy rate" | grep -oP "\d+\.\d+")
    bm25_test=$(echo "$bm25_output" | grep "test retriever top-3 accuracy rate" | grep -oP "\d+\.\d+")
    bm25_train_am=$(echo "$bm25_output" | grep "train retriever top-3 ambiguous accuracy rate" | grep -oP "\d+\.\d+")
    bm25_val_am=$(echo "$bm25_output" | grep "val retriever top-3 ambiguous accuracy rate" | grep -oP "\d+\.\d+")
    bm25_test_am=$(echo "$bm25_output" | grep "test retriever top-3 ambiguous accuracy rate" | grep -oP "\d+\.\d+")

    # Run Fine-tuned Retriever inference (Un-Finetuned)
    unfinetuned_output=$(python inference/retriever_finetune_inference.py \
        --retrieval_model_path all-MiniLM-L6-v2 \
        --max_seq_length 256 \
        --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
        --input_query_file ./data/standard_process/${LIB}/API_inquiry_annotate.json \
        --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
        --retrieved_api_nums 3 \
        --LIB ${LIB} \
        --filter_composite)

    # Parse Fine-tuned Retriever Un-Finetuned output
    unfinetuned_train=$(echo "$unfinetuned_output" | grep "Training Accuracy" | grep -oP "\d+\.\d+")
    unfinetuned_val=$(echo "$unfinetuned_output" | grep "Val Accuracy" | grep -oP "\d+\.\d+")
    unfinetuned_test=$(echo "$unfinetuned_output" | grep "Test Accuracy" | grep -oP "\d+\.\d+")
    unfinetuned_train_amb=$(echo "$unfinetuned_output" | grep "Train ambiguous Accuracy" | grep -oP "\d+\.\d+")
    unfinetuned_val_amb=$(echo "$unfinetuned_output" | grep "Val ambiguous Accuracy" | grep -oP "\d+\.\d+")
    unfinetuned_test_amb=$(echo "$unfinetuned_output" | grep "Test ambiguous Accuracy" | grep -oP "\d+\.\d+")


    # Run Fine-tuned Retriever inference (Finetuned)
    finetuned_output=$(python inference/retriever_finetune_inference.py \
        --retrieval_model_path ./hugging_models/retriever_model_finetuned/${LIB}/assigned \
        --max_seq_length 256 \
        --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
        --input_query_file ./data/standard_process/${LIB}/API_inquiry_annotate.json \
        --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
        --retrieved_api_nums 3 \
        --LIB ${LIB} \
        --filter_composite)

    # Parse Fine-tuned Retriever Finetuned output
    finetuned_train=$(echo "$finetuned_output" | grep "Training Accuracy" | grep -oP "\d+\.\d+")
    finetuned_val=$(echo "$finetuned_output" | grep "Val Accuracy" | grep -oP "\d+\.\d+")
    finetuned_test=$(echo "$finetuned_output" | grep "Test Accuracy" | grep -oP "\d+\.\d+")
    finetuned_train_amb=$(echo "$finetuned_output" | grep "Train ambiguous Accuracy" | grep -oP "\d+\.\d+")
    finetuned_val_amb=$(echo "$finetuned_output" | grep "Val ambiguous Accuracy" | grep -oP "\d+\.\d+")
    finetuned_test_amb=$(echo "$finetuned_output" | grep "Test ambiguous Accuracy" | grep -oP "\d+\.\d+")

    # Combine results and append to CSV
    echo "${LIB},${bm25_train},${bm25_val},${bm25_test},${unfinetuned_train},${unfinetuned_val},${unfinetuned_test},${finetuned_train},${finetuned_val},${finetuned_test},${bm25_train_am},${bm25_val_am},${bm25_test_am},${unfinetuned_train_amb},${unfinetuned_val_amb},${unfinetuned_test_amb},${finetuned_train_amb},${finetuned_val_amb},${finetuned_test_amb}" >> $csv_file

    echo "Completed processing for LIB=$LIB"
done

# Plot the results using the Python script
python scripts/step5_analysis_compare_retriever.py
