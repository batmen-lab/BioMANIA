#!/bin/bash

# usage: bash -x scripts/step4_analysis_retriever.sh

export HUGGINGPATH=./hugging_models
libs=("scanpy" "squidpy" "ehrapy" "snapatac2")

mkdir -p output
mkdir -p plot/
csv_file="plot/retriever_topk_results.csv"

echo "LIB,retrieved_api_nums,Training Accuracy,Validation Accuracy,Test Accuracy,Training ambiguous Accuracy,val ambiguous Accuracy,test ambiguous Accuracy" > $csv_file

for LIB in "${libs[@]}"; do
    echo "Processing LIB: $LIB"
    for nums in 1 2 3 5 8 10; do
        echo "Running with retrieved_api_nums=${nums}"
        output=$(python inference/retriever_finetune_inference.py \
            --retrieval_model_path ./hugging_models/retriever_model_finetuned/${LIB}/assigned \
            --max_seq_length 256 \
            --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
            --input_query_file ./data/standard_process/${LIB}/API_inquiry_annotate.json \
            --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
            --retrieved_api_nums $nums \
            --LIB $LIB \
            --filter_composite)
            
        train_acc=$(echo "$output" | grep "Train Accuracy" | grep -oP "\d+\.\d+")
        val_acc=$(echo "$output" | grep "Val Accuracy" | grep -oP "\d+\.\d+")
        test_acc=$(echo "$output" | grep "Test Accuracy" | grep -oP "\d+\.\d+")
        amb_train_acc=$(echo "$output" | grep "Train ambiguous Accuracy" | grep -oP "\d+\.\d+")
        amb_val_acc=$(echo "$output" | grep "Val ambiguous Accuracy" | grep -oP "\d+\.\d+")
        amb_test_acc=$(echo "$output" | grep "Test ambiguous Accuracy" | grep -oP "\d+\.\d+")
        echo "${LIB},${nums},${train_acc},${val_acc},${test_acc},${amb_train_acc},${amb_val_acc},${amb_test_acc}" >> $csv_file

        echo "Completed for LIB=$LIB with retrieved_api_nums=${nums}"
    done
done

python scripts/step4_analysis_retriever.py