# usage: sh step2_train_models.sh

# train chitchat model
python models/chitchat_classification.py --LIB ${LIB} --ratio_1_to_3 1.0 --ratio_2_to_3 1.0 --embed_method st_untrained

# fine-tune retriever
mkdir ./hugging_models/retriever_model_finetuned/${LIB}
python models/train_retriever.py \
    --data_path ./data/standard_process/${LIB}/retriever_train_data/ \
    --model_name all-MiniLM-L6-v2 \
    --output_path ./hugging_models/retriever_model_finetuned/${LIB} \
    --num_epochs 20 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --optimize_top_k 3 \
    --plot_dir ./plot/${LIB}/retriever/ \
    --gpu "0"

# (Optional) multi-label classification
#export TOKENIZERS_PARALLELISM=true
#python models/data_classification.py \
#    --pretrained_path ./hugging_models/llama-2-finetuned/checkpoints/lite-llama2/lit-llama.pth \
#    --tokenizer_path ./hugging_models/llama-2-finetuned/checkpoints/tokenizer.model \
#    --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
#    --retriever_path ./hugging_models/retriever_model_finetuned/${LIB}/assigned/ \
#    --data_dir ./data/standard_process/${LIB}/API_inquiry_annotate.json \
#    --out_dir ./hugging_models/llama-2-finetuned/${LIB}/finetuned/ \
#    --plot_dir ./plot/${LIB}/classification \
#    --device_count 1 \
#    --top_k 3 \
#    --debug_mode "1" \
#    --save_path ./data/standard_process/${LIB}/classification_train \
#    --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
#    --API_composite_dir ./data/standard_process/${LIB}/API_composite.json \
#    --batch_size 8 \
#    --retrieved_path ./data/standard_process/${LIB} \
#    --LIB ${LIB}


# finetune model
#python models/train_classification.py \
#    --data_dir ./data/standard_process/${LIB}/classification_train/ \
#    --out_dir ./hugging_models/llama-2-finetuned/${LIB}/finetuned/ \
#    --plot_dir ./plot/${LIB}/classification \
#    --max_iters 120 \
#    --batch_size 8

