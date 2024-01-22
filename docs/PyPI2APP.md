
### Training from scratch

We provide a robust training script for additional customization and enhancement of the BioMANIA project. Follow the steps in the Training section to modify library settings, download materials, generate JSON files, and train models.

Feel free to skip all optional steps below to have a quick start with minimum effort.

1. Modify the library setting in `Lib_cheatsheet.json`.
```bash
{
    ...
    # standard input for PyPI tool
    'scanpy':{
        "LIB":'scanpy',
        "LIB_ALIAS":'scanpy',
        "API_HTML_PATH": 'scanpy.readthedocs.io/en/latest/api/index.html',
        "GITHUB_LINK": "https://github.com/scverse/scanpy",
        "READTHEDOC_LINK": "https://scanpy.readthedocs.io/",
        "TUTORIAL_HTML_PATH":"scanpy.readthedocs.io/en/latest/tutorials",
        "TUTORIAL_GITHUB":"https://github.com/scverse/scanpy-tutorials",
    },
    ...
    # simplest input
    'scikit-learn':{
        "LIB":'scikit-learn', # NECESSARY. Download using PyPI command. 
        "LIB_ALIAS":'sklearn', # NECESSARY. import this alias to execute.
        "API_HTML_PATH": null, # OPTIONAL. Filter out only the APIs intended for user usage
        "GITHUB_LINK": null, # OPTIONAL. Download from the Github repo if you want to use the latest version of code
        "READTHEDOC_LINK": null, # OPTIONAL. If you don't have a specific API page and want we to search API page for you from the READTHEDOC Link
        "TUTORIAL_HTML_PATH": null, # OPTIONAL. If you want to establish composite API by tutorials from readthedoc page
        "TUTORIAL_GITHUB": null, # OPTIONAL. If you have a tutorial github repo and want to use ipynbs for producing composite API
    }
}
```

We use scanpy here as an example. You can substitute it using your lib.
```bash
export LIB=scanpy
```

> **We download API_HTML_PATH instead of the whole READTHEDOC for saving time.**

> **Notice that the READTHEDOC version should be compatible with your PyPI version, otherwise it may ignore some APIs.**

(Optional) Feel free to skip these two scripts if you don't need `composite_API`. They aim to download the necessary readthedoc materials to folder `../../resources/readthedoc_files` with:
```bash
# Under folder `BioMANIA/src`
# download materials according to your provided url links
python dataloader/utils/other_download.py --LIB ${LIB}
# generate codes for your downloaded tutorial files, support for either html, ipynb.
python dataloader/utils/tutorial_loader_strategy.py --LIB ${LIB} --file_type 'ipynb'
```

NOTE it requires API_HTML_PATH, READTHEDOC_PATH and TUTORIAL_GITHUB to run the above script!

Install the PyPI library by below, or other ways that recommended from their Github.
```bash
pip install {LIB}
```

To use web UI smoothly, don't forget to add the new lib information to `BioMANIA/chatbot_ui_biomania/components/Chat/LibCardSelect.tsx`. Also add the new lib logo to `BioMANIA/chatbot_ui_biomania/public/apps/`.

2. Generate API_init.json using the provided script.
```bash
python dataloader/get_API_init_from_sourcecode.py --LIB ${LIB}
```

Note: If you have prepared an API list txt file, you can add `--api_txt_path your_file_path` to extract the API information. The sequence is firstly to recognize the API txt file, if not given then recognize the API html page, finally we start from Lib_ALIAS and check all its submodules.

3. (Optional) Generate API_composite.json automatically with:
```bash
# get composite API if you have already provided tutorial
# REQUEST 
python dataloader/get_API_composite_from_tutorial.py --LIB ${LIB}
# or skip it by running
cp -r ./data/standard_process/${LIB}/API_init.json ./data/standard_process/${LIB}/API_composite.json
```

- NOTE that it requires TUTORIAL_GITHUB to run the first script!
- Notice that this step also requires OpenAI API to polish the docstring of the composite API.  
- For precision's sake, it is more recommended to wrap the API yourself if conditions permit.

If you skip this step, ensure that you contain a file of `./data/standard_process/{LIB}/API_composite.json` to guarantee that the following steps can run smoothly.

4. Following this, create instructions, and split the data for retriever training preparation.
```bash
python dataloader/preprocess_retriever_data.py --LIB ${LIB}
```

(Optional) You can validate of your annotated API_inquiry_annotate.json with 
```bash
python dataloader/check_valid_API_annotate.py ${LIB}
```

Tips:
- (Optional) The automatically generated API_inquiry_annotate.json do not have human annotated data here, you need to annotate the API_inquiry_annotate.json by yourself if you want to test performance on human annotate data.
- If you skip the above step, please only refer to `train/val` performance instead of `test` performance.
- The time cost is related with the total number of APIs of lib and the `paid OpenAI account`.

5. Train the api/non-api classification model.
```bash
python models/chitchat_classification.py --LIB ${LIB} --ratio_1_to_3 1.0 --ratio_2_to_3 1.0 --embed_method st_trained
# or train a classification model on multicorpus of 12 bio-tools.
# python models/chitchat_classification_multicorpus.py
```

If you train a multicorpus one, please remember to copy the saved `.csv` and `.pkl` files under `data/standard_process/multicorpus/` and put under your lib's path.

6. (Optional) Try the unpretrained bm25 retriever for a quick inference on your generated instructions.
```bash
python inference/retriever_bm25_inference.py --LIB ${LIB} --top_k 3
```

7. Fine-tune the retriever.
You can finetune the retriever based on the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model
```bash
mkdir ./hugging_models/retriever_model_finetuned/${LIB}
python models/train_retriever.py \
    --data_path ./data/standard_process/${LIB}/retriever_train_data/ \
    --model_name bert-base-uncased \
    --output_path ./hugging_models/retriever_model_finetuned/${LIB} \
    --num_epochs 25 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --optimize_top_k 3 \
    --plot_dir ./plot/${LIB}/retriever/ \
    --gpu "1"
```

You can check the training performance curve under `./src/plot/${LIB}/` to determine whether you need more number of epochs.

8. Test the inference performance using:
```bash 
export HUGGINGPATH=./hugging_models
python inference/retriever_finetune_inference.py  \
    --retrieval_model_path ./hugging_models/retriever_model_finetuned/${LIB}/assigned \
    --max_seq_length 256 \
    --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
    --input_query_file ./data/standard_process/${LIB}/API_inquiry_annotate.json \
    --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
    --retrieved_api_nums 3 \
    --LIB ${LIB}
```

You can refer to `src/plot/${LIB}/error_train.json` for detailed error case.

9. (Optional) Test api name prediction using gpt baseline.

**Run code inside gpt_baseline.ipynb to check results.** You can either choose top_k, gpt3.5/gpt4 model, random shot/similar shot example, narrowed retrieved api list/whole api list parameters here. The performance described in our paper was evaluated using GPT versions GPT-3.5-turbo-16k-0613 and GPT-4-0613.

Note that using GPT-4 can be costly and is intended solely for reproducibility purposes.

10. (Optional) Test api name prediction using classification model.

Besides, even though we use gpt prompt to predict api during inference period, we also provide an api-name prediction classification model

Please refer to [lit-llama](https://github.com/Lightning-AI/lit-llama) for getting llama weights and preprocessing. 

process data:
```bash
export TOKENIZERS_PARALLELISM=true
python models/data_classification.py \
    --pretrained_path ./hugging_models/llama-2-finetuned/checkpoints/lite-llama2/lit-llama.pth \
    --tokenizer_path ./hugging_models/llama-2-finetuned/checkpoints/tokenizer.model \
    --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
    --retriever_path ./hugging_models/retriever_model_finetuned/${LIB}/assigned/ \
    --data_dir ./data/standard_process/${LIB}/API_inquiry_annotate.json \
    --out_dir ./hugging_models/llama-2-finetuned/${LIB}/finetuned/ \
    --plot_dir ./plot/${LIB}/classification \
    --device_count 1 \
    --top_k 10 \
    --debug_mode "1" \
    --save_path ./data/standard_process/${LIB}/classification_train \
    --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
    --API_composite_dir ./data/standard_process/${LIB}/API_composite.json \
    --batch_size 8 \
    --retrieved_path ./data/standard_process/${LIB} \
    --LIB ${LIB}
```

Then, finetune model:
```bash
python models/train_classification.py \
    --data_dir ./data/standard_process/${LIB}/classification_train/ \
    --out_dir ./hugging_models/llama-2-finetuned/${LIB}/finetuned/ \
    --plot_dir ./plot/${LIB}/classification \
    --max_iters 120 \
    --batch_size 8
```

Finally, check the performance:
```bash
python models/inference_classification.py \
    --data_dir ./data/standard_process/${LIB}/classification_train/ \
    --checkpoint_dir ./hugging_models/llama-2-finetuned/${LIB}/finetuned/combined_model_checkpoint.pth \
    --batch_size 1
```

Next you may want to check the performance by generating reports in [Report_Generation](./Report_Generation.md)
