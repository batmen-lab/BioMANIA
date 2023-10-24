### ReadMe for BioChat Project

[TODO]

[TODO]Check the above step one by one, avoid any error. 

[TODO]Delete any private information like API keys.

[TODO]Delete any extra information, like unused models and scripts.

[TODO]Update the above paths and replace them with placeholder paths.

[TODO]Polish the README document.

[TODO]Add references to the project and make sure all used references are in English.

[TODO]change the installation page to a more appropriate image

[TODO]GPU/CPU mode

[TODO]Add report

[TODO]Add video demo

This ReadMe provides an overview of the steps involved in setting up and running the BioChat project. The project focuses on providing a chatbot interface for interacting with APIs and answering questions related to various libraries and frameworks.

Our pipeline is as below:
![](images/overview_v2.jpg)

We provide a demo chatbot UI 
![](images/UI.jpg)

### setting up
Set up for environment
```shell
pip install -r requirements.txt
```

### User Interface (UI)
The User Interface (UI) for the BioChat project allows you to interact with the chatbot and utilize its functionalities. Below is an explanation of the provided code and how to launch the UI:

In the following demos, We use LIB=scanpy as the example 

```shell
export LIB=scanpy
CUDA_VISIBLE_DEVICES=0 \
python deploy/inference_dialog_server.py \
    --retrieval_model_path ./retriever_model_finetuned/${LIB}/assigned/ \
    --top_k 3 \
    --device_count 1
```
After running the Python script above, the back-end service will start.

If you run the front-end service and back-end service on different device, you can run the ngrok service script.
```shell
ngrok http 5000
```

Then downloading the chatbot-UI-Biochat code and modify the url in utils/server/index.ts
```shell
export const url = "https://localhost:5000";
```
change this url to the link you get from ngrok page after running `ngrok http 5000`

run:
```shell
npm run dev
```
This will start the front-end service.

The chatbot's server should be up and running, ready to install library and accept user queries.

### Inference
If you want to skip the installation step and enter user query at first, you can download the data and models below.

[Download the data and models](https://drive.google.com/drive/folders/1vWef2csBMe-PSPqA9pY2IVCY_JT5ac7p?usp=drive_link) from our Google Drive link.
Once you've downloaded the files, proceed to organize them as follows:

Data Organization:

Place the data files into the directory: src/data/standard_process/${LIB}/

Model Organization:

Store the model files in the directory: src/hugging_models/retriever_model_finetuned/${LIB}/assigned/

Chitchat data Organization:

Place the data files into the directory: src/data/

Here is the example of the subfolders under data folder

```
data
├── conversations
│   ├── api_data.csv
│   ├── test_freq.json
│   ├── test_rare.json
│   ├── train.json
│   ├── valid_freq.json
│   └── valid_rare.json
├── others-data
│   ├── api_data.csv
│   ├── combined_data.csv
│   ├── dialogue_questions.csv
│   ├── final_data.csv
│   ├── qna_chitchat_caring.tsv
│   ├── qna_chitchat_enthusiastic.tsv
│   ├── qna_chitchat_friendly.tsv
│   ├── qna_chitchat_professional.tsv
│   ├── qna_chitchat_witty.tsv
│   ├── test_data.csv
│   └── train_data.csv
├── standard_process
│   ├── API_base.json
│   ├── API_inquiry_annotate_ori.json
│   ├── pyteomics
│   ├── qiime2
│   ├── scanpy
│   ├── scikit-bio
│   └── squidpy
```

By following these steps, you will have the necessary data and models in the correct locations for your project.

Currently, we only provide scanpy data and pretrained models. If you want to test it on more libraries, you can use our new library installation service.

### Installation of new library

By entering the materials link under the `custom` mode on the lib selection, we can install the new library as below
![](../images/install_1.jpg)

![](../images/install_2.jpg)

![](../images/install_3.jpg)

### Training
Besides UI service, we also provide training script.

Firstly, you can modify the library setting under the configs/model_config.py
```shell
LIB = 'scanpy'
USER_INPUT =     
{
    ...
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
}
```
you can download the code, readthedoc, tutorial materials after then.

Step2, you can generate API_init.json by script:
```shell
python dataloader/get_API_init_from_sourcecode.py
```

Step3, you can generate API_composite.json by script:
```shell
python dataloader/get_API_composite_from_tutorial.py
```

Step4, you can generate instruction and generate API_inquiry.json, API_inquiry_annotate.json, and train/val/test split here.
```shell
python dataloader/preprocess_retriever_data.py --LIB scanpy
```
Notice that the automatically generated API_inquiry_annotate.json do not have human annotated data here, you need to annotate the API_inquiry_annotate.json by yourself

Step5, you can train api/non-api classification model
```shell
python models/chitchat_classification.py
```

Step6, you can try bm25 retriever without training here.
```shell
python inference/retriever_bm25_inference.py --LIB scanpy --top_k 1
```

Also, you can finetune the retriever based on the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model
```shell
export LIB=scanpy
mkdir /home/z6dong/BioChat/hugging_models/retriever_model_finetuned/${LIB}
python models/train_retriever.py \
    --data_path ./data/standard_process/${LIB}/retriever_train_data/ \
    --model_name bert-base-uncased \
    --output_path /home/z6dong/BioChat/hugging_models/retriever_model_finetuned/${LIB} \
    --num_epochs 25 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --optimize_top_k 3 \
    --plot_dir ./plot/${LIB}/retriever/
```

Step7, after training, you can test the inference
```shell 
export LIB=scanpy
export HUGGINGPATH=/home/z6dong/BioChat/hugging_models
python inference/retriever_finetune_inference.py  \
    --retrieval_model_path /home/z6dong/BioChat/hugging_models/retriever_model_finetuned/${LIB}/assigned/ \
    --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
    --input_query_file ./data/standard_process/${LIB}/API_inquiry_annotate.json \
    --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
    --retrieved_api_nums 1

export LIB=scanpy
export HUGGINGPATH=/home/z6dong/BioChat/hugging_models
python inference/retriever_finetune_inference.py  \
    --retrieval_model_path bert-base-uncased \
    --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
    --input_query_file ./data/standard_process/${LIB}/API_inquiry_annotate.json \
    --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
    --retrieved_api_nums 1
```

Besides, even though we use gpt prompt to predict api, we also provide an api-name prediction classification model

process data:
```shell
CUDA_VISIBLE_DEVICES=0
export LIB=scanpy
export TOKENIZERS_PARALLELISM=true
python models/data_classification.py \
    --pretrained_path /home/z6dong/BioChat/hugging_models/llama-2-finetuned/checkpoints/lite-llama2/lit-llama.pth \
    --tokenizer_path /home/z6dong/BioChat/hugging_models/llama-2-finetuned/checkpoints/tokenizer.model \
    --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
    --retriever_path /home/z6dong/BioChat/hugging_models/retriever_model_finetuned/${LIB}/assigned/ \
    --data_dir ./data/standard_process/${LIB}/API_inquiry_annotate.json \
    --out_dir /home/z6dong/BioChat/hugging_models/llama-2-finetuned/${LIB}/finetuned/ \
    --plot_dir ./plot/${LIB}/classification \
    --device_count 1 \
    --top_k 10 \
    --debug_mode "1" \
    --save_path ./data/standard_process/${LIB}/classification_train \
    --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
    --API_composite_dir ./data/standard_process/${LIB}/API_composite.json \
    --batch_size 8 \
    --retrieved_path ./data/standard_process/${LIB}
```

Then, finetune model:
```shell
export LIB=scanpy
CUDA_VISIBLE_DEVICES=0 \
python models/train_classification.py \
    --data_dir ./data/standard_process/${LIB}/classification_train/ \
    --out_dir /home/z6dong/BioChat/hugging_models/llama-2-finetuned/${LIB}/finetuned/ \
    --plot_dir ./plot/${LIB}/classification \
    --max_iters 120 \
    --batch_size 8
```

Finally, check the performance:
```shell
export LIB=scanpy
CUDA_VISIBLE_DEVICES=0 \
python models/inference_classification.py \
    --data_dir ./data/standard_process/${LIB}/classification_train/ \
    --checkpoint_dir /home/z6dong/BioChat/hugging_models/llama-2-finetuned/${LIB}/finetuned/combined_model_checkpoint.pth \
    --batch_size 1
```

### Reference

Thanks to the references below:
- [Toolbench](https://github.com/OpenBMB/ToolBench) 
- [Chatbot-UI](https://github.com/mckaywrigley/chatbot-ui)
- [Toolbench-UI](https://github.com/lilbillybiscuit/chatbot-ui-toolllama)
- [Retriever](https://huggingface.co/bert-base-uncased)
- [Topical-Chat-data](https://github.com/alexa/Topical-Chat)
- [ChitChat-data](https://github.com/microsoft/botframework-cli/blob/main/packages/qnamaker/docs/chit-chat-dataset.md)
- [lit-llama](https://github.com/Lightning-AI/lit-llama)

