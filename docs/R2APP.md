### R Package converting to BioMANIA APP README

This README framework provides a step-by-step guide for integrating R packages into the BioMANIA app. The process is streamlined for R, leveraging its capabilities for direct package usage and documentation access. Python scripts are still used for data preparation and model training. 

#### Step 1: Prerequisites
Before starting, ensure that R and `rpy2` are installed. `rpy2` allows for running R within Python, bridging the two environments.

- **Install R**: Download and install the latest version of R from [CRAN](https://cran.r-project.org/).
- **Install rpy2**: In your Python environment, run 
```shell
pip install rpy2
```

#### Step 2: Generating API Initialization File
Use a Python script to generate the `./data/R/{Your_R_Library_Name}/API_init.json` file. This file contains initial information about the APIs in the R package.

- Run the following command:
```bash
python dataloader/get_API_init_from_sourcecode_R.py --LIB [Your_R_Library_Name]
cp -r ./data/R/{Your_R_Library_Name}/API_init.json /data/R/{Your_R_Library_Name}/API_composite.json
```

#### Step 3: Preparing Retriever Data
Prepare the data required for the retriever model using another Python script. This process generates the instruction under `./data/R/{your_r_lib_name}/API_inquiry.json` file and `./data/R/{your_r_lib_name}/retriever_train_data/` folder.

- Execute the command:
```bash
python dataloader/prepare_retriever_data.py --LIB [Your_R_Library_Name]
```

#### Step 4: Training Models
Train the chat classification and retriever models. These models are crucial for the app's functionality.

1. **Chitchat Classification Model**: 
- Train the model using the following command:
```bash
python models/chitchat_classification.py --LIB ${LIB}
```

2. **Retriever Model**: 
- Train the model with this command (adjust the command with necessary parameters):
```bash
export LIB=scanpy
CUDA_VISIBLE_DEVICES=0
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
--plot_dir ./plot/${LIB}/retriever/
```

#### Step 5: Use model
Start back-end UI service with:
```bash
export LIB=ggplot2
CUDA_VISIBLE_DEVICES=0 \
python deploy/inference_dialog_server_R.py \
    --retrieval_model_path ./hugging_models/retriever_model_finetuned/${LIB}/assigned \
    --top_k 3
```

Install and start the front-end service in a new terminal with:
```bash
cd src/chatbot_ui_biomania
npm i # install
export BACKEND_URL="https://[ngrok_id].ngrok-free.app" # "http://localhost:5000";
npm run dev # run
```

Your chatbot server is now operational at `http://localhost:3000/en`, primed to process user queries.

#### Key Differences Between R and Python Integration

- **Library Loading**: In R, use `library(LIB)` to load packages directly. There's no need to modify `Lib_cheatsheet.json` as in Python.
- **Documentation Access**: R documentation can be accessed through `help()`, `??`, or the `.__doc__` attribute after converting R functions to Python via `rpy2`.
- **Arguments Information**: R documentation didn't always provide `type` information for parameters.
- **Simplified Process**: The process for R integration is more straightforward, focusing primarily on data preparation and model training, without the need to adjust library settings extensively.

#### Final Notes
This framework outlines the key steps and differences for integrating an R package into BioMANIA. Adjust the Python commands and paths according to your package's specifics. If you have any questions or need assistance with specific steps, feel free to reach out!