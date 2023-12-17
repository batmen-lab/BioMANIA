
# Q&A for BioMANIA README:

## Library and Framework Support
**Q: What libraries and frameworks does BioMANIA support?**

A: BioMANIA supports various libraries in bioinformatics, including scanpy, squidpy, scenicplus, scikit-bio, pyopenms, pyteomics, deap, eletoolkit, qiime2, scvi-tools, biopython, biotite. Furthermore, we support two tools from Batmen-lab, sonata and MIOSTONE.

## Switching Libraries
**Q: Can I switch between different libraries within one dialog?**

A: Some tools, like Qiime2 and Scenicplus, require individual environments due to their dependencies. In such cases, users can not include them in one dialog, and need to stop the script and manually switch to the specific Conda environment to run the script.

**Q: How can I switch between different libraries within one dialog?**

A: By clicking on the gear icon in the upper right corner of the UI page and then switch the library.

## Inference Speed and Configuration
**Q: What factors influence the inference speed?**

A: The inference speed is influenced by the OpenAI key and the backend device. Using a `paid OpenAI key` and running the backend on a `GPU` significantly speeds up the inference process. One only needs around 3s to load the retriever model on a piece of `RTX A6000`.

## File Management
**Q: Where are uploaded files saved?**

A: All uploaded files are stored under the `BioMANIA/src/tmp` folder. When providing filename parameters for API requests, users should use the format `./tmp/your_file_name`.

**Q: How can I handle large file transfers to avoid performance issues?**

A: Large file transfers may lead to memory issues. If file transfer is substantial, users can directly copy the file to the src/tmp/ folder to mitigate JavaScript heap memory exhaustion. We have provided this issue by supporting the download through Drive URLs.

Follow the instructions for uploading your data using Google Drive:
- Package your data, even if it's just one file, into a ZIP archive.
- Upload the ZIP file to Google Drive and obtain the file's shareable link.
- Extract the `file_id` from the link and enter it into the `URL` input field on UI page.
- Our program will automatically downloading it and extracted the ZIP file, you can use `./tmp/your_file_name` to infer. It still follows your original file structure.

## Online Demo and Stability
**Q: For online demo, sometimes it shows a network error and does not display results.**

A: One reason is that this system currently has one single backend, and simultaneous user requests may lead to network issue. Another issue is that the operation's stability is influenced by the netowrk of both requested device and our server. Even though we have implemented session management to distinguish requests from different users, it is recommended to run locally on your device for optimal performance.

**Q: I interrupted the conversation, can I continue from the last time?**

A: We have implemented state storage, supporting the ability to resume conversations from the last saved state.

## Installation issues
**Q: What Node.js versions are supported in your project?**

A: version 20.7.0 on Linux as our device.

**Q: How to address the issue if I can not receive message from backend on UI page?**

A: The error may be due to various reasons. 
- Check if the backend and frontend are running correctly. 
- Be careful for the `http/https`, `PORT`, `url` in `chatbot_ui_biomania/utils/server/index.ts`, and the effectiveness of the API key, as they will affect the connection between backend and frontend service.
- Please double check that if your device support `localhost` communication. If not, try the alternative links displayed in the backend logs.

**Q: How to set up services on separate devices?**

A: We provide the start_script.sh to run backend and frontend command simultaneously, you can run each command separately on different devices. Before that, we need to solve the issues of network communication.

Initiate the [ngrok service](https://ngrok.com/docs/getting-started/) script in a new terminal on the same device with back-end device and get the print url like `https://[ngrok_id].ngrok-free.app` with:
```bash
ngrok http 5000
```

Then you can substitute the url from `chatbot_ui_biomania/utils/server/index.ts` in front-end code and start its service

Don't forget to add an OpenAI API key under `BioMANIA/src/.env` path

**Q: To create a tool, what is the minimum material I need to provide?**

A: The `LIB` and `LIB_ALIAS` in configs/Lib_cheatsheet.json is enough for establishing the service. 

- However, incorporating an `API HTML page` will specifically assist in filtering the targeted API exposed to users. 
- Including a `tutorial page` will help for producing `composite_API` by combining a set of commonly together used APIs.

**Q: How to still filter the APIs unwanted if I don't have a valid API page?**

A: You can modify the code from `get_API_init_from_sourcecode.py`:
```bash
# modify ori_content_keys as a list of API you want
ori_content_keys = ['API1', 'API2', ...]
results = get_docparam_from_source(ori_content_keys, lib_name)
```
This `ori_content_keys` was expected to gained from API page, you can modify it manually instead.

## Retriever Training and Performance
**Q: How to train a retriever on my tool faster?**

A: You can either finetuned base on other pretrained model. Don't forget to download the pretrained models from drive link and put it under the corresponding path.

```bash
export LIB=scanpy # example
export OTHER_LIB=squidpy # example
CUDA_VISIBLE_DEVICES=0 # if you use gpu
mkdir ./hugging_models/retriever_model_finetuned/${LIB}
python models/train_retriever.py \
    --data_path ./data/standard_process/${LIB}/retriever_train_data/ \
    --model_name ./hugging_models/retriever_model_finetuned/${OTHER_LIB}/assigned/ \
    --output_path ./hugging_models/retriever_model_finetuned/${LIB} \
    --num_epochs 25 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --optimize_top_k 3 \
    --plot_dir ./plot/${LIB}/retriever/
```

Or finetuned on our pretrained model which is trained on the 12 bio-tools. Note that you need to download the multicorpus.zip from drive link and put it under the corresponding path.

```bash
export LIB=scanpy
export OTHER_LIB=multicorpus
CUDA_VISIBLE_DEVICES=0 # if you use gpu
mkdir ./hugging_models/retriever_model_finetuned/${LIB}
python models/train_retriever.py \
    --data_path ./data/standard_process/${LIB}/retriever_train_data/ \
    --model_name ./hugging_models/retriever_model_finetuned/${OTHER_LIB}/assigned/ \
    --output_path ./hugging_models/retriever_model_finetuned/${LIB} \
    --num_epochs 25 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --optimize_top_k 3 \
    --plot_dir ./plot/${LIB}/retriever/
```

**Q: How to check the performance on the model not pretrained?**

A: You can check it with substituting the `retrieval_model_path`:
```bash
export LIB=scanpy
CUDA_VISIBLE_DEVICES=0 # if you use gpu
export HUGGINGPATH=./hugging_models
python inference/retriever_finetune_inference.py  \
    --retrieval_model_path bert-base-uncased \
    --corpus_tsv_path ./data/standard_process/${LIB}/retriever_train_data/corpus.tsv \
    --input_query_file ./data/standard_process/${LIB}/API_inquiry_annotate.json \
    --idx_file ./data/standard_process/${LIB}/API_instruction_testval_query_ids.json \
    --retrieved_api_nums 3 \
    --LIB ${LIB}
```

**Q: Why don't you provide all the retriever models on your drive url?**

A: The memory usage of each model is quite substantial. Currently we include each individual model within the individual `Docker image`. We also release `a model trained on 12 bio-tools` for the convenience of others for fine-tuning purposes.

**Q: Sometimes the image obtained on UI is quite small to see, how to obtain better visualization?**

A: Currently we only support required parameters inference, the images are generated by default `figsize, color` settings of corresponding API. However, you can try `report generation` in readme file, it will extract and save the images from the dialog json gained from `export chat` button of UI.

**Q: Why the retriever performance is low on my PyPI tool.**

A: There are many reasons. One prominent factor is the presence of a significant number of ambiguous APIs in your tool. Our paper has explained the definition of Ambiguous APIs, 
- API names are similar
- The description parts in the Docstring are similar or even same
After you run the test script of `retriever_finetune_inference`, you will obtain detailed error case files as `src/plot/${LIB}/error_train.json`. You can check if most error cases are due to ambiguous API.

If so, there's no need to worry because during the inference stage, we will present all ambiguous APIs to the frontend for user selection, thus avoiding issues caused by ambiguous API disparities

Another significant reason is the undertraining. You can check the curve under `./src/plot/${LIB}/` to determine whether you need more number of epochs.

**Q: How to improve the performance on my tool?**

A: Regarding improvements, we suggest two approaches. 
- First, you can either provide an API page or modify the rules in the `filter_specific_apis` function within `get_API_init_from_source_code` to exclude the types of APIs you don't want to see during the inference stage. 
- Another approach is to modify the Docstrings to ensure that meaning of APIs do not appear too similar. 

For `filter_specific_apis` function. Currently we remove API type with `property/constant/builtin`, remove API without docstring, API without input/output simultaneously. Most retained APIs are of type `function/method/Class`, which is more meaningful for user query inference. You can check your API_init.json and modify rules accordingly!

## Issues during transferring your tool to BioMANIA APP

**Q: Why doesn't it visualize the figure output using plotting API?**

A: Please ensure that the images are displayed, similar to `plt.show()`. Our program will automatically capture the image output and visualize it in the frontend.

**Q: Why are some of my parameters considered as special parameters?**

A: For typing input data using text input boxes, only basic types such as str, int, float, list, etc., are allowed. Anything beyond this, requiring loading from data, is considered special parameters. 

We will look for corresponding parameters of the same type in the environment based on special types. This process is automatic. Therefore, please make every effort to ensure the accuracy of parameter types.

**Q: Why are some of my parameters considered as type `None`?**

A: You can adjust the parameter information by modifying parameters type like 
```bash
def function(param: int) -> float:
    """
    docstring placeholder here
    """
    function_body here
```
**Q: If I have many candidates for a particular parameter prediction, can it be automatically recognized by the system?**

A: We will visualize all candidates of parameters of the same type in the frontend for user selection.

**Q: How to use class type API? Do I need to initialize it to use their methods?**

A: You can directly request the methods or functions. Our system will automatically check if we have initialized this class instance before. If we have, we will skip the initialization; otherwise, we will proceed with the initialization.

**Q: How to input basic parameters like str, int, List?**

A: Just type them as how it calls in python programming. You can refer to the `./examples` for example usage.

**Q: Why does it only prompt me for certain parameters, and some parameters are not prompted?**

A: We currently only support required parameters (without default value), as there are so many optional parameters in some libs. It's recommended to only expose those required parameters which need to be inferred.

