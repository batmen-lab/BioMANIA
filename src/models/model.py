from configs.model_config import *
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import openai
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import torch
from peft import PeftModel,get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training


def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

def LLM_model(local=True):
    """
    https://python.langchain.com/docs/modules/model_io/models/llms/integrations/openai
    https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_hub
    https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
    """
    if llm_model_dict[LLM_MODEL]['platform']=='OPENAI':
        llm = OpenAI(temperature=TEMPERATURE,model_name='gpt-3.5-turbo-16k')
        tokenizer = None
    elif llm_model_dict[LLM_MODEL]['platform']=='GORILLA':
        openai.api_key = "EMPTY" # Key is ignored and does not matter
        openai.api_base = "http://34.132.127.197:8000/v1"
        llm = ChatOpenAI(model_name=LLM_MODEL, temperature=TEMPERATURE)
        tokenizer = None
    elif llm_model_dict[LLM_MODEL]['platform']=='HUGGINGFACE':
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        if local:
            llm = AutoModel.from_pretrained(llm_model_dict[LLM_MODEL]['LOCALPATH'], trust_remote_code=True).half().to(LLM_DEVICE)
        else:
            llm = AutoModel.from_pretrained(LLM_MODEL, trust_remote_code=True, cache_dir=HUGGINGPATH).half().to(LLM_DEVICE)
        llm = llm.eval()
    elif llm_model_dict[LLM_MODEL]['platform']=='HUGGINGFACEHUB':
        # TODO: not supported by langchain yet
        llm = HuggingFaceHub(repo_id=LLM_MODEL, model_kwargs={"temperature": TEMPERATURE, "max_length": 2000})
        tokenizer = None
    elif llm_model_dict[LLM_MODEL]['platform']=='PEFT' and LLM_MODEL=='QLoRA':
        # QLora: https://colab.research.google.com/drive/17XEqL1JcmVWjHkT-WczdYkJlNINacwG7?usp=sharing#scrollTo=2QK51MtdsMLu
        model_name = "huggyllama/llama-7b"
        adapters_name = 'timdettmers/guanaco-7b'
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
        llm = PeftModel.from_pretrained(model, adapters_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif llm_model_dict[LLM_MODEL]['platform']=='PEFT' and LLM_MODEL=='llama-2-7b-chat-hf':
        tokenizer = LlamaTokenizer.from_pretrained(llm_model_dict[LLM_MODEL]['LOCALPATH'])
        model = LlamaForCausalLM.from_pretrained(
            llm_model_dict[LLM_MODEL]['LOCALPATH'],
            load_in_8bit=True,
            device_map='auto',
            torch_dtype=torch.float16 if FP16 else torch.float,
        )
        llm, lora_config = create_peft_config(model)
    else:
        raise NotImplementedError
    print('Using model: ', LLM_MODEL)
    return llm, tokenizer

def LLM_response(llm,tokenizer,chat_prompt,history=[],kwargs={}):
    """
    get response from LLM
    """
    if llm_model_dict[LLM_MODEL]['platform'] in ['OPENAI']:
        response = llm.predict(chat_prompt)
        history.append([chat_prompt, response])
    elif llm_model_dict[LLM_MODEL]['platform'] in ['HUGGINGFACE']:
        response, history = llm.chat(tokenizer, chat_prompt, history=history)
    elif llm_model_dict[LLM_MODEL]['platform']=='PEFT':
        model_input = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
        llm.eval()
        with torch.no_grad():
            response = tokenizer.decode(llm.generate(**model_input, max_new_tokens=MAX_NEW_TOKENS)[0], skip_special_tokens=False) # [len(model_input['input_ids']):]
        response = response[len(chat_prompt):]
        history.append([chat_prompt, response])
    else:
        raise NotImplementedError
    return response, history

def embedding_model():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return embeddings
