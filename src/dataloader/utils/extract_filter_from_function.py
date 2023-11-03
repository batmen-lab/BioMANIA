import json
from base_graph_ast import *

from configs.model_config import *
from model.model import *
from prompt.prompt import *

class JSONFilter:
    """
    Initializes a JSON filter class with a strategy
    """
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def filter_keys(self, json_data):
        return self.strategy.filter_keys(json_data)

class NonUnderscoreFilterStrategy:
    """
    Remove keys that start with an underscore.
    """
    def filter_keys(self, json_data):
        filtered_data = {}
        for key, value in json_data.items():
            if not key.startswith('_'):
                filtered_data[key] = value
        return filtered_data

class IndegreeSortFilterStrategy:
    """
    Remove keys with indegrees above a threshold by loading precomputed indegree values from a JSON file.
    """
    def __init__(self):
        compute_degree_graph()
        with open(os.path.join(LIB_ANALYSIS_PATH,'sorted_nodes.json'), 'r') as file:
            self.indegree_json = json.load(file)
        print('sorted_nodes.json loaded succesfully!')
    
    def filter_keys(self, json_data, indegree_threshold=10):
        filtered_data = {}
        for key in json_data:
            if key in self.indegree_json and self.indegree_json[key]['in_degree'] <= indegree_threshold:
                filtered_data[key] = json_data[key]
        return filtered_data

class StrContainFilterStrategy:
    """
    Filters keys shown in the content
    """
    def filter_keys(self, json_data, content):
        filtered_data = {}
        for key, value in json_data.items():
            if key in content:
                filtered_data[key] = value
        return filtered_data

class WebAPIFilterStrategy:
    """
    Filters keys appeared in web API
    """
    def __init__(self, ):
        # load html, groundtruth
        content = process_html(API_HTML_PATH)
        # process APIs
        ori_content_keys = list(set([i for i in content.split(' ') if (LIB_ALIAS in i) and ('.' in i) and (not i.split('.')[-1].isupper())]))#  
        self.web_API = {}
        for item in ori_content_keys:
            key = item.split(".")[-1]
            if key not in self.web_API:
                self.web_API[key] = []
            self.web_API[key].append(item)

    def filter_keys(self, json_data):
        filtered_data = {}
        for key, value in json_data.items():
            if key in self.web_API:
                filtered_data[key] = value
        return filtered_data

class LLMFilterStrategy:
    """
    Use LLM to find API answer with given function keyword
    """
    def __init__(self, ):
        # Read the JSON data
        with open(os.path.join(LIB_ANALYSIS_PATH,'API_func.json'), 'r') as f:
            data = json.load(f)
        print(len(data),' func detected!')

        strategy = IndegreeSortFilterStrategy() 
        json_filter = JSONFilter(strategy)
        filtered_data = json_filter.filter_keys(data)
        print('After step1, ', len(filtered_data),' func detected!')

        # load index html
        content = process_html(API_HTML_PATH)
        self.content_list = split_string_with_limit(content, limit=500)
        print('split html content into ', len(content_list), ' contents!')
        # Step2: check if item is in html
        strategy = StrContainFilterStrategy() 
        json_filter = JSONFilter(strategy)
        self.filtered_data = json_filter.filter_keys(filtered_data)
        print('After step2, ', len(filtered_data),' func detected!')
        
        # LLM and prompt
        self.llm, self.tokenizer = LLM_model()
        self.chat_prompt = Factory_prompt_json("askfullAPI")
        
    def filter_keys(self, json_data):
        """
        Filters keys present in LLM data from the input json data
        """
        self.run_llm()
        LLM_data = self.clean_from_llmanswer(self.answer_api_available)
        # assume LLM_data is True here
        filtered_data = {}
        for key, value in json_data.items():
            if key in LLM_data:
                filtered_data[key] = value
        return filtered_data
    
    def run_llm(self,):
        """
        Runs LLM API for each data and stores the output in a JSON file
        """
        all_func = list(self.filtered_data.keys())
        self.answer_api_available = {}
        for key in all_func:
            self.answer_api_available[key] = []
            for content in self.content_list:
                if key not in content:
                    continue
                kwargs = {"API":key,"content":content}
                print(f'Can you help to find the full command of keyword {key}?')
                response, history = LLM_response(self.llm,self.tokenizer,self.chat_prompt,history=[],kwargs=kwargs)
                print('Agent:',response)
                self.answer_api_available[key].append(response)
        with open(os.path.join(LIB_ANALYSIS_PATH,'askLLM_API.json'), 'w') as f:
            json.dump(self.answer_api_available, f, indent=4)
    
    def clean_from_llmanswer(self,json_data):
        """
        Removes data with keywords of same key name from json data
        """
        LLM_data = {}
        for key, value_list in json_data.items():
            filtered_list = []
            for value in value_list:
                content = value.split('[')[-1].split(']')[0].strip()
                last_token = content.split('.')[-1]
                if last_token == key:
                    filtered_list.append(value)
            if filtered_list:
                filtered_data[key] = filtered_list
        return LLM_data

def process_html(html_path):
    """
    Loads HTML file content after removing excessive spaces.
    """
    webcontent = BSHTMLLoader(html_path).load()
    content = ' '.join([i.page_content for i in webcontent])
    # To remove large blocks of whitespace without removing spaces between words, ensuring the shortest possible input for LLM.
    content = re.sub(r'\s+', ' ', content)
    return content

def split_string_with_limit(string, limit):
    """
    # Splits a string into chunks of words not exceeding the limit length
    """
    words = string.split()
    chunks = []
    current_chunk = ""
    current_count = 0
    for word in words:
        if current_count + len(word) > limit:
            chunks.append(current_chunk.strip())
            current_chunk = "" 
            current_count = 0
        current_chunk += word + " "
        current_count += len(word) + 1
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


if __name__=='__main__':
    # load json
    with open(os.path.join(LIB_ANALYSIS_PATH,'API_func.json'), 'r') as file:
        data = json.load(file)
    strategy = IndegreeSortFilterStrategy() 
    json_filter = JSONFilter(strategy)
    filtered_data = json_filter.filter_keys(data)
    # save json
    with open(os.path.join(LIB_ANALYSIS_PATH,'API_func_filtered.json'), 'w') as file:
        file.write(json.dumps(filtered_data, indent=4))
