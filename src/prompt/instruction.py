Task_Description_of_Singletool_oneapi_Instructions_whole = f"""
You are provided with the API function, its descriptions, the parameters required for each API function. 
Your task involves creating a total of 10 totally differentiate user queries for a given API function, 5 should be detailed and specific and other 5 brief and concise. Each query is innovative.
These queries illustrate only on how to accomplish the exact task that the API is designed for, and the user never intend to use API/function/tool to solve the task.  
Incorporate randomly generated values for required parameters, ensuring variation among queries based on their types.
Never explicitly mentioning any keywords of API function names in your response.
You should also avoid asking for the input parameters required by the API call, but instead directly provide the parameter in your query.
"""

Other_Requirements_singletool_oneapi_whole = f"""
Create queries in line with the given requirements and inputs. 
These queries should display a diverse range of sentence structures: 
some queries should be in the form of imperative sentences, others declarative, and yet others interrogative. 
Equally, they should encompass a variety of tones, with some being polite, some being straightforward, some like layman.
Ensure they vary in length. 
Aim to include a number of engaging queries as long as they relate to API calls. 
Keep in mind that 
- Queries should contain about 10-20 words, avoiding direct references to API calls (e.g., 'xx.yy.zz(parameters)'), library names (e.g., 'use API in xx'), function names (e.g., 'zz'), academic references (e.g., 'Zhang21 et al.'), or specific parameter types (e.g. 'AnnData data matrix object').
- Use backticks (`) for quotes, not single quotes (').
- Keep language natural and task-focused, avoiding overly technical or vague terms (e.g. 'data with all observations', 'using the given API').
- Ensure each query is distinct without repeating the same information (e.g. `based on the data object for the given data`).
- Queries should be structurally unique, with precise and diverse vocabulary.
- Format responses as a list in JSON: [{{"Query": "(query content)"}}, {{"Query": "(query content)"}}, {{"Query": "(query content)"}},{{"Query": "(query content)"}}, {{"Query": "(query content)"}}, {{"Query": "(query content)"}},{{"Query": "(query content)"}}, {{"Query": "(query content)"}}, {{"Query": "(query content)"}},{{"Query": "(query content)"}}].
"""
