Task_Description_of_Singletool_oneapi_Instructions = """
You are provided with one API function, its descriptions, the parameters and returns information required for each API function. 
Your task involves creating 5 varied, innovative, and detailed user queries that employ the given API function. 
The queries exemplify how to utilize the API call. A query should only use the given API. 
Additionally, you must incorporate the input parameters required for each API call. 
To achieve this, generate random information for required parameters according to its type.
The 5 queries should be very specific. 
Note that you shouldn't ask 'which API to use', rather, simply state your needs that can be addressed by these APIs. 
Never explicitly mentioning the specific calling of API functions in your response.
You should also avoid asking for the input parameters required by the API call, but instead directly provide the parameter in your query.
"""
Other_Requirements_singletool_oneapi = """
Please produce 5 queries in line with the given requirements and inputs. 
These 5 queries should display a diverse range of sentence structures: 
some queries should be in the form of imperative sentences, others declarative, and yet others interrogative. 
Equally, they should encompass a variety of tones, with some being polite, some being straightforward, some like layman.
Ensure they vary in length. 
Aim to include a number of engaging queries as long as they relate to API calls. 
Keep in mind that 
- Each query should consist of a minimum of twenty words. 
- Never explicitly mentioning the specific calling of API functions in your response. For example, never include API as 'xx.yy.zz(parameters)'.
- The response must be a list of effective json. 
- For quotation, use ` .
- Never include the reference paper in your response.
- Never including library keyword or specific type in sentence. Instead, use their description, you can find type description in return type description or parameters type description.
- Restricted to the response format: [{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"}]
"""

# simplified
Task_Description_of_Singletool_oneapi_Instructions_simple = """
You are provided with one API function, its descriptions, the parameters and returns information required for each API function. 
Your task involves creating 5 varied, innovative, and summarized user queries that employ the given API function. 
The queries exemplify how to utilize the API call. A query should only use the given API. 
Additionally, you must incorporate the input parameters required for each API call. 
To achieve this, generate random information for required parameters according to its type.
The 5 queries should be concised. 
Note that you shouldn't ask 'which API to use', rather, simply state your needs that can be addressed by these APIs. 
Never explicitly mentioning the specific calling of API functions in your response.
You should also avoid asking for the input parameters required by the API call, but instead directly provide the parameter in your query.
"""
Other_Requirements_singletool_oneapi_simple = """
Please produce 5 queries in line with the given requirements and inputs. 
These 5 queries should display a diverse range of sentence structures: 
some queries should be in the form of imperative sentences, others declarative, and yet others interrogative. 
Equally, they should encompass a variety of tones, with some being polite, some being straightforward, some like layman.
Ensure they vary in length. 
Aim to include a number of engaging queries as long as they relate to API calls. 
Keep in mind that 
- Each query should consist of a minimum of fifteen words. 
- Never explicitly mentioning the specific calling of API functions in your response. For example, never include API as 'xx.yy.zz(parameters)'.
- The response must be a list of effective json. 
- For quotation, use ` .
- Never include the reference paper in your response.
- Never including library keyword or specific type in sentence. Instead, use their description, you can find type description in return type description or parameters type description.
- Restricted to the response format: [{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"}]
"""
