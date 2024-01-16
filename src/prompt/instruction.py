def Task_Description_of_Singletool_oneapi_Instructions_template(detailed_summarized, could_must, specific_concised):
    return f"""
You are provided with one API function, its descriptions, the parameters and returns information required for each API function. 
Your task involves creating 5 varied, innovative, and {detailed_summarized} user queries that employ the given API function. 
The queries exemplify how to utilize the API call. A query should only use the given API. 
Additionally, you {could_must} incorporate the input parameters required for each API call. 
To achieve this, generate random information for required parameters according to its type.
The 5 queries should be very {specific_concised}. 
Note that you shouldn't ask 'which API to use', rather, simply state your needs that can be addressed by these APIs. 
Never explicitly mentioning the specific calling of API functions in your response.
You should also avoid asking for the input parameters required by the API call, but instead directly provide the parameter in your query.
"""

Task_Description_of_Singletool_oneapi_Instructions = Task_Description_of_Singletool_oneapi_Instructions_template("detailed", "could", "specific")

Task_Description_of_Singletool_oneapi_Instructions_simple = Task_Description_of_Singletool_oneapi_Instructions_template("summarized", "could", "concised")


def Other_Requirements_singletool_oneapi_template(word_minimum_number):
    return f"""
Please produce 5 queries in line with the given requirements and inputs. 
These 5 queries should display a diverse range of sentence structures: 
some queries should be in the form of imperative sentences, others declarative, and yet others interrogative. 
Equally, they should encompass a variety of tones, with some being polite, some being straightforward, some like layman.
Ensure they vary in length. 
Aim to include a number of engaging queries as long as they relate to API calls. 
Keep in mind that 
- Queries should be around {word_minimum_number} words, and avoid explicit mentions of API calls like 'xx.yy.zz(parameters)', or PyPI lib name as 'use API in xx', or reference paper like 'Zhang21 et al.', or API function keywords, or specific parameters type.
- For quotation, use ` instead of '.
- Avoid technical terms like 'the given API' or unmeaningful terms like 'data with all observations' in your inquiry; keep it natural and focus on the user's intention to accomplish a real task.
- Avoiding repeated the same information within each inquiry, like "based on the data object for the given data".
- Queries should be unique in structure and phrasing, and vocabulary should be varied precise, accurate, and diverse. 
- Restricted to the response format as a list of effective json: [{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"},{"Query": "(query content)"}]
"""

Other_Requirements_singletool_oneapi_simple = Other_Requirements_singletool_oneapi_template("fifteen")

Other_Requirements_singletool_oneapi = Other_Requirements_singletool_oneapi_template("twenty")