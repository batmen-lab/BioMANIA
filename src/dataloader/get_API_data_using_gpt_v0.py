# [BIOAGENT]
from typing import List, Optional, Literal
from pydantic import BaseModel
import importlib
import inspect
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class Parameter(BaseModel):
    name: str
    type: Optional[str]
    default: Optional[str]
    optional: bool
    description: str

class Returns(BaseModel):
    type: Optional[str]
    description: str

class APIDefinition(BaseModel):
    Parameters: List[Parameter]
    Returns: Returns
    Docstring: str
    example: str
    api_type: Literal["function", "method", "class"]
    api_calling: str
    api_name: str

class APIDefinitions(BaseModel):
    api_definitions: List[APIDefinition]

MODULE_DATA = {
    "bigg": [
        "download",
        "genes",
        "metabolites",
        "models",
        "reactions",
        "search",
        "services",
        "version",
    ],
    # ...   
}

def get_API_data_extraction_prompt(api_list: list[str], module_documentation: str, module_code: str) -> str:
    api_list_bullets = "\n".join([f"- {api}" for api in api_list])

    return f"""
Instructions:
\"\"\"
- Given following Module Documentation and Module Code, extract the API definitions for the following functions/methods/classes in the API List.
- For each API, extract the following definition:
    - Parameters: List of dictionaries containing the following keys:
        - name: Name of the parameter
        - type: Python type of the parameter in string format if available or inferable from the document and the code, otherwise null. If the type is a custom class, use the class name in string format.
        - default: Default value of the parameter in string format if available, otherwise null
        - optional: true if the parameter is optional, otherwise false
        - description: Short description of the parameter
    Do not include self or cls in the parameters.
    - Returns: Dictionary containing the following keys:
        - type: Python type of the return value in string format if available or inferable from the document and the code, otherwise null. If the type is a custom class, use the class name in string format.
        - description: Short description of the return value
    - Docstring: Docstring of the function/method/class, reference the original docstring from the code if available, as well as descriptions from the documentation. Write it according to the Docstring Format provided below.
    - example: An example of how to call the API (same format as "api_calling" but replace $ with actual values) if provided in the documentation or the code, otherwise empty string.
    - api_type: "function" or "method" or "class",
    - api_calling: "<api_name>(<parameter1>=$, <parameter2>=$, ...)" (use "api_name" from below and "Parameters" from above, do not replace $ with actual values)
    - api_name: "bioservices.<module_name>.<function_name>" (function) or "bioservices.<class_name>.<method_name>" (method) or "bioservices.<class_name>" (class)
- Return the extracted API definitions in the following JSON format.
\"\"\"
---

Docstring Format:
\"\"\"
<description about the API>

Parameters:
-----------
<parameter1> : <type>
               <description>
<parameter2> : <type>
               <description>
...

Returns:
--------
<type>
    <description>

Examples:
--------
<one or more examples of how to call the API, annotate each line with ">>> " to indicate a Python code snippet>
\"\"\"
---

API List:
\"\"\"
{api_list_bullets}
\"\"\"
---

Module Documentation:
\"\"\"
{module_documentation}
\"\"\"
---

Module Code:
\"\"\"
{module_code}
\"\"\"
---

Output JSON Format:
\"\"\"
{{
    "api_definitions": [
        {{
            "Parameters": [
                {{
                    "name": str,
                    "type": str or null,
                    "default": str or null,
                    "optional": bool,
                    "description": str
                }},
                ...
            ],
            "Returns": {{
                "type": str or null,
                "description": str
            }},
            "Docstring": str,
            "example": str,
            "api_type": "function" or "method" or "class",
            "api_calling": str,
            "api_name": str
        }},
        ...
    ]
}}
\"\"\"
""".strip("\n")

def get_module_source(full_module_name: str) -> str:
    """
    Returns the source code of the specified module.

    Parameters:
        module_name (str): The fully-qualified name of the module 
                           (e.g., 'bioservices.uniprot').

    Returns:
        str: The source code of the module, or an error message if it cannot be retrieved.
    """
    try:
        # Dynamically import the module using its name.
        module = importlib.import_module(full_module_name)
    except ImportError as ie:
        return f"Error importing module: {ie}"

    try:
        # Retrieve and return the source code of the module.
        source_code = inspect.getsource(module)
        return source_code
    except OSError as ose:
        return f"Error retrieving source code: {ose}"

def fetch_section_content(url: str) -> str:
    """
    Fetches and returns the plain text content of a specific section of a readthedocs page,
    determined by the fragment identifier in the URL.
    
    Args:
        url (str): The full URL (including the fragment, e.g., 
                   'https://bioservices.readthedocs.io/en/main/references.html#module-bioservices.bigg').
    
    Returns:
        str: The plain text content of the section.
        
    Raises:
        ValueError: If the URL doesn't contain a fragment or if no matching section is found.
    """
    # Get the page content
    response = requests.get(url)
    response.raise_for_status()
    html = response.text

    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Extract fragment from URL (e.g., "module-bioservices.bigg")
    fragment = urlparse(url).fragment
    if not fragment:
        raise ValueError("The URL does not contain a fragment identifier.")
    
    # Locate the element with the corresponding id
    section = soup.find(id=fragment)
    if section is None:
        raise ValueError(f"No section found for fragment: {fragment}")

    # If the section element is a header (e.g., <h2>), collect it and its following siblings
    # until the next header of the same or higher level is encountered.
    if section.name.startswith('h') and section.name[1].isdigit():
        header_level = int(section.name[1])
        content_elements = [section]
        for sibling in section.find_next_siblings():
            if sibling.name and sibling.name.startswith('h') and sibling.name[1].isdigit():
                if int(sibling.name[1]) <= header_level:
                    break
            content_elements.append(sibling)
        # Join the text of all elements with spaces and newlines as needed.
        return "\n".join(elem.get_text(separator=" ", strip=True) for elem in content_elements)
    else:
        # Otherwise, just return the text of the found element.
        return section.get_text(separator=" ", strip=True)

def extract_API_data_using_gpt(module_name: str) -> APIDefinitions:
    api_list = MODULE_DATA[module_name]
    module_documentation = fetch_section_content(f"https://bioservices.readthedocs.io/en/main/references.html#module-bioservices.{module_name}")
    module_code = get_module_source(f"bioservices.{module_name}")
    prompt = get_API_data_extraction_prompt(api_list, module_documentation, module_code)
    print(prompt)

if __name__ == "__main__":
    module_name = "bigg"
    extract_API_data_using_gpt(module_name)
