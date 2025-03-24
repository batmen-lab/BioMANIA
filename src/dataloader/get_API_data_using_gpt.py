# [BIOAGENT]
import os
import importlib
import inspect
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from pydantic import BaseModel
from urllib.parse import urlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..gpt.gpt_updated_interface import query_structured_output_openai
from ..gpt.utils import save_json

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
    api_type: Literal["function", "method", "class", "other"]
    api_name: str
    api_calling: str
    example: str

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
    "biocontainers": [
        "get_stats",
        "get_tools",
        "get_versions_one_tool",
        "services",
    ],
    "biodbnet": [
        "db2db",
        "dbFind",
        "dbOrtho",
        "dbReport",
        "dbWalk",
        "getDirectOutputsForInput",
        "getInputs",
        "getOutputsForInput",
        "services",
    ],
    "biogrid": [
        "biogrid",
        "exP",
        "query",
        "searchString",
        "taxId",
    ],
    "biomodels": [
        "add_attribute_to_xml",
        "add_dataset_to_xml",
        "add_filter_to_xml",
        "attributes",
        "cert",
        "clear_cache",
        "configuration",
        "content_types",
        "create_attribute",
        "create_filter",
        "custom_query",
        "databases",
        "datasets",
        "debug_message",
        "delete_cache",
        "delete_one",
        "devtools",
        "displayNames",
        "easyXML",
        "easyXMLConversion",
        "filters",
        "getUserAgent",
        "get_async",
        "get_datasets",
        "get_headers",
        "get_one",
        "get_sync",
        "get_xml",
        "host",
        "hosts",
        "http_delete",
        "http_get",
        "http_post",
        "http_put",
        "last_response",
        "logging",
        "lookfor",
        "marts",
        "name",
        "names",
        "new_query",
        "on_web",
        "post_one",
        "proxies",
        "pubmed",
        "query",
        "registry",
        "requests_per_sec",
        "response_codes",
        "save_str_to_image",
        "session",
        "settings",
        "url",
        "valid_attributes",
        "version",
    ],
    "chebi": [
        "conv",
        "devtools",
        "easyXML",
        "easyXMLConversion",
        "getAllOntologyChildrenInPath",
        "getCompleteEntity",
        "getCompleteEntityByList",
        "getLiteEntity",
        "getOntologyChildren",
        "getOntologyParents",
        "getStructureSearch",
        "getUpdatedPolymer",
        "logging",
        "name",
        "on_web",
        "pubmed",
        "requests_per_sec",
        "response_codes",
        "save_str_to_image",
        "serv",
        "settings",
        "suds",
        "url",
        "wsdl_create_factory",
        "wsdl_methods",
        "wsdl_methods_info",
    ],
    "chembl": [
        "compounds2accession",
        "format",
        "get_ATC",
        "get_activity",
        "get_approved_drugs",
        "get_assay",
        "get_binding_site",
        "get_biotherapeutic",
        "get_cell_line",
        "get_chembl_id_lookup",
        "get_compound_record",
        "get_compound_structural_alert",
        "get_document",
        "get_document_similarity",
        "get_document_term",
        "get_drug",
        "get_drug_indication",
        "get_go_slim",
        "get_image",
        "get_mechanism",
        "get_metabolism",
        "get_molecule",
        "get_molecule_form",
        "get_organism",
        "get_similarity",
        "get_source",
        "get_status",
        "get_status_resources",
        "get_substructure",
        "get_target",
        "get_target_component",
        "get_target_prediction",
        "get_target_relation",
        "get_tissue",
        "get_xref_source",
        "order_by",
        "page_meta",
        "search_activity",
        "search_assay",
        "search_chembl_id_lookup",
        "search_document",
        "search_molecule",
        "search_target",
        "services",
    ],
    "cog": [
        "get_all_cogs_definition",
        "get_cog_definition_by_cog_id",
        "get_cog_definition_by_name",
        "get_cogs",
        "get_cogs_by_assembly_id",
        "get_cogs_by_category",
        "get_cogs_by_category_id",
        "get_cogs_by_gene",
        "get_cogs_by_id",
        "get_cogs_by_id_and_category",
        "get_cogs_by_id_and_organism",
        "get_cogs_by_organism",
        "get_cogs_by_protein_name",
        "get_cogs_by_taxon_id",
        "get_taxonomic_categories",
        "get_taxonomic_category_by_name",
        "search_organism",
        "services",
        "show_progress",
    ],
    "dbfetch": [
        "fetch",
        "get_all_database_info",
        "get_database_format_styles",
        "get_database_formats",
        "get_database_info",
        "services",
        "supported_databases",
    ],
    "ena": [
        "data_warehouse",
        "get_data",
        "get_taxon",
        "services",
        "url",
    ],
    "ensembl": [
        # No APIs provided
    ],
    "eutils": [
        "ECitMatch",
        "EFetch",
        "EGQuery",
        "EInfo",
        "ELink",
        "EPost",
        "ESearch",
        "ESpell",
        "ESummary",
        "databases",
        "email",
        "help",
        "parse_xml",
        "services",
        "snp_summary",
        "taxonomy_summary",
        "tool",
    ],
    "hgnc": [
        "fetch",
        "get_info",
        "search",
        "searchable_fields",
        "services",
        "stored_fields",
    ],
    "intact_complex": [
        "details",
        "search",
        "services",
    ],
    "kegg": [
        # No APIs provided
    ],
    "ncbiblast": [
        "checkInterval",
        "databases",
        "get_parameter_details",
        "get_parameters",
        "get_result",
        "get_result_types",
        "get_status",
        "parameters",
        "run",
        "services",
        "wait",
    ],
    "omnipath": [
        "get_about",
        "get_info",
        "get_interactions",
        "get_network",
        "get_ptms",
        "get_resources",
        "services",
    ],
    "pathwaycommons": [
        "default_extension",
        "easyXMLConversion",
        "get",
        "get_sifgraph_common_stream",
        "get_sifgraph_neighborhood",
        "get_sifgraph_pathsbetween",
        "get_sifgraph_pathsfromto",
        "graph",
        "search",
        "services",
        "top_pathways",
        "traverse",
    ],
    "pdbe": [
        "get_assembly",
        "get_binding_sites",
        "get_drugbank_annotation",
        "get_electron_density_statistics",
        "get_experiment",
        "get_files",
        "get_functional_annotation",
        "get_ligand_monomers",
        "get_modified_residues",
        "get_molecules",
        "get_mutated_residues",
        "get_nmr_resources",
        "get_observed_ranges",
        "get_observed_ranges_in_pdb_chain",
        "get_observed_residues_ratio",
        "get_related_dataset",
        "get_related_publications",
        "get_release_status",
        "get_residue_listing",
        "get_residue_listing_in_pdb_chain",
        "get_secondary_structure",
        "get_summary",
        "services",
    ],
    "pride": [
        "get_peptide_evidence",
        "get_project",
        "get_project_files",
        "get_projects",
        "get_projects_count",
        "get_protein_evidences",
        "get_stats",
        "services",
    ],
    "psicquic": [
        "activeDBs",
        "buffer",
        "convert",
        "convertAll",
        "formats",
        "getInteractionCounter",
        "getName",
        "knownName",
        "mappingOneDB",
        "postCleaning",
        "postCleaningAll",
        "preCleaning",
        "print_status",
        "query",
        "queryAll",
        "read_registry",
        "registry",
        "registry_actives",
        "registry_counts",
        "registry_names",
        "registry_restexamples",
        "registry_restricted",
        "registry_resturls",
        "registry_soapurls",
        "registry_versions",
        "services",
        "uniprot",
    ],
    "quickgo": [
        "Annotation",
        "Annotation_from_goid",
        "gene_product_search",
        "get_go_ancestors",
        "get_go_chart",
        "get_go_children",
        "get_go_paths",
        "get_go_terms",
        "go_search",
        "services",
    ],
    "reactome": [
        "debugLevel",
        "get_complex_subunits",
        "get_complexes",
        "get_discover",
        "get_diseases",
        "get_diseases_doid",
        "get_entity_componentOf",
        "get_entity_otherForms",
        "get_event_ancestors",
        "get_eventsHierarchy",
        "get_exporter_diagram",
        "get_exporter_fireworks",
        "get_exporter_reaction",
        "get_exporter_sbml",
        "get_interactors_psicquic_molecule_details",
        "get_interactors_psicquic_molecule_summary",
        "get_interactors_psicquic_resources",
        "get_interactors_static_molecule_details",
        "get_interactors_static_molecule_pathways",
        "get_interactors_static_molecule_summary",
        "get_mapping_identifier_pathways",
        "get_mapping_identifier_reactions",
        "get_pathway_containedEvents",
        "get_pathway_containedEvents_by_attribute",
        "get_pathways_low_diagram_entity",
        "get_pathways_low_diagram_entity_allForms",
        "get_pathways_low_entity",
        "get_pathways_low_entity_allForms",
        "get_pathways_top",
        "get_references",
        "get_species_all",
        "get_species_main",
        "name",
        "search_facet",
        "search_facet_query",
        "search_query",
        "search_spellcheck",
        "search_suggest",
        "services",
        "version",
    ],
    "rhea": [
        "get_metabolites",
        "query",
        "search",
        "services",
    ],
    "unichem": [
        "get_all_src_ids",
        "get_compounds",
        "get_connectivity",
        "get_id_from_name",
        "get_images",
        "get_inchi_from_inchikey",
        "get_source_info_by_id",
        "get_source_info_by_name",
        "get_sources",
        "get_sources_by_inchikey",
        "get_sources_by_inchikey_verbose",
        "get_structure",
        "services",
        "source_ids",
    ],
    "uniprot": [
        "get_df",
        "get_fasta",
        "mapping",
        "quick_search",
        "retrieve",
        "search",
        "services",
        "uniref",
        "valid_mapping",
    ],
    "wikipathways": [
        "createPathway",
        "findInteractions",
        "findPathwaysByLiterature",
        "findPathwaysByText",
        "findPathwaysByXref",
        "getColoredPathway",
        "getCurationTags",
        "getCurationTagsByName",
        "getOntologyTermsByPathway",
        "getPathway",
        "getPathwayAs",
        "getPathwayHistory",
        "getPathwayInfo",
        "getPathwaysByOntologyTerm",
        "getPathwaysByParentOntologyTerm",
        "getRecentChanges",
        "listOrganisms",
        "listPathways",
        "login",
        "organism",
        "organisms",
        "removeCurationTag",
        "saveCurationTag",
        "savePathwayAs",
        "services",
        "showPathwayInBrowser",
        "updatePathway",
    ],
}

MODULE_NAME_TO_CLASS_NAME = {
    "bigg": "BiGG",
    "biocontainers": "BioContainers",
    "biodbnet": "BioDBNet",
    "biogrid": "BioGRID",
    "biomodels": "BioModels",
    "chebi": "ChEBI",
    "chembl": "ChEMBL",
    "cog": "COG",
    "dbfetch": "DbFetch",
    "ena": "ENA",
    "ensembl": "Ensembl",
    "eutils": "EUtils",
    "hgnc": "HGNC",
    "intact_complex": "IntActComplex",
    "kegg": "KEGG",
    "ncbiblast": "NCBIBlast",
    "omnipath": "OmniPath",
    "pathwaycommons": "PathwayCommons",
    "pdbe": "PDBe",
    "pride": "PRIDE",
    "psicquic": "PSICQUIC",
    "quickgo": "QuickGO",
    "reactome": "Reactome",
    "rhea": "Rhea",
    "unichem": "UniChem",
    "uniprot": "UniProt",
    "wikipathways": "WikiPathways",
}

def get_API_data_extraction_prompt(api_name: str, module_name: str, module_documentation: str, module_code: str) -> str:
    return f"""
Instructions:
\"\"\"
- Given following Module Documentation and Module Code, extract the API definition for "{api_name}" API from module "{module_name}".
- Parameters: List of dictionaries containing the following keys:
    - name: Name of the parameter
    - type: Python type of the parameter in string format if available or inferable from the document and the code, otherwise null. If the type is a custom class, use the class name in string format.
    - default: Default value of the parameter in string format if available, otherwise null. If the default value is None, use "None".
    - optional: true if the parameter is optional, otherwise false
    - description: Short description of the parameter. If the parameter only accepts a fixed set of values, list them in the description.
Do not include self or cls in the parameters.
- Returns: Dictionary containing the following keys:
    - type: Python type of the return value in string format if available or inferable from the document and the code, otherwise null. If the type is a custom class, use the class name in string format.
    - description: Short description of the return value
- Docstring: Docstring of the function/method/class, reference the original docstring from the code if available, as well as descriptions from the documentation. Write it according to the Docstring Format provided below.
- api_type: "function" or "method" or "class" or "other" (class attribute, imported API, independent variable, etc.),
- api_name: "bioservices.<module_name>.<function_name>" (function) or "bioservices.<class_name>.<method_name>" (method) or "bioservices.<class_name>" (class)
- api_calling: "<api_name>(<parameter1>=$, <parameter2>=$, ...)" (use "api_name" from below and "Parameters" from above, do not replace $ with actual values)
- example: "<api_name>(<parameter1>=$, <parameter2>=$, ...)" (use "api_name" from below and "Parameters" from above, replace $ with actual values) if an example of how to call the API is provided in the documentation or the code, otherwise empty string.
- Return the extracted API definition in the following JSON format.
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
    "api_type": "function" or "method" or "class" or "other",
    "api_name": str
    "api_calling": str,
    "example": str,
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

def extract_and_save_API_data(module_names: list[str]):
    # Dictionary to store API results grouped by module.
    results = {}

    # List to hold all tasks: each task is a tuple (module_name, api_name, prompt).
    tasks = []
    for module_name in module_names:
        # Pre-fetch module documentation and code once per module.
        try:
            module_documentation = fetch_section_content(
                f"https://bioservices.readthedocs.io/en/main/references.html#module-bioservices.{module_name}"
            )
        except:
            print(f"Error fetching documentation for module: {module_name}")
            module_documentation = ""
        module_code = get_module_source(f"bioservices.{module_name}")
        # Prepare a sub-dictionary for this module.
        results[module_name] = {}
        # Loop over API names as defined in MODULE_DATA, skipping "services".
        for api_name in MODULE_DATA[module_name]:
            if api_name == "services":
                continue
            prompt = get_API_data_extraction_prompt(api_name, module_name, module_documentation, module_code)
            tasks.append((module_name, api_name, prompt))

    # Execute all API extraction tasks concurrently.
    with ThreadPoolExecutor(max_workers=4) as executor: # Adjust max_workers as needed.
        future_to_task = {
            executor.submit(
                query_structured_output_openai,
                prompt,
                data_model=APIDefinition,
                model='gpt-4o-2024-11-20'
                # model='gpt-4o-mini-2024-07-18'
            ): (module_name, api_name)
            for module_name, api_name, prompt in tasks
        }
        # As tasks complete, store their results.
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task)):
            module_name, api_name = future_to_task[future]
            try:
                results[module_name][api_name] = future.result()
            except Exception as e:
                results[module_name][api_name] = None  # Or handle the exception as needed.

    # Re-order API definitions per module to match the original order.
    ordered_results = {}
    for module_name in module_names:
        ordered_api_definitions = {}
        for api_name in MODULE_DATA[module_name]:
            if api_name == "services":
                continue
            ordered_api_definitions[api_name] = results[module_name].get(api_name)
        ordered_results[module_name] = ordered_api_definitions

    OUTPUT_DIR = os.path.join('data', 'standard_process', 'bioservices')
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "API_data.json")
    save_json(OUTPUT_FILE, ordered_results)

if __name__ == "__main__":
    num_apis = sum([len(MODULE_DATA[module_name]) for module_name in MODULE_DATA])
    print(f"Total number of APIs: {num_apis}")
    module_names = list(MODULE_DATA.keys())
    extract_and_save_API_data(module_names)

