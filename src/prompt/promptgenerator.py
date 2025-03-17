"""
Author: Zhengyuan Dong
Date Created: January 16, 2024
Last Modified: May 22, 2024
Description: aggregate prompts for all tasks
Email: zydong122@gmail.com
"""

from abc import ABC, abstractmethod

class PromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, *args):
        pass

class CompositeDocstringPromptBuilder(PromptBuilder):
    # from prompt/composite.py
    def build_prompt(self, API_description, func_inputs, func_outputs, description_text):
        return f"""
Write a concise docstring for an invisible function in Python, focusing solely on its core functionality derived from the sequential composition of sub APIs.
- API Description: {API_description}
- Parameters: {func_inputs}
- Returns: {func_outputs}
- Additional Description: {description_text}
Craft a 1-2 sentence docstring that extracts and polishes the core information. The response should be in reStructuredText format, excluding specific API names and unprofessional terms. Remember to use parameter details only to refine the core functionality explanation, not for plain input/output information.
"""

class CompositeNamePromptBuilder(PromptBuilder):
    # from prompt/composite.py
    def build_prompt(self, sub_API_names, llm_docstring):
        return f"""Your task is to suggest an appropriate name for the given invisible function:
- Here are the sub API used together with function's docstring, please consider the API name to generate function name. sub API names: {sub_API_names}, 
function docstring: ```{llm_docstring}```
- Your name should consist of 4 to 5 keywords that combined with `_`, name should be recognizable and contain as much information as you can in keywords, and should display API information in a sequential order.
Your Response format: {{'func_name': (your designed function name)}}
Please do not include other information except for response format.
"""

class InstructionGenerationPromptBuilder(PromptBuilder):
    # [BIOAGENT]
    def build_prompt(self, API_name, docs):
        """
        v0:
        Prompt No.1:
        Instruction: Generate 10 examples that uses the function {API_name}(...) to train an intelligent code assistant for computational biology.
        Each example should be a JSON dictionary with fields "instruction" and "code".
        The text field is a single line of language instruction the user would say to the code assistant.
        The code field is a single line of code the code assistant should generate given the user's instruction.
        Use one line per example. Be specific when writing the user instruction - write it as if a human user would say it to the assistant.
        Never include function name or parameters name. Vary tones among queries: polite, straightforward, casual.
        Emphasize intent and methods of the function and distinguish between Target and Contrasting descriptions.
        For example, given Target description "Scatter plot in UMAP basis" Contrasting description ["Scatter plot in PHATE basis"], generated examples `Please create a scatter plot in the UMAP basis`should include intent of "create a scatter plot" and details "in the UMAP basis", omitting "PHATE specifics".
        ---

        Prompt No.2:
        Refer to the following code documentation on what each function does, and the distinctions between Target and Contrasting descriptions.
        docstring of {docs}
        """

        bioservices_modules = {
            "arrayexpress": "The `bioservices.arrayexpress` module is part of the `bioservices` Python package and is designed for programmatic access to the ArrayExpress database. ArrayExpress is a public repository of high-throughput functional genomics data, including transcriptomics, epigenomics, and proteomics experiments. The module allows users to query and retrieve metadata and experimental datasets from ArrayExpress, facilitating bioinformatics research and analysis of gene expression and other omics data.",
            "bigg": "The `bioservices.bigg` module is part of the BioServices library, a Python package designed for interacting with various biological databases and services. The `bioservices.bigg` module provides tools to access and interact with the BiGG (Biochemical Genetic and Genomic knowledgebase) database. BiGG focuses on genome-scale metabolic models and provides data on metabolites, reactions, and genome-scale networks in systems biology. This module allows users to query and retrieve information related to metabolic pathways, reactions, and metabolites in an easy and programmatic way.",
            "biocontainers": "The `bioservices.biocontainers` module is part of the BioServices library, which is designed to simplify access to online biological databases and services. Specifically, the `biocontainers` module provides programmatic access to the BioContainers registry. BioContainers is a community-driven project offering standardized and reusable software containers for bioinformatics. \n\nThis module allows users to query and retrieve information about bioinformatics tools and their corresponding containerized versions, leveraging container technologies such as Docker and Singularity. It handles metadata about software, including container details like version, authorship, and links to containerized tools stored in registries such as DockerHub and Quay.io. It is useful for automating workflows in bioinformatics by identifying and utilizing containerized software environments.",
            "biodbnet": "The `bioservices.biodbnet` module is part of the BioServices library in Python and is designed to provide easy access to the **BioDBnet** web service, which is a platform for database identifier conversions across various biological databases. This module handles data related to biological database identifiers, such as gene or protein identifiers, and allows users to map, convert, or retrieve data about identifiers across multiple databases (e.g., NCBI, UniProt, Ensembl). It is particularly useful for researchers needing to integrate or analyze data from heterogeneous biological sources.",
            "biogrid": "The `bioservices.biogrid` module in the BioServices library is designed to provide programmatic access to the **BioGRID database**. BioGRID is a biological database that specializes in storing and providing interaction data, particularly for protein-protein and genetic interactions across multiple organisms. This module enables users to query and retrieve interaction datasets, facilitating computational analysis and integration with other bioinformatics workflows.",
            "biomart": "The `bioservices.biomart` module is part of the BioServices Python library, which provides access to *BioMart*, a widely used web service for retrieving large datasets from biological databases. This module allows Python users to programmatically interact with BioMart, enabling tasks such as querying and extracting biological data, including gene annotations, protein information, genomic features, and other datasets related to genomics, proteomics, and functional genomics. It is particularly useful for managing and analyzing biological data across various species and data sources in a structured, efficient manner.",
            "biomodels": "The `bioservices.biomodels` module in the BioServices library provides programmatic access to the BioModels database. BioModels is a repository of computational models related to biological processes. The module allows users to interact with the database and retrieve various types of data, including curated models of biochemical and cellular systems. It supports formats like SBML (Systems Biology Markup Language) and offers tools for querying, searching, and downloading models for computational biology research.",
            "chebi": "The `bioservices.chebi` module is part of the BioServices package, which provides a programmatic interface to access data from biological databases. The `bioservices.chebi` module specifically works with ChEBI (Chemical Entities of Biological Interest), a database of molecular entities focused on 'small' chemical compounds. It allows users to retrieve and interact with data about chemical compounds, such as chemical names, molecular structures, synonyms, ontology classifications, and other annotations. This module is particularly useful for researchers needing to programmatically access chemical and biochemical information for computational biology and chemistry tasks.",
            "chembl": "The `bioservices.chembl` module from the BioServices library provides a programmatic interface to the **ChEMBL database**, a resource that contains bioactivity data for drug discovery. This module allows users to query ChEMBL\u2019s comprehensive data on bioactive molecules, drug targets, ligand-receptor interactions, assays, and more. It is particularly useful for retrieving chemical properties, pharmacological information, and molecular structures relevant to pharmaceutical research and development.",
            "cog": "The `bioservices.cog` module in the BioServices library is designed to facilitate interaction with the Comparative Genomics (COG) database. The COG database organizes orthologous groups of proteins and functional categories derived from microbial genomes. This module allows users to retrieve data related to orthologous relationships, protein annotations, and functional classifications, aiding in genomic analysis and comparative studies. It primarily handles data related to gene families, protein functions, and evolutionary relationships.",
            "dbfetch": "The `bioservices.dbfetch` module is a part of the BioServices package, which allows programmatic access to various bioinformatics web services. Specifically, the `dbfetch` module is used to interact with the **EBI (European Bioinformatics Institute) DBFetch service**, enabling users to retrieve data from various biological databases. It handles a wide range of biological data types, such as nucleotide and protein sequences, genomic annotations, and other database entries from resources like UniProt, Ensembl, and EMBL. This module is particularly useful for fetching database records in specific formats (e.g., FASTA, XML, TXT) for downstream analysis.",
            "ena": "The `bioservices.ena` module is part of the BioServices library, which provides programmatic access to various biological data services. Specifically, the `ena` submodule interfaces with the European Nucleotide Archive (ENA), a comprehensive repository for nucleotide sequences and associated metadata. This module allows users to query and retrieve nucleotide sequence data, such as raw sequencing read data, assembled genomes, and transcriptomes, as well as metadata related to experimental projects, samples, and sequences. It is particularly useful for bioinformaticians and researchers working with genomics and transcriptomics data.",
            "ensembl": "The **`bioservices.ensembl`** module is part of the BioServices library, which provides access to various biological web services. The `bioservices.ensembl` module specifically facilitates programmatic access to the Ensembl REST API, which offers a wide range of genomic data. This data includes information about genes, transcripts, variants, regulatory features, sequences, and comparative genomics. Researchers can use this module to query the Ensembl database for species-specific genomic annotations, retrieve sequences, perform mappings between identifiers, and analyze genomic features associated with various organisms. It is particularly useful for bioinformatics applications and large-scale genomic data integration pipelines.",
            "eutils": "The `bioservices.eutils` module is part of the BioServices library, which provides Python interfaces to various bioinformatics web services. Specifically, the `eutils` module offers access to the NCBI Entrez Programming Utilities (E-utilities). These utilities allow users to interact programmatically with the vast array of biological and biomedical data provided by NCBI, including information about genes, proteins, nucleotides, PubMed articles, genomes, and more. The module is particularly useful for querying, retrieving, and analyzing NCBI data efficiently.",
            "eva": "The `bioservices.eva` module in the BioServices library provides programmatic access to the European Variation Archive (EVA), a repository for archiving and sharing genetic variation data. It interfaces with EVA\u2019s API to retrieve and query data related to genomic variants, such as single nucleotide polymorphisms (SNPs), insertions, deletions, and other types of genetic variation. This module facilitates the exploration of variant data, annotations, and related genomic information, typically used in bioinformatics and genomics research.",
            "hgnc": "The `bioservices.hgnc` module in the BioServices library is a Python module designed to interact with the HGNC (HUGO Gene Nomenclature Committee) database. This database provides standardized and unique names and symbols for human genes. The module enables users to programmatically query and retrieve information about human gene symbols, gene names, aliases, and other gene-related metadata. It facilitates the integration of gene nomenclature into bioinformatics workflows, ensuring consistency and accuracy when working with human genetic data.",
            "intact_complex": "The `bioservices.intact_complex` module from the BioServices library is designed to interact with the **IntAct Complex Portal**, a resource that provides detailed information about protein complexes. It facilitates access to and management of data related to experimentally validated protein complexes, such as their composition, functional annotations, and structural information. This module enables researchers to retrieve and analyze complex-specific data for use in systems biology, structural biology, and other bioinformatics studies.",
            "kegg": "The `bioservices.kegg` module is part of the BioServices library, which provides easy access to web services used in bioinformatics. Specifically, the `bioservices.kegg` module interacts with the KEGG (Kyoto Encyclopedia of Genes and Genomes) database. It allows users to query and retrieve various types of biological and biochemical data, such as information about genes, proteins, pathways, compounds, reactions, and organisms. This module simplifies accessing and integrating KEGG data into bioinformatics workflows and research projects.",
            "muscle": "The `bioservices.muscle` module is part of the BioServices library, which provides Python interfaces to access web services and tools commonly used in bioinformatics. Specifically, the `bioservices.muscle` module offers programmatic access to the MUSCLE tool, a well-known multiple sequence alignment tool. It handles biological sequence data, such as nucleotide or protein sequences, and allows users to perform tasks like aligning multiple sequences to identify regions of similarity, which can be useful for phylogenetics, structural biology, and functional analysis. This module simplifies access to MUSCLE functionality by integrating it into Python workflows.",
            "mygeneinfo": "The `bioservices.mygeneinfo` module is part of the BioServices Python package, which provides tools for programmatically interacting with various biological web services. The `mygeneinfo` submodule is specifically designed to interface with the MyGene.info service, which is a high-performance API for querying and retrieving gene-related data. It allows users to access curated and up-to-date information on genes across multiple species, including data such as gene identifiers, gene symbols, gene functions, gene ontology (GO) terms, pathways, diseases, orthologs, and interactions, among other attributes. This module simplifies the process of retrieving gene-centric data for computational biology and bioinformatics research projects.",
            "ncbiblast": "The `bioservices.ncbiblast` module is part of the BioServices library, which is designed to facilitate access to online biological services. The `ncbiblast` module specifically provides an interface to the NCBI BLAST (Basic Local Alignment Search Tool) web service. It allows users to perform sequence similarity searches by submitting nucleotide or protein sequences to the NCBI BLAST service and retrieving the results programmatically. The module handles biological sequence data (DNA, RNA, and protein sequences) and returns BLAST results, which typically include alignments, scores, and statistical significance values that help in identifying homologous sequences or functional annotations.",
            "omicsdi": "The `bioservices.omicsdi` module is part of the BioServices library, which provides tools for accessing various life science-related web services. Specifically, `bioservices.omicsdi` interfaces with **Omics Discovery Index (OmicsDI)**, a platform that integrates and indexes omics datasets from multiple public repositories. It handles data related to multiple omics fields, including genomics, transcriptomics, proteomics, metabolomics, and more. This module enables users to easily query, retrieve, and explore omics datasets, supporting bioinformatics and systems biology research.",
            "omnipath": "The `bioservices.omnipath` module is part of the BioServices library, which is designed for easy access to bioinformatics web services. The `omnipath` module provides an interface to the OmniPath database, a comprehensive resource for molecular interaction data. It primarily handles data related to biological signaling pathways, protein-protein interactions, enzyme-substrate relationships, regulatory mechanisms, post-translational modifications, and other types of molecular networks. The module allows users to query and retrieve these data types programmatically, facilitating integrative analyses and research in molecular and systems biology.",
            "panther": "The `bioservices.panther` module is part of the BioServices library and provides access to the PANTHER (Protein Analysis Through Evolutionary Relationships) database and tools. PANTHER is widely used for functional annotation of genes and proteins, as well as for analyzing gene functionality across species. The module allows users to interact with the PANTHER database programmatically to retrieve information related to gene/protein functions, molecular pathways, gene ontology (GO) terms, and evolutionary relationships. It is particularly valuable for tasks like functional classification, pathway analysis, and large-scale genomics or proteomics data analysis.",
            "pathwaycommons": "The `bioservices.pathwaycommons` module is part of the BioServices library and provides access to **Pathway Commons**, an integrated resource containing biological pathway and interaction data from multiple databases. This module enables users to programmatically query and retrieve information about pathways, molecular interactions, and gene networks. It handles data related to gene and protein interactions, signaling pathways, metabolic networks, and other biological processes crucial for systems biology and network-based analysis.",
            "pdb": "The `bioservices.pdb` module, part of the BioServices library, provides an interface to the Protein Data Bank (PDB) RESTful web service. It is designed to enable users to programmatically interact with and retrieve data from the PDB, a repository for 3D structural data of biological macromolecules such as proteins, nucleic acids, and complexes. This module facilitates querying structural information, metadata, and related records stored in the PDB database, allowing for streamlined access to structural biology data for computational biology and bioinformatics tasks.",
            "pdbe": "The `bioservices.pdbe` module in the **BioServices** library is designed to provide programmatic access to the **Protein Data Bank in Europe (PDBe)**, a resource for structural biology data. The module facilitates querying and retrieving data from the PDBe RESTful services, which primarily include information about macromolecular structures, such as proteins, nucleic acids, and their complexes. This includes structural metadata (e.g., PDB entries), experimental details, annotations, and more, making it useful for bioinformaticians and researchers working with 3D structural data.",
            "pfam": "The `bioservices.pfam` module from the BioServices Python library provides access to the Pfam database, a comprehensive collection of protein families, domains, and functional regions. It allows users to retrieve and interact with data related to protein family classifications, sequence alignments, hidden Markov models (HMMs), and functional annotations. This module is useful for bioinformatics tasks such as identifying protein domains or exploring conserved features across families of proteins. It interacts programmatically with the Pfam web services to fetch and manage this data efficiently.",
            "pride": "The `bioservices.pride` module, part of the BioServices library, is designed to interact with the PRIDE (Proteomics Identification Database) database. PRIDE is a proteomics data repository maintained by the European Bioinformatics Institute (EBI) and provides access to proteomics datasets, including protein and peptide identifications, along with associated metadata. This module enables users to programmatically query, retrieve, and analyze proteomics data from PRIDE, facilitating bioinformatics workflows and large-scale analysis of proteomics experiments.",
            "psicquic": "The `bioservices.psicquic` module is part of the BioServices library, which provides a programmatic interface to web services in bioinformatics. This specific module interacts with the PSICQUIC (Proteomics Standard Initiative Common QUery InterfaCe) web service, which is designed for querying molecular interaction data. The module allows users to programmatically access, retrieve, and process data about protein-protein interactions and other types of molecular interactions from various databases that implement the PSICQUIC standard. This enables streamlined integration and analysis of interaction data within bioinformatics workflows.",
            "pubchem": "The `bioservices.pubchem` module in BioServices is a Python interface designed to interact with the [PubChem](https://pubchem.ncbi.nlm.nih.gov/) database, which is a comprehensive public repository for chemical substances, compounds, and bioactivity data. This module enables users to programmatically access and query data from PubChem, such as chemical structure information, properties, biological activities, safety and toxicity profiles, and more. It is particularly useful for researchers in cheminformatics, bioinformatics, and related fields who want to retrieve or analyze chemical-related data using Python.",
            "quickgo": "The `bioservices.quickgo` module is a part of the BioServices package, which provides access to the QuickGO web service. QuickGO is a tool for browsing the Gene Ontology (GO), a resource that categorizes gene functions in a structured way, including biological processes, molecular functions, and cellular components. This module allows users to interact programmatically with the QuickGO API to retrieve, query, and analyze GO annotations and terms. It handles data related to GO terms, ontologies, evidence codes, species, annotation properties, and more, enabling biological data integration and analysis.",
            "reactome": "The `bioservices.reactome` module is part of the BioServices library, which provides a programmatic interface to interact with the Reactome database. Reactome is a curated, open-source knowledgebase of biological pathways and reactions, covering areas such as metabolism, signaling, and gene expression. The `bioservices.reactome` module enables users to query and retrieve data related to pathways, reactions, proteins, metabolites, and their relationships. It facilitates easy access to pathway information, species-specific biological processes, and molecular interaction data, making it particularly useful for bioinformatics and systems biology research.",
            "rhea": "The `bioservices.rhea` module in BioServices is designed to interact with the Rhea database, a curated resource of biochemical reactions. This module facilitates access to data on chemical reactions, including reaction participants, stoichiometry, and reaction directionality. The data it handles is primarily focused on enzymatic and biochemical reactions, enabling researchers to retrieve and analyze information about metabolic pathways and enzyme function.",
            "seqret": "As of my latest training data (October 2023), the BioServices library is a Python package designed for simplifying access to various biological web services. However, there is no specific mention of a module named `bioservices.seqret` in the official BioServices documentation or community resources. It is possible that you are referring to a function/service wrapper within BioServices that interfaces with *EMBOSS SeqRet*, a tool commonly used to retrieve (and optionally reformat) sequences in bioinformatics.\n\nIf `bioservices.seqret` exists or relates to EMBOSS SeqRet, it likely handles biological sequence data (e.g., DNA, RNA, or protein sequences) and might be used to retrieve or manipulate sequence formats. For further clarification, you may need to check the most recent BioServices documentation or provide additional context.",
            "unichem": "The `bioservices.unichem` module in the BioServices library provides an interface to UniChem, a web service that facilitates the cross-referencing of chemical structures between various chemistry-related databases. It primarily handles chemical compound data, such as identifiers and structural information, and allows users to retrieve and map compound IDs across multiple data sources, making it useful for cheminformatics and bioinformatics tasks.",
            "uniprot": "The `bioservices.uniprot` module is part of the **BioServices** library, designed to facilitate programmatic access to the UniProt database. UniProt is a comprehensive resource for protein sequence and functional information, widely used in bioinformatics. This module enables users to interact with UniProt's web services, allowing the retrieval of protein-related data such as sequences, functional annotations, taxonomy, domains, post-translational modifications, and more. It handles querying of protein records, downloading data in various formats, and integrating protein information into computational workflows.",
            "wikipathway": "The `bioservices.wikipathway` module in the **BioServices** library is designed to interact with WikiPathways, a platform focused on collaborative pathway curation. This module provides programmatic access to WikiPathways\u2019 content via their web services. It allows users to query and retrieve biological pathways, metadata, and related information such as gene and metabolite interactions. The data handled by this module includes pathway representations, pathway annotations, associated gene and metabolite data, and other curated biological information from WikiPathways, making it useful for systems biology, bioinformatics, and pathway analysis tasks."
        }

        api_module = API_name.split('.')[1].lower()

        bioservices_module_description = bioservices_modules[api_module] if api_module in bioservices_modules else "No description."

        # Only for testing, remove the print statements in production
        print("=" * 50)
        print("API Name:", API_name)
        print("API Module:", api_module)
        print(bioservices_module_description)

        prompt1 = f"""
Instruction:
Generate 10 examples that uses the function {API_name}(...) from BioServices to train an intelligent code assistant for computational biology.
Each example should be a JSON dictionary with fields "instruction" and "code".
- The text field is a single line of language instruction the user would say to the code assistant.
- The code field is a single line of code the code assistant should generate given the user's instruction.
- Use one line per example. Be specific when writing the user instruction - write it as if a human user would say it to the assistant.
Text Field Instructions:
- Make sure the text field is descriptive enough that you can map it back to the function it is referring to with very high confidence.
- IMPORTANT: You must not include the function name or parameters name in the text field!
- You must include parameters value in the text field if they cannot be deduced from generic query alone, make sure to include them in a natural-sounding way (for example, parameter value formatted in snake casing should be converted to natural-sounding words), but they must be specific values, not generic filler words.
- Make sure to refer to the BioServices module description below to include context about the corresponding module such that the text field has enough context on what specific bioinformatics data it's retrieving or what bioinformatics task it is performing, without it being generic.
- Overall, for each text field, you should be able to map it back to the corresponding BioServices module, function, and parameters value with very high confidence.
- Make sure every text field is equally descriptive, but vary the tone between: polite, straightforward, casual.
Code Field Instructions:
- IMPORTANT: Make sure the code field follows the {API_name}(...) format, do not modify {API_name} section by shortening it!
- Make sure to include the correct parameters with values in (...) section.
- Make sure each parameter value is specific valid value, not generic value.
BioServices Module description:
{bioservices_module_description}
Emphasize intent and methods of the function and distinguish between Target and Contrasting descriptions.
For example, given Target description "Scatter plot in UMAP basis" Contrasting description ["Scatter plot in PHATE basis"], generated examples `Please create a scatter plot in the UMAP basis`should include intent of "create a scatter plot" and details "in the UMAP basis", omitting "PHATE specifics".
"""
        prompt2 = f"""
Refer to the following code documentation on what each function does, and the distinctions between Target and Contrasting descriptions.
docstring of {docs}
"""
        return prompt1, prompt2

    # from prompt/instruction.py
    def build_prompt_old(self, API_name, docs):
        return f"""
Instruction: Generate 10 examples that uses the function {API_name}(...) to train an intelligent code assistant for computational biology. Each example should be a JSON dictionary with fields "instruction" and "code". The text field is a single line of language instruction the user would say to the code assistant. The code field is a single line of code the code assistant should generate given the user's instruction. Use one line per example. Be specific when writing the user instruction - write it as if a human user would say it to the assistant. Never include function name or parameters name. Vary tones among queries: polite, straightforward, casual. Emphasize intent and methods of the function and distinguish between Target and Contrasting descriptions. For example, given Target description "Scatter plot in UMAP basis" Contrasting description ["Scatter plot in PHATE basis"], generated examples `Please create a scatter plot in the UMAP basis`should include intent of "create a scatter plot" and details "in the UMAP basis", omitting "PHATE specifics".""",f"""Refer to the following code documentation on what each function does, and the distinctions between Target and Contrasting descriptions. \ndocstring of {docs}\n"""

class ParametersPromptBuilder(PromptBuilder):
    # from prompt/parameters.py
    def build_prompt(self, user_query, api_docstring, parameters_name_list, param_type):
        common_part = f"""
Task Description: Determine from the API DOCSTRING and USER QUERY whether each parameter needs a predicted value. Based on the parameter type and meaning, infer if it needs to fill in a value or modify the default value, and return all the parameters that need prediction or modification with their predicted values.
Instructions:
1. Extract values and assign to parameters: If there are values explicitly appearing in the USER QUERY, such as those quoted with '', values with underscores like value_value, or list/tuple/dict like ['A', 'B'], they are likely to be actual parameter values. When handling these values:
Boolean values: If we are predicting boolean values, you can ignore these explicitly mentioned values.
Integer/float/str/list/tuple values: If we are predicting these types of values, you can assign them to the most corresponding parameters based on the modifiers of the extracted values.
2. Extract and convert values if necessary: Some values might not be obvious and require conversion (e.g., converting "Diffusion Map" to "diffmap" if "Diffusion Map" appears in the user query, as this is the correct usage for some API. Another example is converting "blank space delimiter" to "delimiter=' '"). Ensure that the returned values are correctly formatted, case-sensitive, and directly usable in API calls.
3. Identify relevant parameters: From the USER QUERY and API DOCSTRING, identify which parameters require value extraction or default value modification based on their type and meaning.
4. Never generate fake values: Only include values explicitly extracted in the USER QUERY or that can be clearly inferred. Do not generate fake values for remaining parameters.
5. Response format: Format results as {{"param name": "extracted value"}}, ensuring they are executable and match the correct type from the API DOCSTRING. For example, return ['a', 'b'] for list type instead of Tuple ('a', 'b'). Only return parameters with non-empty values (i.e. not None).
6. Avoid complexity: Avoid using 'or' in parameter values and ensure no overly complex escape characters. The format should be directly loadable by json.loads.
7. Distinguish similar parameters: For similar parameters (e.g., 'knn' and 'knn_max'), distinguish them in meaning and do not mix them up. If there are two parameter candidates with the same meaning for one value, fillin them both with the extracted value.
8. Maintain Format and transfer format if necessary: Ensure the final returned values match the expected parameter type, including complex structures like [("keyxx", 1)] for parameters like "obsm_keys" with types like "Iterable[tuple[str, int]]". If USER QUERY contains list, tuple, or dictionary values, extract them as a whole and keep the format as a value.
"""
        if param_type=='boolean':
            return f"""
{common_part}
Boolean:
- Determine the state (True/False) based on the full USER QUERY and the parameter description, rather than just a part of it. If the 'True/False' value is not explicitly mentioned but the meaning is implied in the USER QUERY, still predict the parameter.

Examples:
USER QUERY: "... with logarithmic axes" => {{"log":"True"}}
USER QUERY: "... without logarithmic axes" => {{"log":"False"}}
USER QUERY: "... hide the default legends and the colorbar legend" => {{"show":"False", "show_colorbar":"False"}}
USER QUERY: "... and create a copy of the data structure" => {{"copy": "True"}}, do not predict "data"

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""
        elif param_type=='literal':
            return f"""
{param_type}
Literal:
- Find the candidate with the same level of informativeness. If there are multiple candidates containing the keyword, but the query provides only minimal information, select the candidate with the least information content. If a parameter name is mentioned without a value, skip it.

Examples:
USER QUERY: "... with basis 'Diffusion Map'" => {{"basis":'diffmap'}}
USER QUERY: "... blank space delimiter" => {{"delimiter":' '}}

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""
        else: # if param_type=='int'
            return f"""
{common_part}
int, float, str, list[str], tuple[str]:
- Choose only parameters with explicitly mentioned values in the USER QUERY. If a parameter name is mentioned without a value (e.g., "a specific group/key" where "group/key" is the parameter name but no value is given), ignore that parameter.

Examples:
USER QUERY: "... with chunk size 6000" => {{"chunked":"True", "chunk_size":"6000"}}
USER QUERY: "... with groups ['A', 'B', 'C']" => {{"groups":['A', 'B', 'C']}}
USER QUERY: "... with at least 100 counts." => {{"min_counts": 100}}
USER QUERY: "... for a specific layer 'X_new'?" => {{"layer": "X_new"}}, do not assign "X_new" to "X"

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""

class SummaryPromptBuilder(PromptBuilder):
    # from prompt/summary.py
    def build_prompt(self, user_query, api_docstring):
        return f"""
Help to summarize the task with its solution in layman tone, it should be understandable by non professional programmers. Starts from description of the user query `{user_query}` and includes the selected API function's Docstring `{api_docstring}`.
Please use the template as `The task is ..., we solved it by ...`. The response should be no more than two sentences
"""

class SummaryPromptFullBuilder(PromptBuilder):
    # from prompt/summary.py
    def build_prompt(self, user_query, api_docstring, execution_code):
        return f"""
Help to summarize the task with its solution in layman tone, it should be understandable by non professional programmers. Starts from description of the user query `{user_query}` and includes the selected API function `{api_docstring}`. The generated code is `{execution_code}`.
Please use the template as `The task is ..., we solved it by ...`. The response should be no more than three sentences. Additionally, the interpretation encompasses explanations for the parameters utilized in the generated code.
"""

class ExecutionCorrectionPromptBuilder(PromptBuilder):
    def build_prompt(self, user_input, history_record, error_code, error_message, variables, LIB):
        return f"""
Your task is to correct a Python code snippet based on the provided information. The user's inquiry is represented by '{user_input}'. The history of successful executions is provided in '{history_record}', and variables in the namespace are supplied in a dictionary '{variables}'. Execute the error code snippet '{error_code}' and capture the error message '{error_message}'. Analyze the error to determine its root cause. Then, using the entire API name instead of abbreviation in the format '{LIB}.xx.yy'. Ensure any new variables created are named with the prefix 'result_' followed by digits, without reusing existing variable names. If you feel necessary to perform attribute operations similar to 'result_1.var_names.intersection(result_2.var_names)' or subset an AnnData object by columns like 'adata_ref = adata_ref[:, var_names]', go ahead. If you feel importing some libraries are necessary, go ahead. Maintain consistency with the style of previously executed code. Ensure that the corrected code, given the variables in the namespace, can be executed directly without errors. Return the corrected code snippet in the format: '\"\"\"your_corrected_code\"\"\"'. Do not include additional descriptions.
"""# please return minimum line of codes that you think is necessary to execute for the task related inquiry

class LLMPromptBuilder(PromptBuilder):
    def build_prompt(self, user_input, history_record, variables, LIB):
        return f"""
Your task is to generate a Python code snippet based on the provided information. The user's inquiry is represented by '{user_input}'. The history of successful executions is provided in '{history_record}', and variables in the namespace are supplied in a dictionary '{variables}'. Analyze the user intent about the task to generate code. Then, using the entire API name instead of abbreviation in the format '{LIB}.xx.yy'. Ensure any new variables created are named with the prefix 'result_' followed by digits, without reusing existing variable names. If you feel necessary to perform attribute operations similar to 'result_1.var_names.intersection(result_2.var_names)' or subset an AnnData object by columns like 'adata_ref = adata_ref[:, var_names]', go ahead. If you feel importing some libraries are necessary, go ahead. Maintain consistency with the style of previously executed code. Ensure that the generated code, given the variables in the namespace, can be executed directly without errors. Return the corrected code snippet in the format: '\"\"\"your_corrected_code\"\"\"'. Do not include additional descriptions.
"""# please return minimum line of codes that you think is necessary to execute for the task related inquiry

class MultiTaskPromptBuilder(PromptBuilder):
    def build_prompt(self, LIB, goal_description, data_list=[]):
        prompt = f"""
Create step-by-step task plan with subtasks to achieve the goal.
Tone: Vary the tone appropriately across tasks, such as polite, straightforward, or casual.
Subtask length: Each subtask should consist of around 10 words.
Subtask scope: Each subtask should invoke only one API from the {LIB} PyPI library. Split subtasks if they contain more than one action. For example, use "Please filter the data" alongside "Could you normalize the data?" instead of "Please filter and normalize." Use "Please load/filter/normalize dataset1" and "Can you load/filter/normalize dataset2" instead of "How to load/filter/normalize dataset1 and dataset2.". Use "Integrate two datasets" (since integration requires two datasets).
Subtask order: Ensure prerequisites appear before subsequent API calls.
Number of subtasks: Create approximately 5 tasks based on complexity. The visualization step should appear last and not be included in previous steps.
Focus on Operations: Concentrate on actions and objectives without over-explaining the object being operated on. For instance, "Please create scatter plot in UMAP basis" emphasizes the action rather than the object (data). Use "Could you process the data" instead of "Could you process the data for spatial analysis."
Avoid Dataset Mentions: Aside from the data-loading step, avoid directly mentioning the dataset. For example, use "Could you draw scatter plot for the data" instead of "Could you draw scatter plot for the pbmc3k data?". Notice that never mention keywords like `pbmc3k`, `pbmc68k` except for the data loading step.
Goal-Oriented Structure: State the goal at the beginning of each subtask. Be clear and concise, omit API names, and focus on key actions only.
Only respond in JSON format strictly enclosed in double quotes, adhering to the Response Format.
Exclude any extraneous content from your response.
---
Examples:
Goal: use Squidpy for spatial analysis of Imaging Mass Cytometry data, focusing on the spatial distribution and interactions of cell types
Response:
{{"plan": [
"step 1: Load pre-processed Imaging Mass Cytometry data.",
"step 2: Could you show cell type clusters in spatial context?",
"step 3: Please calculate co-occurrence of cell types across spatial dimensions.",
"step 4: Compute neighborhood enrichment.",
"step 5: Can you plot neighborhood enrichment results?",
]}}
---
Now finish the goal with the following information:
Goal: {goal_description}\n
Response Format:
{{"plan": ['step 1: content', 'step 2: content', ... ]}}
""" 
# "step 8: Analyze interaction matrix to count interactions among clusters.",
# "step 9: Compute network centrality scores to evaluate the importance of each cell type in the spatial graph.",

# Specify API and function names and avoid including non PyPI functions in your answered code.
# Avoid creating steps that are too coarse or too detailed. 
# Only include keywords in the subtask.
# If a file path is provided, use it to load the data. If no file path is provided, use the built-in dataset API to load the default dataset. If several filepaths are provided, either use them in one subtask or in different subtasks regarding the task description.
        prompt += f"Input: Your data is provided in the format 'file path: file description' from {data_list}" if data_list else "You have no local data. Please use only APIs that access built-in datasets and avoid those requiring local data loading."
        return prompt

class ExecutorPromptBuilder(PromptBuilder):
    def build_prompt(self, api_docstring, namespace_variables, error_code, possible_solution="", api_examples="", success_history_code="", goal_description=""):
        if possible_solution:
            possible_solution_info = f"\nPossible solution from similar issues from Github Issue Discussion:\n{possible_solution}"
        else:
            possible_solution_info = ""
        # remove api_examples as it is already included in the api docstring
        if api_examples and api_examples != "{}":
            api_examples_info = f"\nHere are some examples. Identify the key prerequisite APIs or the correct usage of the target API to address the bug. Do not copy their parameters; predict based on the variable information in our namespace: {api_examples}.\n"
        else:
            api_examples_info = ""
        prompt = f"""
Task: Analyze and correct the most recent failed Python script attempt based on the provided traceback information.
Make corrections considering all previous failed attempts and the associated error details, ensuring that prior mistakes are not repeated. You must make some valid and meaningful change, and keep the key correction step you did correctly in past attempt. Please ensure each time you make a correction, you are moving towards the correct solution.
Include all necessary library imports at the beginning. 
Ensure the correct execution order for API dependencies like pandas, numpy, and AnnData, based on the traceback error and variables. 
Use only variables from successful executions. 
Remove unnecessary or incorrect parameters, ensuring required ones are in proper order. 
Adjust misused attributes or values, especially for AnnData object-specific attributes. 
Preprocess data if needed using relevant tools or APIs before the main API call. 
Remove unnecessary optional parameters causing errors. 
Respond with the corrected code in JSON format. 
Refer to namespace variables for their values and attributes; avoid variables with None.
The function returned variable should not be the same name as the function name.

Common Errors to Address:
Import Verification: Confirm necessary libraries are imported.
API Usage: Replace or continue with the correct API as needed.
Parameter Handling: Streamline parameters to essentials, removing any incorrect or irrelevant ones.
Prerequisite API Calls: Include any necessary pre-API steps.
Identify and address indirect errors by deducing the root cause. Present the logical steps in the 'analysis' section and the corresponding code in the 'code' section.
KeyError: Ensure to use the exist and correct attribute from existing variables in namespace. Or use the correct API to generate the attribute before executing this subtask. Sometimes the key error is due to that there lack of previous API. 
TypeError: Sometimes the processed variable gets None, please use other variable in namespace instead of the null variable.
AttributeError: Sometimes the message 'NoneType' object indicates that the variable is None, verify that you're not using a None variable and replace it with a valid one from the current namespace. Or check whether the annotations or parameters you're accessing (e.g., .var, .obs, etc.) are indeed present in the object. It's possible that the attribute you're trying to use does not exist for this specific object or is incorrectly referenced.
ValueError: Be careful for choosing the attributes from `.obs` and `.var` of AnnData object, as some API requires annotations are from the same attribute.

Here are information:
Success execution History: {success_history_code}
Existing Namespace variables: {namespace_variables}
Current Goal: {goal_description}
History Failed Attempts with their tracebacks:\n {error_code}
{possible_solution_info}{api_examples_info}
API Docstring: {api_docstring}.
Response Format: {{"analysis": "Explain how to correct the bug in 2 sentences, including the reason of the bug, and how to correct.", "code": "Contain the Corrected failed attempt code, exclude code from `success execution history`, exclude `save_plot_with_timestamp()`, exclude any comments in the code. Notice to set any parameters named `inplace`, `show` as True, while `copy` as False"}}
""" # You only need to keep required parameters from previous trial codes, only keep minimum optional parameters necessary for task. Remove optional parameters from error code which cause the problem. Please ensure that required parameters are passed in their proper positional order, as keyword arguments should only be used for optional parameters. You only need to include the task related correct code in your response, do not repeat other API from the success execution history in your response. For parameters starting with 'result_', use only those that exist in the namespace. Do not generate inexist variables.
        return prompt
    
class ModifySubtaskPromptBuilder(PromptBuilder):
    # add parameters
    def build_prompt(self, main_goal, current_subtask,  execution_history, namespace_variables, api_docs):
        query_prompt = f'''
Refine the subtask description by integrating essential parameters and their values from the docstring, ensuring their values are appropriate for the next steps in the code execution. Inherit any valid parameter values from the current subtask, verifying their accuracy and relevance. Check the docstring for API dependencies, required optional parameters, parameter conflicts, duplication, and deprecations. If the main goal provides a method name and the subtask can use this method to accomplish its goal, include the method name in the polished subtask. Include only the parameters with explicitly assigned values; avoid stating default values or parameters with vague values. 
Please avoid assigning values to `inplace`, `copy`, `show` parameters.
Provide only the refined subtask description. Avoid including any extraneous information.
---
Example:
original subtask description: Can you normalize the data and identify 'target_sum' as median of total counts, 'key' with specific value, 'max_num' as '10', 'log' as 'True'
thought: we delete some parameters, remove 'target_sum' because the parameter 'target_sum' just repeats its description and wasn't assign valid value, remove 'key' as it wasn't assign valid value, and the default value of the parameter 'log' is 'True' so we remove it, while only the 'max_num' as '10' is valid and non default and executable.
refined subtask description: Can you normalize the data and set 'max_num' as '10'

Example:
Details to consider
Namespace variables: {{"result_1": "{{'type': 'AnnData', 'value': AnnData object with n_obs \u00d7 n_vars = 3798 \u00d7 36601\n    obs: 'in_tissue', 'array_row', 'array_col'\n    var: 'gene_ids', 'feature_types', 'genome'\n    uns: 'spatial', 'pca'\n    obsm: 'spatial', 'X_pca'\n    varm: 'PCs'}}"}}
Extract necessary parameter details and constraints from API Docstring: def squidpy.gr.ripley(adata=$, cluster_key=@, mode=$, spatial_key=@, metric=@, n_neigh=@, n_simulations=@, n_observations=@, max_dist=@, n_steps=@, seed=@, copy=@):
original subtask description: Can you calculate Ripley's statistics?
refined subtask description: Can you calculate Ripley's statistics with 'cluster_key' set as 'array_row'?

Example:
main goal: Use Scanpy to finish trajectory inference using the PAGA method.
original subtask description: Please perform trajectory inference.
refined subtask description: Please perform trajectory inference using the PAGA method.

Example:
Main goal: Use Scanpy to conduct gene annotation on dataset 3k PBMCs.
original subtask description: Load the built-in dataset.
refined subtask description: Load the built-in dataset 3k PBMCs.
---
Details to consider
Understand context and dependencies from past executed code: {execution_history}
Ensure parameter compatibility for existing namespace variables: {namespace_variables}
Extract necessary parameter details and constraints from API Docstring: {api_docs}
Main goal (that includes this subtask as a step):  {main_goal}
original Subtask description: {current_subtask}
refined subtask description: 
''' # Never include data description in other subtasks except for the data loading subtask. Ensure Goal-Oriented Task Structuring, place the goal description at the beginning of each subtask.
        return query_prompt

class ModifySubtaskCorrectionPromptBuilder(PromptBuilder):
    # correct subtask
    def build_prompt(self, main_goal, current_subtask,  execution_history, namespace_variables, api_docs):
        query_prompt = f'''
Refine the subtask description to more closely align with the functionality and intent of a specific API. Review the docstrings of similar API candidates that will be provided, and polish the task description to ensure it encapsulates the API's capabilities and constraints accurately. Refine the interpretation of the existing task based on the most appropriate API's features. If the main goal provides a method or data name and the subtask can use this method or data to accomplish its goal, include this keyword in the polished subtask. Omit API name from subtask.
---
Example:
Original Subtask description: Can you scale the data to unit variance and zero mean and clip values?
response: Can you scale the data to unit variance and zero mean and clip values at maximum value as '10.0'?

Example:
Main goal: Use Scanpy to finish trajectory inference using the PAGA method.
original subtask description: Please perform trajectory inference in this step.
refined subtask description: Please perform trajectory inference using the PAGA method in this step.

Example:
Main goal: Use Scanpy to conduct gene annotation on dataset 3k PBMCs.
original subtask description: Load the built-in dataset.
refined subtask description: Load the built-in dataset 3k PBMCs.
---
Details to consider:
Main goal (that includes this subtask as a step): {main_goal}.
Extract relevant details from the API docstrings to understand constraints and capabilities: {api_docs}
Review past executed code and namespace variables to ensure compatibility and relevance: {execution_history}, {namespace_variables}
Refine the original subtask description to closely match the intended API functionality: {current_subtask}
response: 
''' # Never include data description in other subtasks except for the data loading subtask. Ensure Goal-Oriented Task Structuring, place the goal description at the beginning of each subtask.
        return query_prompt

class SubtaskCodePromptBuilder(PromptBuilder):
    def build_prompt(self, data_list, goal_description, history_summary, execute_success=False, execute_info=None):
        prompt = f"""You should Act as a biologist proficient in Python programming, the rules must be strictly followed!
Rules:
- Import all necessary libraries before writing codes.
- Do not deviate from the role of a biologist proficient in Python programming.
- Strictly adhere to all rules.
- Utilize input information to craft a comprehensive plan for achieving the goal.
- Never mess up the input and output directory. You can never plan to load data from output filepath.
- You are provided a python terminal with target PyPI library already installed and imported.
- The history of what you have done is provided, you should take the name changes of some files into account, or use some output from previous steps.
- You should use all information you have to write python codes to finish your current task.
- All code requirements must be followed strictly when you write codes.
- Respond solely in JSON format according to the fixed format.
- Limit response content strictly to JSON enclosed in double quotes.
- Exclude any extraneous content from your response.
- Provide detailed responses whenever possible.
- Do not repeat what you have done in history.

Code requirements:
- Import all necessary libraries before writing codes.
- You have a python terminal environment with target PyPI library already installed and imported.
- You can import and use API from basic tools like pandas, numpy, and anndata. If necessary, you can incorporate some intermediate data preprocessing steps using these basic tools.
- Never forget to import the library before you use any API from this library.
- Only use the PyPI functions and APIs to generate code.
- Pay attention to the number of input files and do not miss any.
- Process each file independently and can not use FOR loop.
- Use the path for all files according to input and history.
- Use the default values for all parameters that are not specified.
- Not write anything else except for your JSON response.
- Not repeat what you have done in history.

History: {history_summary}\n
Current Task: {goal_description}\n
Response Format:
{{"tool": "name of the tool you use", "code": "python code to import library and accomplish the current task"}}"""
        if data_list:
            prompt += f"""
You have the following information in a list with the format `file path: file description`. I provide those files to you, so you don't need to prepare the data: {data_list}"""
        else:
            prompt += f"""
You don't have any local data provided. Please use API to load builtin dataset."""
        final_prompt = prompt
        if execute_success:
            pass
        else:
            final_prompt+=f'''{execute_info}''' # You got this error when you write this code last time. You should solve this bug: {. Here are some suggestions: Based on the error message, could you provide suggestions and solutions for the code to resolve the issue? This might involve running certain prefix APIs, preprocessing data, or addressing any missing attributes, or using correct API instead of fake API due to hallucination.
        return final_prompt

class PromptFactory:
    def create_prompt(self, prompt_type, *args):
        if prompt_type == 'composite_docstring':
            return CompositeDocstringPromptBuilder().build_prompt(*args)
        elif prompt_type == 'composite_name':
            return CompositeNamePromptBuilder().build_prompt(*args)
        elif prompt_type == 'instruction_generation':
            return InstructionGenerationPromptBuilder().build_prompt(*args)
        elif prompt_type == 'parameters':
            return ParametersPromptBuilder().build_prompt(*args)
        elif prompt_type == 'summary':
            return SummaryPromptBuilder().build_prompt(*args)
        elif prompt_type == 'summary_full':
            return SummaryPromptFullBuilder().build_prompt(*args)
        #elif prompt_type == 'execution_correction':
        #    return ExecutionCorrectionPromptBuilder().build_prompt(*args)
        elif prompt_type == 'LLM_code_generation':
            return LLMPromptBuilder().build_prompt(*args)
        elif prompt_type == 'multi_task':
            return MultiTaskPromptBuilder().build_prompt(*args)
        elif prompt_type == 'executor_correction':
            return ExecutorPromptBuilder().build_prompt(*args)
        #elif prompt_type == 'subtask_code':
        #    return SubtaskCodePromptBuilder().build_prompt(*args)
        elif prompt_type == 'modify_task_parameters':
            return ModifySubtaskPromptBuilder().build_prompt(*args)
        elif prompt_type == 'modify_task_correction':
            return ModifySubtaskCorrectionPromptBuilder().build_prompt(*args)
        else:
            raise ValueError("Unknown prompt type")

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    factory = PromptFactory()
    prompt = factory.create_prompt('composite_docstring', "API description", "func_inputs", "func_outputs", "description_text")
    print(prompt)
