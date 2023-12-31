<h2 align="center">GitRepo2BioMANIA app</h2>

## Step 1: Prepare Your GitHub Repository

Before you can use a GitHub repository to create a BioMANIA app, you need to make sure your repository is structured correctly and can be installed as a Python package. Follow these steps:

### 1.1. Add `setup.py` File:

Create a `setup.py` file in your GitHub repository. This file is used to package your code so that it can be easily installed via `pip`. Here's a basic example of a `setup.py` file:

```python
import setuptools
import glob
import os

fname = 'requirements.txt'
with open(fname, 'r', encoding='utf-8') as f:
	requirements =  f.read().splitlines()

required = []
dependency_links = []

# Do not add to required lines pointing to Git repositories
EGG_MARK = '#egg='
for line in requirements:
	if line.startswith('-e git:') or line.startswith('-e git+') or \
		line.startswith('git:') or line.startswith('git+'):
		line = line.lstrip('-e ')  # in case that is using "-e"
		if EGG_MARK in line:
			package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
			repository = line[:line.find(EGG_MARK)]
			required.append('%s @ %s' % (package_name, repository))
			dependency_links.append(line)
		else:
			print('Dependency to a git repository should have the format:')
			print('git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name')
	else:
		if line.startswith('_'):
			continue
		required.append(line)

setuptools.setup(
     name='your_project_name',
     use_scm_version=True,
     setup_requires=['setuptools_scm'],
     packages=['your_project_name'],
     package_dir={'': 'src'},
     py_modules=["your_project_name"+'.'+os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/your_project_name/*.py')],
     install_requires=required,
     dependency_links=dependency_links,
     author="XXX",
     author_email="xxx@example.com",
     description="your_project_description",
     long_description=open('README.md').read(),
     url="https://github.com/your_github_page/your_repository",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ]
 )
```

Make sure to replace `'your_project_name'` with the actual name of your package and specify your package's dependencies in the `requirements.txt` . Here's a basic example of a `requirements.txt` file:

```python
# below three types of lib are supported
astunparse==1.6.3
beautifulsoup4
git+https://github.com/aertslab/pycisTopic@master#egg=pycisTopic
```

Or you can use `pip freeze > requirements.txt` to obtain it simply.

### 1.2. Add `__init__.py` Files:

Create empty `__init__.py` files in each directory where you want to define a module within your package. These files are necessary to make submodules within your package discoverable by our method through `dir`. For example, if you have a structure like this:

```
github_repo/
├── src/
│   └── your_package/
│       ├── __init__.py
│       ├── module1.py
│       ├── module2.py
│       └── subpackage/
│           ├── __init__.py
│           └── module3.py
│
├── setup.py
└── README.md
```

Make sure both `__init__.py` files are present. Here's a basic example of  `github_repo/src/your_package/__init__.py` file:

```python
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("your_project_name").version
except DistributionNotFound:
    pass

from . import module1
from . import module2
from . import subpackage
```

Don't forget to check that code at the same level needs to use relative imports. For example, when module1 calls module2, it should be done as follows: `from .module2 import your_function_name`.

Notice that if files and submodules have the same name, it may lead to circular import issues.

### 1.3. Install the GitHub Repository:

Install your GitHub repository to your local machine using the `pip install` command:

```bash
# install remotely
pip install git+https://github.com/your_github_page/your_repository.git. 
# Or build locally
pip install  -e .
```

## Step 2: Use BioMANIA

Now that your GitHub repository is properly installed, you can create a BioMANIA app to use it. Feel free to skip any Optional steps to make it easy. Here are the steps:

### 2.1. Create BioMANIA app:

#### 2.1.1 Generate Initial API Configuration:
To obtain `API_init_prepare.json`, use the following script:
```shell
export LIB=scanpy
python Git2APP/get_API_init_from_sourcecode.py --LIB ${LIB}
```

Note: You have the option to alter the filtering rules by modifying filter_specific_apis.

#### 2.1.2. Refine Function Definitions and Docstrings:
(Optional) Now, you need to add docstrings in your source code and refine the function definitions. You can either choose running our provided script and get `API_init.json` with GPT:

```shell
python Git2APP/get_API_docstring_from_sourcecode.py --LIB ${LIB}
```

Tips 
- This script is designed to execute only on APIs lacking docstrings or well-documented parameters. It automatically skips APIs that meet the established documentation standards. Please note that running this script requires a paid OpenAI account. 

- This script is based on LLM responses for modification, and the quality of the results may not be entirely satisfactory. Users need to ensure that the necessary parameters type are provided for inference, as `None` type may lead to execution failures in the API due to missing essential parameters.

- To accommodate the fact that `args`, `kwargs`, and similar parameters in general APIs are optional, we currently filter them out during prediction. Therefore, it's advisable to avoid using args as parameters in the code.

It is better that if you can design the docstrings by yourself as it is more accurate. `NumPy` format is preferred than `reStructuredText` and `Google` format. Here's a basic example of an effective docstring :

```python
from typing import Union
import pandas as pd
from your_module import TreeNode

def add(a:int, b:int) -> int:
    """
    Compute the sum of two integers.

    Parameters:
    -----------
    a : int
        The first integer.
    b : int
        The second integer.

    Returns:
    --------
    int
        The sum of `a` and `b`.
    """
    return a + b
```

You can refer to the prompts available in [BioMANIA](https://www.biorxiv.org/content/10.1101/2023.10.29.564479v1) to add the function body, or either using the [prompt](./src/Git2APP/get_API_docstring_from_sourcecode.py) that modified based on that.

If you already have a well-documented code, generate API_init.json with:
```shell
cp -r data/standard_process/${LIB}/API_init_prepare.json data/standard_process/${LIB}/API_init.json 
```

#### 2.1.3. Run Training Scripts:
For subsequent steps, start from obtaining `API_composite.json` of [`Run with script/Training`](./PyPI2APP.md#training) section. Following these instructions will result in the generation of data files in data/standard_process/your_project_name and model files in hugging_models.

```python
data/standard_process/your_project_name
├── API_composite.json
├── API_init.json
├── API_init_prepare.json
├── API_inquiry.json
├── API_inquiry_annotate.json
├── API_instruction_testval_query_ids.json
├── api_data.csv
├── centroids.pkl
├── retriever_train_data
│   ├── corpus.tsv
│   ├── qrels.test.tsv
│   ├── qrels.train.tsv
│   ├── qrels.val.tsv
│   ├── test.json
│   ├── test.query.txt
│   ├── train.json
│   ├── train.query.txt
│   ├── val.json
│   └── val.query.txt
└── vectorizer.pkl

hugging_models
└── retriever_model_finetuned
    ├── your_project_name
    │   ├── assigned
    │   │   ├── 1_Pooling
    │   │   ├── README.md
    │   │   ├── config.json
    │   │   ├── config_sentence_transformers.json
    │   │   ├── eval
    │   │   ├── model.safetensors
    │   │   ├── modules.json
    │   │   ├── sentence_bert_config.json
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer.json
    │   │   ├── tokenizer_config.json
    │   │   └── vocab.txt
    │   └── tensorboard
    │       └── name_desc
```

### 2.2 Add logo

Add a logo image to `BioMANIA/chatbot_ui_biomania/public/apps/` and modify the link in `BioMANIA/chatbot_ui_biomania/components/Chat/LibCardSelect.tsx`.

Be mindful of the capitalization in library names, as it affects the recognition of the related model data loading paths.

### 2.3 Use UI service.

Follow the steps in [`Run with script/Inference`](../README.md#inference) section in `README` to start UI service. Don’t forget to set an OpenAI key in `.env` file as recommended in `README`.

Remember to update the app's accordingly to your repository improvements.

**Tips: Currently, we do not support real-time communication. Therefore, if there is content that requires a long time to run, such as training a model, it is best to train only one epoch per inquiry. We might plan to support real-time display of results in the near future.**

### 2.4 Share your APP!

Follow the steps in [`Share your APP`](../README.md#share-your-app) section in `README` to introduce your tool to others!


I hope this tutorial helps you create your BioMANIA app with your GitHub-hosted package. If you have any further questions or need assistance with specific steps, feel free to ask!
