# GitRepo2BioMANIA app

**Step 1: Prepare Your GitHub Repository**

Before you can use a GitHub repository to create a BioMANIA app, you need to make sure your repository is structured correctly and can be installed as a Python package. Follow these steps:

**1.1. Add `setup.py` File:**

Create a `setup.py` file in your GitHub repository. This file is used to package your code so that it can be easily installed via `pip`. Here's a basic example of a `setup.py` file:

```python
import setuptools
import glob
import os

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

**1.2. Add `__init__.py` Files:**

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

**1.3. Add docstring**

Now, you need to add docstrings in your source code and refine the function definitions.  Here's a basic example of an effective `docstring` :

```python
from typing import Union
import pandas as pd
from your_module import TreeNode

def compute_gini_gain(dataset: Union[pd.DataFrame, list], node: TreeNode) -> float:
		"""
    Calculate the Gini gain of a given dataset and node using a decision tree classifier.

    :param dataset: The dataset for which Gini gain needs to be calculated.
    :param node: The specific node in the decision tree being evaluated.
    :return: Returns the calculated Gini gain for the given node in the dataset.
    """
    dt = DecisionTreeClassifier(criterion='gini', max_depth=1)
    node_values = node_value(dataset, node)
    dt.fit(node_values.values.reshape(-1, 1), dataset.y)
    impurity_before_split = dt.tree_.impurity[0]
    impurity_left_child = dt.tree_.impurity[1]
    impurity_right_child = dt.tree_.impurity[2]
    n_node_samples = dt.tree_.n_node_samples
    
    gini_before = impurity_before_split
    gini_after = (n_node_samples[1]/n_node_samples[0])*impurity_left_child + (n_node_samples[2]/n_node_samples[0])*impurity_right_child
    gini_gain = gini_before - gini_after
    return gini_gain
```

You can refer to the prompts available in BioMANIA to add the function body, or design by yourself.

```python
# Task Description:

"""You are an expert in Python programming. Your task is to write the docstring for the given information of an invisible function. Interpret the assigned inputs and return variables in the docstring.

The description of used APIs inside this code is: {API description}
The input and output parameter information is as below:
- Parameters: {func inputs}
- Returns: {func outputs}
- The other description associated with the code is: {description text}
- Please extract the core information in 1-2 sentences and polish it. Docstring description should only use 1-2 sentences.

Your Response format is detailed docstring. Please do not include other information except for response information, in reStructuredText format. Never include specific API information in description."""
```

**Step 2: Use BioMANIA**

Now that your GitHub repository is properly installed, you can create a BioMANIA app to use it. Here are the steps:

**2.1. Install the GitHub Repository:**

Install your GitHub repository to your local machine using the `pip install` command:

```bash
pip install git+https://github.com/your_github_page/your_repository.git. 
```

**2.2. Create BioMANIA app:**

Follow the steps in [`Run with script/Training`](README.md#training) section in `README` to create data files under `data/standard_process/your_project_name` and model files under `hugging_models`

```python
data/standard_process/your_project_name
├── API_composite.json
├── API_init.json
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
    ├── deap
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

**2.3 Use UI service.**

Follow the steps in [`Run with script/Inference`](README.md#inference) section in `README` to start UI service. Don’t forget to set an OpenAI key in `.env` file as recommended in `README`.

Remember to update the app's accordingly to your repository improvements.

I hope this tutorial helps you create your BioMANIA app with your GitHub-hosted package. If you have any further questions or need assistance with specific steps, feel free to ask!