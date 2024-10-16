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
pip install git+https://github.com/your_github_page/your_repository.git
# Or build locally
pip install  -e .
```

## Step 2: Use BioMANIA

Now that your GitHub repository is properly installed, you can create a BioMANIA app to use it. Just treat your library as a PyPI package, and follow the steps in [the steps in readme file.](../docs/PyPI2APP.md), start from adding the information to the `Lib_cheatsheet.py`.

### Tips: Refine Function Definitions and Docstrings:
(Optional) If you need to add docstrings in your source code and refine the function definitions. You can either choose using GPT or design by yourself. The latter is better as it is more accurate. `NumPy` format is preferred than `reStructuredText` and `Google` format. Here's a basic example of an effective docstring :

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

You can refer to the prompts available in [BioMANIA](https://www.biorxiv.org/content/10.1101/2023.10.29.564479) to add the function body, or either using the [prompt](./src/Git2APP/get_API_docstring_from_sourcecode.py) that modified based on that.

If you already have a well-documented code, just ignore this step and follow the steps in readme file.

### Tips: Add logo to UI

Add a logo image to `BioMANIA/chatbot_ui_biomania/public/apps/` and modify the link in `BioMANIA/chatbot_ui_biomania/components/Chat/LibCardSelect.tsx`.

Be mindful of the capitalization in library names, as it affects the recognition of the related model data loading paths.

### Tips: Use UI service.

Follow the steps in [`Run with script/Inference`](../README.md#inference) section in `README` to start UI service. Don’t forget to set an OpenAI key in `.env` file as recommended in `README`.

Remember to update the app's accordingly to your repository improvements.

### Tips: Share your APP!

If you want your app to be used by others, follow the steps in [`Share your APP`](../README.md#share-your-app) section in `README` to introduce your tool to others!


I hope this tutorial helps you create your BioMANIA app with your GitHub-hosted package. If you have any further questions or need assistance with specific steps, feel free to ask!
