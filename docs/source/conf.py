import os
import sys
from datetime import datetime
from pathlib import Path

# Setting up the project root and source code path
HERE = Path(__file__).parent
ROOT = HERE.parent  # This should be the 'docs' directory
print(ROOT.parent)
sys.path.insert(0, str(ROOT.parent))  # Adds your source code to sys.path

# -- Project information -----------------------------------------------------
project = 'BioMANIA'
copyright = f'{datetime.now().year}, BioMANIA Team'
author = 'BioMANIA Team'
release = '0.1.0'  # The short X.Y version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    #'sphinx.ext.viewcode',
    #'sphinx.ext.autosummary',
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_default_options = {
    'members': True,
    'inherited-members': True
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    '../../**/**.ipynb_checkpoints', 
    '../../**/__pycache__',
    '../src/autocoop/**',
    '**/Git2APP/**',
    '**/analysis/**',
    '**/R2APP/**',
    '**/build/**',
    '**/dist/**',
    '**/gpt_bak_240416/**',
    '**/output/**',
    '**/plot/**',
    '**/tmp/**',
    '**/scripts/**',
    '**/deploy/**'
]


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "../../images/BioMANIA.png"
html_theme_options = {
    'navigation_depth': 4,
    'logo_only': True,
    'display_version': False,
}

# Setting GitHub URL for Edit on GitHub link
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "batmen-lab",  # Username
    "github_repo": "BioMANIA",  # Repo name
    "github_version": "main/",  # Version
    "conf_py_path": "docs/source/",  # Path in the checkout to the docs root
}

# -- Extension configuration -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

import subprocess
def run_apidoc(app):
    output_dir = "source/"
    module_dir = "../src"
    cmd = [
        "sphinx-apidoc",
        "-o", output_dir,
        module_dir,
        "-H", "BioMANIA", "[", 
    ] + ['"'+i+'"' for i in exclude_patterns] + ["]"]
    subprocess.call(cmd)
    #rename_and_replace_contents(output_dir)
    #update_toctree()

from pathlib import Path
import re

def rename_and_replace_contents(output_dir):
    p = Path(output_dir)
    for rst_file in p.glob("**/*.rst"):
        with open(rst_file, "r") as file:
            content = file.read()
        content = re.sub(r'\bBioMANIA\b', 'src', content)
        with open(rst_file, "w") as file:
            file.write(content)
        if 'src' in rst_file.stem:
            new_name = rst_file.name.replace('BioMANIA', 'src', )
            rst_file.rename(rst_file.parent / new_name)

def update_toctree():
    main_doc_path = Path("source/index.rst")
    rst_files = [str(file.relative_to("source")) for file in Path("source").rglob("*.rst")]
    with open(main_doc_path, "r") as file:
        content = file.read()
    toctree_start = content.find(".. toctree::")
    toctree_end = content.find("\n\n", toctree_start)
    new_toctree_content = content[:toctree_end] + "\n" + "\n".join([f"   {file}" for file in rst_files]) + "\n" + content[toctree_end:]
    with open(main_doc_path, "w") as file:
        file.write(new_toctree_content)

def setup(app):
    app.connect('builder-inited', run_apidoc)
    #app.connect('source-read', ultimate_replace)
