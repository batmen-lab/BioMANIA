"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: The script downloads libraries and gets API information from the source code.
"""
import subprocess

"""libraries = [
    "biopython", "qiime2", "eletoolkit", "pyopenms", "pyteomics",
    "scikit-bio", "emperor", "gneiss", "deap", "tskit", "biotite",
    "sklearn-ann", "scenicplus", "scanorama", "anndata", "scikit-misc",
    "statsmodels", "cellpose", "scvelo", "velocyto", "loom", "mygene",
    "gseapy", "shiny", "fairlearn", "magic-impute"
]"""
libraries = ["dynamo", "cellrank", "pertpy", "moscot", "scCODA", "scarches", "qiime2", "fairlearn", "magic-impute"]

with open("error_log.txt", "w") as log_file:
    for lib in libraries:
        try:
            subprocess.run(f"pip install {lib}", shell=True, check=True, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            log_file.write(f"Error installing {lib}: {e.output}\n")
        commands = [
            f"python -m src.dataloader.utils.other_download --LIB {lib}",
            f"python -m src.dataloader.get_API_init_from_sourcecode --LIB {lib}"
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True, text=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                log_file.write(f"Error executing '{cmd}' for {lib}: {e.output}\n")

