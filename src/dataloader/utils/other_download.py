import subprocess
import os
from configs.model_config import READTHEDOC_PATH, API_HTML, TUTORIAL_GITHUB, ANALYSIS_PATH, LIB
from urllib.parse import urlparse

def download_readthedoc(readthedoc_path, api_html):
    """name = urlparse(READTHEDOC_LINK).netloc
    if not READTHEDOC_LINK.startswith(('http://', 'https://')):
        name = READTHEDOC_LINK.split('/')[0]
    new_filepath = os.path.join(READTHEDOC_PATH,name)
    if os.path.exists(new_filepath):
        print('file exists! Need not url download!')
        return new_filepath"""
    if readthedoc_path and api_html:
        # wget download 
        command = "wget -N -c --no-parent --convert-links --accept-regex '(stable|tutorial|latest|Tutorial)' -l 1 -A.html -r -np -P "+readthedoc_path+ " " +api_html
        os.system(command)
        print('==>Finished downloading readthedoc files!')
    else:
        print('==>Did not provide readthedoc_path or api_html url, skip downloading readthedoc!')

def download_tutorial(tutorial_github, analysis_path, lib_name):
    """
    Clone a GitHub repository to a local directory.

    Parameters:
    - github_url: str. The URL of the GitHub repository.
    - local_dir: str. The local directory where to clone the repository. 
    """
    if tutorial_github:
        github_url = tutorial_github
        local_dir = os.path.join(analysis_path, lib_name,'Git_Tut')
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        result = subprocess.run(['git', 'clone', github_url, local_dir])
        if result.returncode != 0:
            print(f"Error cloning {github_url}. Return code: {result.returncode}")
        else:
            print(f"Successfully cloned {github_url} into {local_dir}")
    else:
        print('==>Did not provide tutorial url, skip downloading tutorial!')

if __name__ == "__main__":
    download_readthedoc(READTHEDOC_PATH, API_HTML)
    download_tutorial(TUTORIAL_GITHUB, ANALYSIS_PATH, LIB)