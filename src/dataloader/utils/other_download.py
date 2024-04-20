import subprocess, argparse, os
from configs.model_config import READTHEDOC_PATH, ANALYSIS_PATH, get_all_variable_from_cheatsheet
from urllib.parse import urlparse, unquote
from typing import Union, List, Optional

def download_readthedoc(readthedoc_path: str, api_html: Union[str, List[str]], source_type: str = 'single') -> None:
    """
    Download ReadTheDocs HTML pages for API documentation and tutorials.

    Parameters
    ----------
    readthedoc_path : str
        The directory path where to save downloaded files.
    api_html : str | list
        The URL(s) of the HTML page(s) to download.
    source_type : str, optional
        'single' for a single HTML file, 'full' for a full website. Default is 'single'.
    """
    def download_single_html(api_html_url):
        # Construct wget command based on the source_type
        if source_type == 'single':
            command = f"wget -N -c --no-parent --convert-links --accept-regex '(stable|tutorial|latest|Tutorial)' -l 1 -A.html -r -np -P {readthedoc_path} {api_html_url}"
        else:  # For a full website
            if not api_html_url.startswith(('http://', 'https://')):
                api_html_url = 'https://' + api_html_url
            command = f"wget -N -c --no-parent --convert-links -r -l inf -np -A.html -P {readthedoc_path} {api_html_url}"
        # Execute the download command
        os.system(command)
    if readthedoc_path and api_html:
        if isinstance(api_html, list):  # Handle api_html as a list of URLs
            for url in api_html:
                download_single_html(url)
            print('==>Finished downloading multiple readthedoc files!')
        else:  # Handle api_html as a single URL
            download_single_html(api_html)
            print('==>Finished downloading readthedoc file!')
    else:
        print('==>Did not provide readthedoc_path or api_html url, skip downloading readthedoc!')

def extract_repo_details(github_url: str) -> Optional[str]:
    """
    Extract user_name and user_repo from a GitHub repository URL.

    Parameters
    ----------
    github_url : str
        The URL of the GitHub repository to analyze.

    Returns
    -------
    str | None
        Returns the formatted string of user_name and user_repo if available, otherwise None.
    """
    if not github_url.startswith("http"):
        github_url = "https://" + github_url
    parsed_url = urlparse(github_url)
    path = parsed_url.path
    clean_path = path.strip("/").rstrip(".git")
    parts = clean_path.split("/")
    if len(parts) >= 2:
        user_name, user_repo = parts[-2], parts[-1]
        return f"{user_name}_{user_repo}"
    else:
        return None

def download_tutorial(tutorial_github: Union[str, List[str]], analysis_path: str, lib_name: str) -> None:
    """
    Clone a GitHub repository to a local directory.

    Parameters
    ----------
    tutorial_github : str | list
        The URL(s) of the GitHub repository or repositories.
    analysis_path : str
        The local directory where to clone the repository.
    lib_name : str
        The library name associated with the repository.
    """
    if not tutorial_github:
        print('==>Did not provide tutorial url, skip downloading tutorial!')
        return
    
    if not isinstance(tutorial_github, list):
        tutorial_github = [tutorial_github]
    
    for github_url in tutorial_github:
        repo_details = extract_repo_details(github_url)
        if repo_details:
            local_dir = os.path.join(analysis_path, lib_name, 'Git_Tut', repo_details)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            result = subprocess.run(['git', 'clone', github_url, local_dir])
            if result.returncode != 0:
                print(f"Error cloning {github_url}. Return code: {result.returncode}")
            else:
                print(f"Successfully cloned {github_url} into {local_dir}")
        else:
            print(f"Unable to extract repository details from URL: {github_url}")

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='PyPI tool')
    args = parser.parse_args()
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    API_HTML, TUTORIAL_GITHUB, TUTORIAL_HTML = [info_json[key] for key in ['API_HTML', 'TUTORIAL_GITHUB', 'TUTORIAL_HTML']]
    download_readthedoc(READTHEDOC_PATH, API_HTML, source_type='single')
    download_readthedoc(ANALYSIS_PATH, TUTORIAL_HTML, source_type='full')
    download_tutorial(TUTORIAL_GITHUB, ANALYSIS_PATH, args.LIB)