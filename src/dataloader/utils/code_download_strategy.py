import os
import subprocess
import importlib
from configs.model_config import LIB, LIB_ALIAS, GITHUB_LINK, GITHUB_PATH

# Download Strategy
class DownloadStrategy:
    def download(self, lib_name):
        pass

class PipInstall(DownloadStrategy):
    def download(self, lib_name):
        try:
            subprocess.check_call(['pip', 'install', lib_name])
        except Exception as e:
            print(f"An error occurred while installing {lib_name} using pip: {str(e)}")

class CondaInstall(DownloadStrategy):
    def download(self, lib_name):
        try:
            subprocess.check_call(['conda', 'install', lib_name])
        except Exception as e:
            print(f"An error occurred while installing {lib_name} using conda: {str(e)}")

class GitInstall(DownloadStrategy):
    def download(self, lib_link, lib_github_path):
        try:
            subprocess.check_call(['git', 'clone', '--depth', '1', lib_link], cwd=lib_github_path)
            name = lib_link.split('/')[-1]
            subprocess.check_call(['pip', 'install', os.path.join(lib_github_path, name)])
        except Exception as e:
            print(f"An error occurred while installing {lib_link} using git: {str(e)}")

def get_lib_localpath(lib_name, lib_alias):
    try:
        exec(f'import {lib_alias}')
        lib_path = eval(f'os.path.dirname({lib_alias}.__file__)')
    except Exception as e:
        print(f"An error occurred while getting the local path of {lib_name}: {str(e)}")
        lib_path = None
    return lib_path

def download_lib(strategy_type, lib_name, lib_link, lib_alias, github_path):
    lib_github_path = os.path.join(github_path,lib_name)
    strategies = {
        "pip": PipInstall(),
        "git": GitInstall(),
        "conda":CondaInstall(),
    }
    strategy = strategies.get(strategy_type)
    if not strategy:
        raise ValueError("Invalid strategy type. Choose from ", list(strategies.keys()))

    try:
        importlib.import_module(lib_alias)
        print(f"Library {lib_name} is already installed. Skip downloading again!")
    except ImportError:
        print(f"Library {lib_name} is not installed. Start downloading now...")
        if strategy_type == 'git':
            strategy.download(lib_link, lib_github_path)
        else:
            strategy.download(lib_name)
    lib_path = get_lib_localpath(lib_name,lib_alias)
    return lib_path

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    lib_path = download_lib('git', LIB, GITHUB_LINK, LIB_ALIAS, GITHUB_PATH)
    print('downloaded in path: ', lib_path)
