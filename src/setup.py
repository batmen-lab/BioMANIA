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
    name='BioMANIA',
    version='0.1',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author='Zhengyuan Dong',
    author_email='z6dong@uwaterloo.ca',
    description='Simplifying bioinformatics data analysis through conversation',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/batmen-lab/BioMANIA",
    package_dir={'BioMANIA': ''},
    packages=['BioMANIA'] + ['BioMANIA.'+p for p in setuptools.find_packages(where="./")],
    license='GPL-3.0',
    install_requires=required,
    dependency_links=dependency_links,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent",
    ],
)
