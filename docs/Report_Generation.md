
## Report Generation

BioMANIA can generate various reports, including Python files, Jupyter notebooks, performance summaries, and common issue logs. Follow the instructions in the Report Generation section to create these reports.

### For chat Python File: 

Firstly, press `export chat` button on UI to get the chat json data. Convert the chat JSON into a Python code using the Chat2Py.py script.

```bash
# cd src
python report/Chat2Py.py report/demo_Preprocessing_and_clustering_3k_PBMCs.json
```
![](https://github.com/batmen-lab/BioMANIA/tree/main/images/pyfile.jpg)


### For chat report

Convert the chat JSON into an [ipynb report](https://github.com/batmen-lab/BioMANIA/blob/main/src/report/demo_Preprocessing_and_clustering_3k_PBMCs.ipynb) using the Chat2jupyter.py script.

```bash
# cd src
python report/Chat2jupyter.py report/demo_Preprocessing_and_clustering_3k_PBMCs.json
```
![](https://github.com/batmen-lab/BioMANIA/tree/main/images/jupyter.jpg)


### For performance report

Combine and sort the performance figures into a short report.

```bash
# cd src
python report/PNG2report.py scanpy
```

Please note that the generation of this report must be based on the premise that the retriever models have already been trained, and the gpt baseline has already been tested. You need to first obtain the results of each model before running this script. Here is a reference for a [demo report](https://github.com/batmen-lab/BioMANIA/tree/main/src/report/performance_report.pdf).

![](https://github.com/batmen-lab/BioMANIA/tree/main/images/performance_report.jpg)


### For common issue report

Displaying common issues in the process of converting Python tools into libraries

```bash
# cd src
python report/Py2report.py scanpy
```

The output files are located in the ./report folder.


![](https://github.com/batmen-lab/BioMANIA/tree/main/images/error_category.jpg)


TODO:

We will provide the below files and the data of more tools later

```
report/Py2report.py
```


