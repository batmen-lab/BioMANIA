# usage: sh step1_data_prepare.sh

# set the LIB
export LIB=ehrapy
# install the LIB
pip install ${LIB}
# get API data from source code
python dataloader/get_API_init_from_sourcecode.py --LIB ${LIB}
# get API_init as placeholder for API_composite
cp -r data/standard_process/${LIB}/API_init.json data/standard_process/${LIB}/API_composite.json
# generate instruction by gpt
python dataloader/preprocess_retriever_data.py --LIB ${LIB}
# (Optional) check whether the test data is correct, valid only for those user who prepare annotated data
python dataloader/check_valid_API_annotate.py --LIB ${LIB}
