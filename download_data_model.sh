# download data and retriever model
pip install gdown
# hugging_models/retriever_model_finetuned/scanpy
gdown https://drive.google.com/uc?id=1kv6-P_Pw4CMUZjAN8LYtv_xoMXu6Q0hA -O your_lib
unzip your_lib
rm -rf your_lib
# hugging_models/retriever_model_finetuned/squidpy
gdown https://drive.google.com/uc?id=1kYFKWfX5G_pK4nB-QIX_aRXx6y7BZ90c -O your_lib
unzip your_lib
rm -rf your_lib
# hugging_models/retriever_model_finetuned/ehrapy
gdown https://drive.google.com/uc?id=12drPGSJDUFGjw4Pd5diJRE2rlDBsx1iM -O your_lib
unzip your_lib
rm -rf your_lib
# hugging_models/retriever_model_finetuned/snapatac2
gdown https://drive.google.com/uc?id=1xTBwsylUgKe1RwSWKdQDKdvapv0FZosh -O your_lib
unzip your_lib
rm -rf your_lib
# data/standard_process/{4LIBs+base}
# data/autocoop/{4LIBs}
# data/others-data
# data/conversations
gdown https://drive.google.com/uc?id=1MSGBq-xg1gV3Tb9D5S9Mm3qLXY_qh0uM -O data_in_one_file
tar -xzvf data_in_one_file
rm -rf data_in_one_file
# ../../resources
gdown https://drive.google.com/uc?id=18ZMorTuFF-Q8_w-V9yTlLsCkw6ONyuss -O your_third_file
unzip your_third_file
rm -rf your_third_file
mv resources ../../resources
