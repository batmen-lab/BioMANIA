# download data and retriever model
pip install gdown==5.1.0
gdown https://drive.google.com/uc?id=1kv6-P_Pw4CMUZjAN8LYtv_xoMXu6Q0hA -O your_lib
unzip your_lib
rm -rf your_lib

gdown https://drive.google.com/uc?id=1l2VCr7UuXiZfNjoePCFBgtSRgizpFSa7 -O your_second_file
tar -xzvf your_second_file
rm -rf your_second_file
# ../../resources
gdown https://drive.google.com/uc?id=18ZMorTuFF-Q8_w-V9yTlLsCkw6ONyuss -O your_third_file
unzip your_third_file
rm -rf your_third_file
mv resources ../../resources

# data/other_data
gdown https://drive.google.com/uc?id=12wovrXBP7UxKXVKxGoRH1M4yTkm_SHY_ -O your_fourth_file
unzip your_fourth_file
rm -rf your_fourth_file
# data/conversations
gdown https://drive.google.com/uc?id=111jnPEzPD6BGMDbVDfO9vpk4g1l8c1XE -O your_fifth_file
unzip your_fifth_file
rm -rf your_fifth_file
# data/standard_process/base
gdown https://drive.google.com/uc?id=1_iNt0kNepFoi_fHckHCKMJ9fnj7JwaJf -O your_sixth_file
tar -xzvf your_sixth_file
mkdir data
mkdir data/standard_process
mv base data/standard_process/
rm -rf your_sixth_file
