# download data and retriever model
mkdir hugging_models
mkdir hugging_models/retriever_model_finetuned
mkdir data
pip install gdown==5.1.0

# retriever model and data
gdown https://drive.google.com/uc?id=1PVsrsKnOdv0VrZw9sjkjsmQTm018MOG5 -O hugging_models/retriever_model_finetuned/your_lib
gdown https://drive.google.com/uc?id=1X-3xnTba9Mxb8SZ8oIEdxhVrGMdFVP-i -O data/your_second_file
gdown https://drive.google.com/uc?id=1NRKLDijLENR1vyQHFNT_vk5lLY4lL1CL -O data/your_third_file
gdown https://drive.google.com/uc?id=1wgYY9CD1hPfqlUUFFqDIeh9l4nEu124t -O data/your_fourth_file
gdown https://drive.google.com/uc?id=15vNIPK8ut8Hudbjwg_G1w0zJjs2gmmRb -O data/your_fifth_file

cd data
unzip your_second_file
unzip your_third_file
unzip your_fourth_file
unzip your_fifth_file
rm -rf your_second_file
rm -rf your_third_file
rm -rf your_fourth_file
rm -rf your_fifth_file
mv base ./standard_process/base
cd ..

cd hugging_models/retriever_model_finetuned
unzip your_lib
rm -rf your_lib
