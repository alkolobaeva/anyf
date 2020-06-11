#! /bin/bash
echo 'Create environment...'
conda create --name af python=3.7
conda activate af
echo 'Installing requirements...'
pip install -r requirements.txt
cd code
echo 'Training the model...'
jupyter nbconvert --to python train.ipynb
python train.py
cd ..
echo 'Starting API...'
python code/api.gyp