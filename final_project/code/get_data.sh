#!/bin/bash

set -e

source $(conda info --base)/etc/profile.d/conda.sh

env_name=$(grep 'name:' environment.yml | cut -d ' ' -f 2)
conda activate $env_name

cd data/raw_data
rm -rf *
kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
kaggle datasets download -d takuok/glove840b300dtxt
unzip '*.zip'
rm *.zip

cd -
