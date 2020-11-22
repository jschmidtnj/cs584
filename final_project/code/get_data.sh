#!/bin/bash

set -e

cd data/raw_data
rm -rf *
kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
unzip *.zip
rm *.zip

cd -
