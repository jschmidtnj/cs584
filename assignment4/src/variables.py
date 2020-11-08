#!/usr/bin/env python3
"""
global user-defined variables
"""

data_folder: str = 'data'
raw_data_folder: str = f'{data_folder}/raw_data'
part_1_data_folder: str = f'{raw_data_folder}/q1'
clean_data_folder: str = f'{data_folder}/clean_data'
models_folder = f'{data_folder}/models'
output_folder: str = 'output'
cnn_folder = f'{models_folder}/cnn'
text_vectorization_folder = f'{models_folder}/vectorization'
cnn_file_name = 'cp.ckpt'

random_state: int = 0

paragraph_key: str = 'paragraph'
label_key: str = 'label'
class_key: str = 'class'
