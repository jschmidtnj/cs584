#!/bin/bash

set -e

output=Assignment1_Schmidt_Joshua.zip

rm -f "$output"

zip -r "$output" *.md environment.yml src proof data -x src/__pycache__/\* src/.ipynb_checkpoints/\* data/\*.txt
