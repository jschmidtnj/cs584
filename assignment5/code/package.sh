#!/bin/bash

set -e

output=assignment_5_code_joshua_schmidt.zip

rm -f "$output"

zip -r "$output" *.md environment.yml src output \
  data/clean_data/.gitignore data/raw_data/.gitignore data/models/.gitignore \
  -x \*\*/__pycache__/\* \*\*/.ipynb_checkpoints/\*
