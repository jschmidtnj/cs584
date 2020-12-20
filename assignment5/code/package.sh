#!/bin/bash

set -e

output=final_project_code_joshua_schmidt.zip

rm -f "$output"

zip -r "$output" *.md environment.yml src output "$report_file" get_data.sh \
  data/clean_data/.gitignore data/raw_data/.gitignore data/models/.gitignore \
  -x \*\*/__pycache__/\* \*\*/.ipynb_checkpoints/\*
