#!/bin/bash

set -e

output=Assignment4_Schmidt_Joshua.zip

rm -f "$output"

report_file=report.pdf

pandoc --pdf-engine=xelatex -o "$report_file" report.md

zip -r "$output" *.md environment.yml src output "$report_file" \
  data/clean_data/.gitignore data/raw_data/.gitignore data/models/.gitignore \
  -x \*\*/__pycache__/\* \*\*/.ipynb_checkpoints/\*
