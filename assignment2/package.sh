#!/bin/bash

set -e

output=Assignment2_Schmidt_Joshua.zip

rm -f "$output"

report_file=report.pdf

pandoc --pdf-engine=xelatex -o "$report_file" report.md

zip -r "$output" *.md environment.yml src proofs "$report_file" \
  output data -x \*\*/__pycache__/\* \*\*/.ipynb_checkpoints/\* data/stanfordSentimentTreebank/\* data/\*.txt
