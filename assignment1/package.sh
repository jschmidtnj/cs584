#!/bin/bash

set -e

output=output.zip

rm -f "$output"

zip -r "$output" *.md environment.yml src proof -x src/__pycache__/\* src/.ipynb_checkpoints/\*
