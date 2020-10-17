#!/bin/bash

# note - script currently works for python and node.
# does not update java dependencies - this needs to be done manually.

# abort on errors
set -e

cd ..

python_paths=("assignment1" "assignment2" "assignment3")

source $(conda info --base)/etc/profile.d/conda.sh
for path in "${python_paths[@]}"
do
  cd "$path"
  env_name=$(grep 'name:' environment.yml | cut -d ' ' -f 2)
  conda activate $env_name
  conda env update --file environment.yml --prune
  conda env export --no-builds | grep -v "^prefix: " > environment.yml
  conda deactivate
  cd -
done

cd scripts
