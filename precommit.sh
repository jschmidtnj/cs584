#!/bin/bash

# abort on errors
set -e

check_changes() {
  if git diff --stat --cached -- "$1" | grep -E "$1"; then
    echo "run precommit for $1"
    return 0
  else
    echo "no changes found for $1"
    return 1
  fi
}

force_run_command="-f"

script_paths=("assignment1/" "assignment2/")

for path in "${script_paths[@]}"
do
  if [ "$1" = "$force_run_command" ] || check_changes "$path" ; then
    cd "$path"
    ./precommit.sh
    cd -
  fi
done

git add -A
