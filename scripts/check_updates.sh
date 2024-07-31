#!/bin/bash

# Initialize variables
version_location="./prompting/__init__.py"
version="__version__"
proc_name="s1_validator_main_process" 
old_args=$@
branch=$(git branch --show-current)  # get current branch.

# Function definitions (same as in run.sh)
version_less_than_or_equal() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

get_version_difference() {
    local tag1="$1"
    local tag2="$2"

    local version1=$(echo "$tag1" | sed 's/v//')
    local version2=$(echo "$tag2" | sed 's/v//')

    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}

        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

read_version_value() {
    while IFS= read -r line; do
        if [[ "$line" == *"$version"* ]]; then
            local value=$(echo "$line" | awk -F '=' '{print $2}' | tr -d ' ')
            strip_quotes $value
            return 0
       
