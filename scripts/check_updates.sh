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
        fi
    done < "$version_location"

    echo ""
}

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)
    
    if [[ "$os_name" == "Linux" ]]; then
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
        if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system"
        return 0
    fi
}

check_variable_value_on_github() {
    local repo="$1"
    local file_path="$2"
    local variable_name="$3"

    local url="https://api.github.com/repos/$repo/contents/$file_path"
    local response=$(curl -s "$url")

    if [[ $response =~ "message" ]]; then
        echo "Error: Failed to retrieve file contents from GitHub."
        return 1
    fi

    local content=$(echo "$response" | tr -d '\n' | jq -r '.content')

    if [[ "$content" == "null" ]]; then
        echo "File '$file_path' not found in the repository."
        return 1
    fi

    local decoded_content=$(echo "$content" | base64 --decode)
    local variable_value=$(echo "$decoded_content" | grep "$variable_name" | awk -F '=' '{print $2}' | tr -d ' ')

    if [[ -z "$variable_value" ]]; then
        echo "Variable '$variable_name' not found in the file '$file_path'."
        return 1
    fi

    strip_quotes $variable_value
}

strip_quotes() {
    local input="$1"
    local stripped="${input#\"}"
    stripped="${stripped%\"}"
    echo "$stripped"
}

# Check if packages are installed.
check_package_installed "jq"
if [ "$?" -eq 1 ]; then
    while true; do
        if [ -d "./.git" ]; then
            latest_version=$(check_variable_value_on_github "macrocosm-os/prompting" "prompting/__init__.py" "__version__ ")

            current_version=$(read_version_value)
            if version_less_than $current_version $latest_version; then
                echo "latest version $latest_version"
                echo "current version $current_version"
                diff=$(get_version_difference $latest_version $current_version)
                if [ "$diff" -eq 1 ]; then
                    echo "current validator version:" "$current_version" 
                    echo "latest validator version:" "$latest_version" 

                    if git pull origin $branch; then
                        echo "New version published. Updating the local copy."

                        poetry install

                        pm2 del auto_run_validator
                        echo "Restarting PM2 process"
                        pm2 restart $proc_name

                        current_version=$(read_version_value)
                        echo ""

                        echo "Restarting script..."
                        ./$(basename $0) $old_args && exit
                    else
                        echo "**Will not update**"
                        echo "It appears you have made changes on your local copy. Please stash your changes using git stash."
                    fi
                else
                    echo "**Will not update**"
                    echo "The local version is $diff versions behind. Please manually update to the latest version and re-run this script."
                fi
            else
                echo "**Skipping update **"
                echo "$current_version is the same as or more than $latest_version. You are likely running locally."
            fi
        else
            echo "The installation does not appear to be done through Git. Please install from source at https://github.com/macrocosm-os/validators and rerun this script."
        fi
        
        sleep 1800
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi
