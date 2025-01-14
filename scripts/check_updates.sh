#!/usr/bin/env bash

INTERVAL=300

REMOTE_BRANCH="main"

get_version_from_pyproject() {
    local file_path="$1"
    # Using grep + cut:
    grep '^version\s*=' "$file_path" 2>/dev/null | \
        sed -E 's/^version\s*=\s*"([^"]+)".*$/\1/'
}

while true
do
    echo "[autoupdater.sh] Checking for updates..."

    LOCAL_VERSION="$(get_version_from_pyproject './pyproject.toml')"

    git fetch origin "$REMOTE_BRANCH" >/dev/null 2>&1

    REMOTE_VERSION="$(git show "origin/$REMOTE_BRANCH:pyproject.toml" | \
        grep '^version\s*=' | sed -E 's/^version\s*=\s*"([^"]+)".*$/\1/')"

    if [ -n "$LOCAL_VERSION" ] && [ -n "$REMOTE_VERSION" ]; then
        if [ "$LOCAL_VERSION" != "$REMOTE_VERSION" ]; then
            echo "[autoupdater.sh] New version detected! Local=$LOCAL_VERSION, Remote=$REMOTE_VERSION."
            echo "[autoupdater.sh] Pulling new changes..."

            git pull origin "$REMOTE_BRANCH"

            echo "[autoupdater.sh] Re-launching run.sh..."
            exec ./run.sh

        else
            echo "[autoupdater.sh] Already up to date. Local=$LOCAL_VERSION, Remote=$REMOTE_VERSION."
        fi
    else
        echo "[autoupdater.sh] Could not determine versions. Local=$LOCAL_VERSION, Remote=$REMOTE_VERSION."
    fi

    sleep "$INTERVAL"
done