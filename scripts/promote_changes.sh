#!/bin/bash

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
NC="\033[0m" # No Color

function echo_info {
    echo -e "${CYAN}[INFO ]${NC} $1"
}

function echo_good_news {
    echo -e "${GREEN}[GOOD ]${NC} $1"
}

function echo_warning {
    echo -e "${YELLOW}[OOPS ]${NC} $1"
}

function echo_error {
    echo -e "${RED}[ERROR]${NC} $1"
}

REMOTE_ORIGIN_SSH="SSH"
REMOTE_ORIGIN_HTTP="HTTP"
REMOTE_ORIGIN_UNEXPECTED="UNEXPECTED"
function get_remote_origin_type {
    case "$1" in
        https:*)
            echo $REMOTE_ORIGIN_HTTP
        ;;
        git@github.com:*)
            echo $REMOTE_ORIGIN_SSH
        ;;
        *)
            echo $REMOTE_ORIGIN_UNEXPECTED
        ;;
    esac
}

REMOTE_HTTP="https://github.com/opentensor/prompting.git"
REMOTE_SSH="git@github.com:opentensor/prompting.git"
REMOTE_NAME="tmp-promote-changes"

function check_if_remote_exists {
    URL_COUNT="$(git remote -v | grep -c $REMOTE_NAME)"
    if (( URL_COUNT > 0 )); then
        echo "yes"
    else
        echo "no"
    fi
}

echo_info "Checking git remote origin..."
EXPECTE_REMOTE_ORIGIN="private"

ORIGIN_BRANCH="$(git remote -v | grep $EXPECTE_REMOTE_ORIGIN | grep push | awk "{ print $2 }")"
case "$ORIGIN_BRANCH" in
    *opentensor/prompting-private*)
        echo_good_news "Correct remote origin"
    ;;
    *)
        echo_error "Unexpected remote origin"
        echo_info "finishing execution..."
        exit 1
    ;;
esac


ORIGIN_TYPE="$(get_remote_origin_type "$ORIGIN_BRANCH")"
case $ORIGIN_TYPE in
    "$REMOTE_ORIGIN_HTTP")
    REMOTE_URL=$REMOTE_HTTP
    ;;
    "$REMOTE_ORIGIN_SSH")
    REMOTE_URL=$REMOTE_SSH
    ;;
esac

echo_info "Creating git remote "$REMOTE_NAME" "$REMOTE_URL" if not exists"

REMOTE_ALREADY_EXIST="$(check_if_remote_exists)"
if [ "$REMOTE_ALREADY_EXIST" == "no" ]; then
    git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

REMOTE_ALREADY_EXIST="$(check_if_remote_exists)"
if [ "$REMOTE_ALREADY_EXIST" == "no" ]; then
    echo_error "Something went wrong, unable to create git remote $REMOTE_NAME"
fi

echo_good_news "Remote exists "$REMOTE_NAME" -> $REMOTE_URL"

USER_NAME="$(git config user.name)"
if [ -z "$USER_NAME" ]; then
    USER_NAME="unknown"
fi

TS=$(date +%Y_%m_%d)
BRANCH_TO_PUSH="promote-changes/$TS/$USER_NAME"

echo_info "Pushing changes to remote branch "$BRANCH_TO_PUSH""
git push "$REMOTE_NAME" "main:$BRANCH_TO_PUSH"

case $OSTYPE in
    "linux-gnu"*)
        xdg-open "https://github.com/opentensor/prompting/pull/new/$BRANCH_TO_PUSH"
    ;;
    "darwin"*)
        open "https://github.com/opentensor/prompting/pull/new/$BRANCH_TO_PUSH"
    ;;
    *)
        echo_warning "OS type "$OSTYPE" not supported."
        echo_good_news "You can open the PR with the following URL: https://github.com/opentensor/prompting/pull/new/$BRANCH_TO_PUSH"
    ;;
esac