#!/usr/bin/env bash
set -euo pipefail

# Create a temporary remote repository
setup_test_remote() {
    echo "Setting up test remote repository..."
    TEST_REMOTE="test_remote"
    rm -rf "$TEST_REMOTE"
    mkdir -p "$TEST_REMOTE"

    # Copy current project files (excluding test_remote and .git)
    for item in *; do
        if [[ "$item" != "$TEST_REMOTE" && "$item" != ".git" ]]; then
            cp -r "$item" "$TEST_REMOTE/"
        fi
    done

    cd "$TEST_REMOTE"
    git init --initial-branch=main
    git config user.email "test@example.com"
    git config user.name "Test User"

    # Initialize new git repo
    git add .
    git commit -m "Initial commit"

    # Update version in pyproject.toml
    sed -i.bak 's/version = "[^"]*"/version = "9.9.9"/' pyproject.toml
    git commit -am "Bump version to 9.9.9"

    cd ..
    echo "Test remote ready at: $(pwd)/$TEST_REMOTE"
}

# Update git remote to point to our test repository
update_git_remote() {
    git remote remove origin 2>/dev/null || true
    git remote add origin "$(pwd)/$TEST_REMOTE"
}

# Wait for update to complete
wait_for_update() {
    echo "Waiting for auto-updater to detect and apply changes..."
    local timeout=60
    local start_time=$(date +%s)

    while true; do
        if [[ $(grep -F '"9.9.9"' pyproject.toml 2>/dev/null) ]]; then
            echo "✅ Update successful! Version updated to 9.9.9"
            return 0
        fi

        local current_time=$(date +%s)
        if (( current_time - start_time > timeout )); then
            echo "❌ Update timed out after ${timeout} seconds"
            return 1
        fi

        sleep 2
        echo -n "."
    done
}

# Main test execution
main() {
    # Store original directory
    ORIGINAL_DIR=$(pwd)

    # Setup test environment
    setup_test_remote
    update_git_remote

    echo "Test environment ready. Current version:"
    grep "version" pyproject.toml

    echo -e "\nStarting run.sh - wait for auto-update to detect new version..."

    # Make the update interval shorter for testing
    export UPDATE_CHECK_INTERVAL=10

    # Start run.sh in background
    ./run.sh &

    # Wait for update to complete
    if wait_for_update; then
        echo "Test completed successfully!"
    else
        echo "Test failed!"
        pm2 delete all
        exit 1
    fi

    # Show final status
    echo -e "\nFinal PM2 status:"
    pm2 status

    echo -e "\nTest completed. To clean up:"
    echo "1. Run 'pm2 delete all' to stop all processes"
    echo "2. Run 'rm -rf test_remote' to remove test repository"
}

main
