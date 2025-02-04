#!/usr/bin/env bash
set -euo pipefail

# Test environment setup
TEST_DIR="updater_test"
INTERVAL=5

# Create test environment
setup_test_environment() {
    echo "Setting up test environment..."
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR"

    if [[ ! -f "$TEST_DIR/autoupdater.sh" ]]; then
        cp "scripts/autoupdater.sh" "$TEST_DIR/" || cp "autoupdater.sh" "$TEST_DIR/"
    fi
    chmod +x "$TEST_DIR/autoupdater.sh"

    cd "$TEST_DIR"

    # Initialize main repo
    git init --initial-branch=main
    git config user.email "test@example.com"
    git config user.name "Test User"

    cat > pyproject.toml << EOF
[project]
name = "test-project"
version = "1.0.0"
EOF

    # Create dummy run.sh
    cat > run.sh << EOF
#!/bin/bash
echo "Running version \$(grep '^version' pyproject.toml | cut -d'"' -f2)"
EOF
    chmod +x run.sh

    # Create initial commit
    git add .
    git commit -m "Initial commit"
    git branch -M main

    # Create a clone to simulate remote updates
    cd ..
    git clone "$TEST_DIR" "${TEST_DIR}_remote"
    cd "${TEST_DIR}_remote"

    # Update remote version
    sed -i.bak 's/version = "1.0.0"/version = "1.1.0"/' pyproject.toml
    git commit -am "Bump version to 1.1.0"
    cd "../$TEST_DIR"
    git remote add origin "../${TEST_DIR}_remote"
}

# Clean up test environment
cleanup() {
    echo "Cleaning up..."
    cd ..
    rm -rf "$TEST_DIR" "${TEST_DIR}_remote"
}

# Run the test
run_test() {
    echo "Starting auto-updater test..."

    # Start the auto-updater in background
    UPDATE_CHECK_INTERVAL=$INTERVAL ./autoupdater.sh &
    UPDATER_PID=$!

    # Wait for a few intervals
    echo "Waiting for auto-updater to detect changes..."
    sleep $((INTERVAL * 2))

    # Kill the auto-updater
    kill $UPDATER_PID || true
    wait $UPDATER_PID 2>/dev/null || true

    # Check results
    LOCAL_VERSION=$(grep '^version' pyproject.toml | cut -d'"' -f2)
    if [ "$LOCAL_VERSION" = "1.1.0" ]; then
        echo "✅ Test passed! Version was updated successfully."
    else
        echo "❌ Test failed! Version was not updated (still $LOCAL_VERSION)"
    fi
}

# Main test execution
main() {
    setup_test_environment
    run_test
    cleanup
}

main
