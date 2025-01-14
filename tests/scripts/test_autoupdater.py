import os
import shutil
import subprocess
import time

import pytest


def test_autoupdater_script():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    autoupdater_path = os.path.join(project_root, "scripts", "autoupdater.sh")
    test_script_path = os.path.join(project_root, "scripts", "test_autoupdater.sh")

    # Check if the scripts exist
    assert os.path.exists(test_script_path), f"Test script not found at {test_script_path}"
    assert os.path.exists(autoupdater_path), f"Autoupdater script not found at {autoupdater_path}"

    # Create test directory
    test_dir = os.path.join(project_root, "updater_test")
    os.makedirs(test_dir, exist_ok=True)

    try:
        # Copy autoupdater.sh to test directory
        shutil.copy2(autoupdater_path, os.path.join(test_dir, "autoupdater.sh"))

        # Run the test script
        process = subprocess.Popen(
            [test_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root,  # Run from project root
        )

        # Wait for the script to finish
        timeout = 60
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.kill()
                pytest.fail("Script timed out after 60 seconds")
            time.sleep(1)

        # Capture the output
        stdout, stderr = process.communicate()

        # Print output for debugging
        print("STDOUT:", stdout)
        print("STDERR:", stderr)

        # Assert that the script ran successfully
        assert process.returncode == 0, f"Script failed with error: {stderr}"

        # Check specific outputs in stdout
        assert "✅ Test passed!" in stdout, "The test did not pass as expected."

    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
