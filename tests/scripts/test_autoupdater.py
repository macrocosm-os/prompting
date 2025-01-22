import os
import shutil
import subprocess
import time

import pytest

def test_autoupdater_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    autoupdater_path = os.path.join(project_root, "scripts", "autoupdater.sh")
    test_script_path = os.path.join(project_root, "scripts", "test_autoupdater.sh")

    assert os.path.exists(test_script_path), f"Test script not found at {test_script_path}"
    assert os.path.exists(autoupdater_path), f"Autoupdater script not found at {autoupdater_path}"

    test_dir = os.path.join(project_root, "updater_test")
    os.makedirs(test_dir, exist_ok=True)

    try:
        subprocess.run(["git", "config", "--global", "user.name", "AutoUpdater"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "autoupdater@example.com"], check=True)

        remote_dir = os.path.join(project_root, "updater_test_remote")
        if os.path.exists(remote_dir):
            shutil.rmtree(remote_dir)

        shutil.copy2(autoupdater_path, os.path.join(test_dir, "autoupdater.sh"))
        process = subprocess.Popen(
            [test_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root,
        )

        timeout = 60
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.kill()
                pytest.fail("Autoupdater testing script timed out after 60 seconds")
            time.sleep(1)

        stdout, stderr = process.communicate()

        assert process.returncode == 0, f"Script failed with error: {stderr}"
        assert "✅ Test passed!" in stdout, "The test did not pass as expected."

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
