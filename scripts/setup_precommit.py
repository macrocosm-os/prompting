import os
import subprocess


def write_to_pre_commit(content):
    hooks_dir = ".git/hooks"
    pre_commit_path = os.path.join(hooks_dir, "pre-commit")

    if os.path.exists(hooks_dir):
        with open(pre_commit_path, "w") as pre_commit_file:
            try:
                pre_commit_file.write(content)
                print("Success: .git/hooks directory was found. Pre-commit hooks set.")
            except Exception as e:
                print(f"Error setting up pre-commit hooks: {e}")

        # Add execute permission to the pre-commit hook file
        try:
            subprocess.run(["chmod", "+x", pre_commit_path], check=True)
            print("Success: Pre-commit hook file set to executable.")
        except subprocess.CalledProcessError as e:
            print(f"Error setting execute permission on pre-commit hook file: {e}")
    else:
        print("Warning: .git/hooks directory not found. Pre-commit hook was not set.")


# Example content to write to pre-commit hook
pre_commit_content = """
#!/bin/bash

# Run Black formatting on staged Python files with specific parameters
git diff-index --cached --name-only --diff-filter=ACMRTUXB HEAD | grep '\.py$' | xargs black

# Add the formatted files to the staging area
git diff --name-only --cached | xargs git add

# Continue with the commit
exit 0
"""

write_to_pre_commit(pre_commit_content)
