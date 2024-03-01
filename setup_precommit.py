import os

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
    else:
        print("Warning: .git/hooks directory not found. Pre-commit hook was not set.")

# Example content to write to pre-commit hook
pre_commit_content = """
#!/bin/bash

# Run Black formatting on staged Python files with specific parameters
git diff-index --cached --name-only --diff-filter=ACMRTUXB HEAD | grep '\.py$' | xargs black --line-length=88 --target-version=py38

# Continue with the commit
exit 0
"""

write_to_pre_commit(pre_commit_content)
