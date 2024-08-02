poetry install --extras "validator"
poetry run pip uninstall -y uvloop
poetry runn python neurons/validator.py
