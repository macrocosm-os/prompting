# Contributing

## Contribute to prompting

Feel like our codebase is missing a feature? We welcome your pull requests!

Issues labeled [good first issue](https://github.com/macrocosm-os/prompting/labels/good%20first%20issue) can be good first contributions, and will help get you familiar with the codebase.

Few pointers for contributions:

- Create your PR against the `staging` branch, not `main`.
- New features need to contain unit tests, must conform to CI/CD style guides and should be documented with the introduction PR.
- PR's can be declared as `Draft` - which signify Work in Progress Pull Requests (which are not finished).

If you are unsure, discuss the feature on our [SN1 server (alpha)](https://discord.gg/bittensor) or in an [issue](https://github.com/macrocosm-os/prompting/issues) before a Pull Request.

## Before sending the PR

### 1. Run pre-commit hook

All unit tests, linters, and formatters must pass. If something is broken, change your code to make it pass.
It means you have introduced a regression or violated the code style.

```bash
poetry run pre-commit run --all-files
```

### 2. Run validator on testnet

Run validator on testnet and analyze the logs for any potential issues.
```bash
# Modify .env.validator with the required fields.
bash install.sh && bash run.sh
```

## Creating the PR

### 1. Write concise PR description
Recommended example of template for PR description:
```markdown
## Changes
  - Add some new feature;
  - Fix issue with some function;
  - Implement new sorting algorithm.
```

### 2. Check remote CI/CD PR status
All Github Actions checks must pass.
