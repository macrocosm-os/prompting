<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <source srcset="./assets/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

# **Bittensor SN1** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

This repository is the **official codebase for Bittensor Subnet 1 (SN1) v1.0.0+, which was released on 22nd January 2024**. To learn more about the Bittensor project and the underlying mechanics, [read here.](https://docs.bittensor.com/).

# Introduction

This repo defines an incentive mechanism to create a distributed conversational AI for Subnet 1 (SN1).

Validators and miners are based on large language models (LLM). The validation process uses **internet-scale datasets and goal-driven behaviour to drive human-like conversations**.


</div>

# Usage

<div align="center">

**[For Validators](./docs/validator.md)** · **[For Miners](./docs/epistula_miner.md)** · **[API Documentation]((./docs/API_docs.md))**


</div>

# Agentic Tasks

Subnet one utilizes the concept of "Tasks" to control the behavior of miners. Validator create a variety of tasks, which include a "challenge" for the miner to solve, and sends them to 100 miners, scoring all the completions they send back.

## Task Descriptions

### 1. **QA (Question Answering)**
The miner receives a question about a specific section from a Wikipedia page. The miner must then find the original context in the specified section and use it to return an accurate answer. References are generated using the validators privileged knowledge of the context, and miner complestions are scored based on similarity metrics.

### 2. **Inference**
A question is given with some pre-seeded information and a random seed. The miner must perform an inference based on this information to provide the correct answer. Completions are scored based on similarity metrics.

### 3. **MultiChoice**
The miner is presented with a question from Wikipedia along with four possible answers (A, B, C, or D). The miner must search Wikipedia and return the correct answer by selecting one of the given options. Miner completions are scored by Regex matching.

### 5. **Programming**
The miner receives a code snippet that is incomplete. The task is to complete the code snippet to perform its intended function. The validator generates a reference using it's internal LLM, and the miner is scored based on its similarity to this reference.

### 6. **Web Retrieval**
The miner is given a question based on a random web page and must return a scraped website that contains the answer. This requires searching the web to locate the most accurate and reliable source to provide the answer. The miner is scored based on the embedding similarity between the answer it returns and the original website that the validator generated the reference from.

### 7. **Multistep Reasoning**
The miner is given a complex problem that requires multiple steps to solve. Each step builds upon the previous one, and the miner must provide intermediate results before arriving at the final answer. The validator generates a reference solution using its internal LLM, and the miner is scored based on the accuracy and coherence of the intermediate and final results.

# API Documentation

For detailed information on the available API endpoints, request/response formats, and usage examples, please refer to the [API Documentation](./docs/API_docs.md).

# Contribute
<div align="center">

**[Contribution guide](./assets/CONTRIBUTING.md)**

</div>
