
# Validation
The design of the network's incentive mechanism is based on two important requirements:

### 1. Validation should mimic human interactions

It is imperative that the validation process engages with miners in the same way as real users. The reasons for this are as follows:
- Miners will compete and continuously improve at performing the validation task(s), and so this fine tuning should be aligned with the goals of the subnet.
- It should not be possible to distinguish between validation and API client queries so that miners always serve requests (even when they do not receive emissions for doing so).

In the context of this subnet, miners are required to be intelligent AI assistants that provide helpful and correct responses to a range of queries.

### 2. Reward models should mimic human preferences

In our experience, we have found that it is tricky to evaluate whether miner responses are high quality. Existing methods typically rely on using LLMs to score completions given a prompt, but this is often exploited and gives rise to many adversarial strategies.

In the present version, the validator produces one or more **reference** answers which all miner responses are compared to. Those which are most similar to the reference answer will attain the highest rewards and ultimately gain the most incentive.

**We presently use a combination of string literal similarity and semantic similarity as the basis for rewarding.**

# Tools
Contexts, which are the basis of conversations, are from external APIs (which we call tools) which ensure that conversations remain grounded in factuality. Contexts are also used to obtain ground-truth answers.

Currently, the tooling stack includes:
1. Wikipedia API
2. StackOverflow
3. mathgenerator

More tooling will be included in future releases.

# Tasks
The validation process supports an ever-growing number of tasks. Tasks drive agent behaviour based on specific goals, such as;
- Question answering
- Summarization
- Code debugging
- Mathematics
 and more.

Tasks contain a **query** (basic question/problem) and a **reference** (ideal answer), where a downstream HumanAgent creates a more nuanced version of the **query**.

# Agents

In order to mimic human interactions, validators participate in a roleplaying game where they take on the persona of **random** human users. Equipped with this persona and a task, validators prompt miners in a style and tone that is similar to humans and drive the conversation in order to reach a pre-defined goal. We refer to these prompts as **challenges**.

Challenges are based on the query by wrapping the query in an agent persona which results in a lossy "one-way" function. This results in challenges that are overall more interesting, and less predictable.

The [diagram below](#validation-diagram) illustrates the validation flow.

#### Our approach innovatively transforms straightforward queries into complex challenges, a process akin to a 'hash function', requiring advanced NLP for resolution. This transformation is crucial for preventing simple lookups in source documents, ensuring that responses necessitate authentic analytical effort.


# Validation Diagram
![sn1 overview](../assets/sn1-overview.png)
