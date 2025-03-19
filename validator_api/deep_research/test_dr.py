from validator_api.deep_research.orchestrator import Orchestrator


orchestrator = Orchestrator()
orchestrator.run(messages=[
    {
        "role": "user",
        "content": "How many marbles fit into a bathtub?"
    }
])






