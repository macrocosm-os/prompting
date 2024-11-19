## Epistula Usage


You will need to initialize an `StreamingEpistulaClient` in order to send streaming requests to miners. It can send both streaming and non-streaming requests using the `send_streaming_request` and `send_request` methods respectively.

You then create a Synapse (which is simply a pydantic model) that you can send to the miner.

```python
epistula_client = StreamingEpistulaClient(wallet=bt.wallet(name="validator", hotkey="validator_hotkey"), metagraph=bt.subtensor().metagraph(netuid=1), mode="mock")

synapse = StreamPromptingSynapse(messages=["This is a test message that should be echoed back"], task_name="test", roles=["user"])
results = await StreamResultsParser().parse_streaming_response(epistula_client.send_streaming_request(synapse=synapse, miner_uids=[1,2,3]), synapse=synapse)
```

The miner will then response either with a modified Synapse (same json as sent) in the case of a non-streaming request.

In case of a streaming request, the miner will respond with a stream of {"uid": int, "chunk": str} objects. Which can be parsed using the `StreamResultsParser` as shown above.
