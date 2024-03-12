# Creating Stream Miners 

Altering your miner code to be compatible with streaming will be a development push for all miners on the network. As such, we want to make this as streamlined as possible to minimize the time needed to go from 0->100%. 

Miner architectures now require that you are running a syncronous `forward` method, with an internal `async _forward` method. The code below provides a basic outline of how the `async _forward` method should be structured. There are two main points here:

1. Adding data to the buffer and sending it when it reaches the config.neuron.streaming_batch_size
2. Sending the final buffer of data if inference is finished, and there are less tokens than the batch size. 

```python
def forward(self, synapse: StreamPromptingSynapse) -> Awaitable:
    async def _forward(
        self,
        **kwargs
        send: Send,
    ):

        buffer = []
        timeout_reached = False

        try:
            for token in streamer:
                buffer.append(token)

                if time.time() - init_time > timeout_threshold:
                    bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                    timeout_reached = True
                    break

                if len(buffer) == self.config.neuron.streaming_batch_size:
                    joined_buffer = "".join(buffer)
                    bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    buffer = []

            if (
                buffer and not timeout_reached
            ):  # Don't send the last buffer of data if timeout.
                joined_buffer = "".join(buffer)
                temp_completion += joined_buffer
                bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,
                    }
                )

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True

    token_streamer = partial(
        _forward,
        self,
        **kwargs,
    )

    return synapse.create_streaming_response(token_streamer)
```

HuggingFace miners require you to run a separate inference thread in the background, add to a queue, and manually clear it at the end of the `async _forward` method. For specific implemenations, please refer to the `prompting/miners/hf_miner.py` for additional details. 

It is **necessary** that forward method of the miner class returns this `synapse.create_streaming_response(token_streamer)`. As seen, the `token_streamer` is a partial function that takes in a `send` packet. This packet will be sent by the bittensor middleware to facilitate the communications between the validator and the miner. 

