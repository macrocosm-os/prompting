
import torch 
from prompting.llm import load_pipeline
from prompting.llm import HuggingFaceLLM
from threading import Thread
from langchain.llms import HuggingFacePipeline

def main():
    model_id = "HuggingFaceH4/zephyr-7b-beta"
    pipe, streamer = load_pipeline(
        model_id=model_id,
        torch_dtype=torch.float16,
        device="cuda",
        is_streamer=True, #You could check this somewhere else to automatically set. 
        )

    system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."

    prompt = "Tell me something interesting about space"

    # messages = [{"content": system_prompt, "role": "system"}]
    # message = messages + [{"content": prompt, "role": "user"}]

    llm = HuggingFaceLLM(
        llm_pipeline=pipe,
        system_prompt=system_prompt,
        max_new_tokens=256,
        temperature=0.70,
        top_k=50,
        top_p=0.95,
    )

    thread = Thread(target=llm.query, kwargs=dict(message = prompt))
    thread.start()

    # buffer = [] 
    # for t in streamer:
    #     buffer.append(t)
    #     if len(buffer) == 12: 
    #         print(buffer)

    async def _prompt():
        ii = 0
        buffer = [] 
        for token in streamer: 
            buffer.append(token)
            if len(buffer) == 12: 
                print(f"{ii} : {buffer}")
                buffer = [] 

        if buffer: 
            print(buffer)
        

    import asyncio
    asyncio.run(_prompt())


if __name__ == "__main__": 
    main()