import asyncio
import datetime
import json
import random
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from starlette.responses import StreamingResponse

from shared import settings

shared_settings = settings.shared_settings
from shared.epistula import SynapseStreamResult, query_miners
from validator_api import scoring_queue
from validator_api.api_management import validate_api_key
from validator_api.chat_completion import chat_completion
from validator_api.mixture_of_miners import mixture_of_miners
# from validator_api.test_time_inference import generate_response
from validator_api.utils import filter_available_uids

router = APIRouter()
N_MINERS = 5


import httpx
async def query_deepseek(messages: list, stream: bool = True, max_tokens: int = 1024, temperature: float = 0.7):
    """
    Send requests to the DeepSeek API endpoint.
    
    Args:
        messages (list): List of message dictionaries with role and content
        stream (bool): Whether to stream the response
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        
    Returns:
        Response from the API
    """
    url = "https://chutes-deepseek-ai-deepseek-r1-distill-llama-70b.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer FILL_ME_IN",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "messages": messages,
        "stream": stream,
        "max_new_tokens": max_tokens,
        "temperature": temperature
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response
    
@router.post("/v1/chat/completions")
async def completions(request: Request, api_key: str = Depends(validate_api_key)):
    """Main endpoint that handles both regular and mixture of miners chat completion."""
    try:
        body = await request.json()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))
        if body.get("uids"):
            try:
                uids = [int(uid) for uid in body.get("uids")]
            except:
                pass
        else:
            uids = filter_available_uids(
                task=body.get("task"), model=body.get("model"), test=shared_settings.API_TEST_MODE, n_miners=N_MINERS
            )
        if not uids:
            raise HTTPException(status_code=500, detail="No available miners")

        # Choose between regular completion and mixture of miners.
        if body.get("test_time_inference", False):
            return await test_time_inference(
                body=body, target_uids=body.get("uids", None)
            )
        if body.get("mixture", False):
            return await mixture_of_miners(body, uids=uids)
        else:
            return await chat_completion(body, uids=uids)

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)


@router.post("/web_retrieval")
async def web_retrieval(
    search_query: str,
    n_miners: int = 10,
    n_results: int = 5,
    max_response_time: int = 10,
    api_key: str = Depends(validate_api_key),
    target_uids: list[str] = None,
):
    if target_uids:
        uids = target_uids
        try:
            uids = [int(uid) for uid in uids]
        except:
            pass
    else:
        uids = filter_available_uids(task="WebRetrievalTask", test=shared_settings.API_TEST_MODE, n_miners=n_miners)
        uids = random.sample(uids, min(len(uids), n_miners))

    if len(uids) == 0:
        raise HTTPException(status_code=500, detail="No available miners")

    body = {
        "seed": random.randint(0, 1_000_000),
        "sampling_parameters": shared_settings.SAMPLING_PARAMS,
        "task": "WebRetrievalTask",
        "target_results": n_results,
        "timeout": max_response_time,
        "messages": [
            {"role": "user", "content": search_query},
        ],
    }

    timeout_seconds = 30
    stream_results = await query_miners(uids, body, timeout_seconds)
    results = [
        "".join(res.accumulated_chunks)
        for res in stream_results
        if isinstance(res, SynapseStreamResult) and res.accumulated_chunks
    ]
    distinct_results = list(np.unique(results))
    loaded_results = []
    for result in distinct_results:
        try:
            loaded_results.append(json.loads(result))
            logger.info(f"🔍 Result: {result}")
        except Exception:
            logger.error(f"🔍 Result: {result}")
    if len(loaded_results) == 0:
        raise HTTPException(status_code=500, detail="No miner responded successfully")

    collected_chunks_list = [res.accumulated_chunks if res and res.accumulated_chunks else [] for res in stream_results]
    asyncio.create_task(scoring_queue.scoring_queue.append_response(uids=uids, body=body, chunks=collected_chunks_list))
    return loaded_results

from prompting.llms.apis.llm_messages import LLMMessage, LLMMessages
from prompting.llms.apis.llm_wrapper import LLMWrapper
from shared.epistula import make_openai_query


wrapper = LLMWrapper()
async def make_gpt_query(messages, model: str = None):
    return wrapper.chat_complete(
        messages=LLMMessages(*[LLMMessage(role=m["role"], content=m["content"]) for m in messages]),
        model=model,
    )

from neurons.miners.epistula_miner.web_retrieval import get_websites_with_similarity
    
async def conduct_deep_research(body, model: str = None, target_uids: list[str] = None):
        format_message = lambda msg: f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'index': 0, 'finish_reason': None}]})}\n\n"
        
        messages = body["messages"]

        web_search_messages = [{"role": "system", "content": f"""You are a tool that is used to find relevant google search terms to help with answering questions asked by users.

    Current date: {datetime.datetime.now().strftime('%Y-%m-%d')}
    Current time: {datetime.datetime.now().strftime('%H:%M:%S')}

    Respond with a json dictionary with the following keys:
    - "search_term": a specific, focused search term that would yield the most relevant results. The search term should be concise but descriptive enough to find accurate information.
    - "reasoning": a short explanation of why a web search is or is not needed.
    - "search_term_needed": a boolean value indicating whether a web search is needed.
    """}] + messages
        yield format_message('APEX SYSTEM: Determining necessary tools...')
        try:
            search_query = await make_gpt_query(messages=web_search_messages, model="gpt-4o-mini")
            search_query = json.loads(search_query)
        except Exception as e:
            logger.exception(f"Error in determining necessary tools: {e}")
            yield format_message(f"\nAPEX SYSTEM: Error in determining necessary tools - assuming no tools are needed")
            search_query = {"search_term_needed": False}
        if not search_query["search_term_needed"]:
            logger.debug("No search needed")
        else:
            # data = "'choices': [{'delta': {'content': 'Searching web for" + search_query["search_term"] +"}'}, 'index': 0, 'finish_reason': None}]})}"
            # yield f"data: {data}"
            yield format_message(f"\nAPEX SYSTEM: Searching web for \"{search_query['search_term']}\"")
            try:
                web_results = await web_retrieval(search_query["search_term"], n_miners=N_MINERS, n_results=5, max_response_time=10, target_uids=target_uids)
            except Exception as e:
                logger.exception(f"Error in web retrieval from miner, backing up...: {e}")
                try:
                    web_results = [await get_websites_with_similarity(search_query["search_term"], n_miners=N_MINERS, n_results=5, max_response_time=10, target_uids=target_uids)]
                except Exception as e:
                    logger.exception(f"Error in web retrieval from backup miner: {e}")
                    yield format_message(f"\nAPEX SYSTEM: Error in web retrieval, answering without web information: {e}")
                    web_results = []
            web_results = [r for result in web_results for r in result]
            seen_urls = set()
            web_results = [r for r in web_results if r["url"] not in seen_urls and not seen_urls.add(r["url"])]
            web_results_str = "\n\n"+"-"*20+"\n\n".join([f"website {i+1}: {r['url']}\n\n{r['content']}" for i, r in enumerate(web_results)])
            yield format_message(f"\nAPEX SYSTEM: Web search found {len(web_results)} results")
            messages.append({"role": "assistant", "content": f"Here are web search results that may help you with answering the user's previous question. Use them as you see fit, but make sure to cite the source: {web_results_str}"})

            
        logger.debug(f"BODY: {body}")

        yield format_message(f"\nAPEX SYSTEM: Beginning reasoning...\n")

        response = ""
        for i in range(3):
            if response.lower().strip():
                yield format_message(f"\nAPEX SYSTEM: More research needed...")
            body["messages"] = messages
            response_chunks = []
            try:
                miner_response = await make_openai_query(body=body, uid=target_uids[0], wallet=shared_settings.WALLET, metagraph=shared_settings.METAGRAPH, timeout_seconds=30, stream=True)
            except Exception as e:
                logger.exception(f"Error in making openai query: {e}")
                miner_response = await query_deepseek(messages=messages, stream=True, max_tokens=50, temperature=0.7)
            async for chunk in miner_response:
                print(str(chunk))
                if "<think>" in str(chunk):
                    logger.debug(f"THINK: {chunk}")
                    continue
                if "[DONE]" in str(chunk) or not chunk.choices:
                    logger.debug(f"DONE: {chunk}")
                    break
                response_chunks.append(chunk.choices[0].delta.content)
                yield format_message(chunk.choices[0].delta.content)
            response = "".join(response_chunks)
            logger.debug(f"RESPONSE {i}: {response}")
            messages += [{"role": "assistant", "content": response}]
            if response.lower().strip():
                yield format_message(f"\nAPEX SYSTEM: Reasoning complete, determining need for further research...")
            await asyncio.sleep(3)
        yield format_message(f"\nAPEX SYSTEM: No further research needed, determining final answer...")
        final_answer = await make_gpt_query(messages=messages + [{"role": "assistant", "content": "Now, I will synthesize the information I have gathered online and in my reasoning process in one final answer. The answer should be concise and to the point, but contain all relevant information needed to get to the conclusion. If any online sources were used, the must be referenced properly using markdown links. Keep the final answer formatted in proper markdown. Ensure the final answer directly answers the user's question."}], model="gpt-4o")
        yield format_message(f"\nAPEX SYSTEM: Final answer:\n\n{final_answer}")


            

        
            

        yield "data: [DONE]\n\n"
        return

async def test_time_inference(body, model: str = None, target_uids: list[str] = None):
    return StreamingResponse(
            conduct_deep_research(body, model, target_uids),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    
