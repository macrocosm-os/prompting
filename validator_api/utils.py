import httpx
from loguru import logger

from shared.settings import shared_settings
from validator_api.validator_forwarding import ValidatorRegistry

validator_registry = ValidatorRegistry()

# make class w/ getter that yields validator_axon (creates from shared_settings) based on criterea (stake*x*Y)
 
# TODO: Modify this so that all the forwarded responses are sent in a single request. This is both more efficient but
# also means that on the validator side all responses are scored at once, speeding up the scoring process.
async def forward_response(uids: int, body: dict[str, any], chunks: list[str]):
    uids = [int(u) for u in uids]
    chunk_dict = {u: c for u, c in zip(uids, chunks)}
    logger.info(f"Forwarding response from uid {uids} to scoring with body: {body} and chunks: {chunks}")

    if body.get("task") != "InferenceTask" and body.get("task") != "WebRetrievalTask":
        logger.debug(f"Skipping forwarding for non- inference/web retrieval task: {body.get('task')}")
        return

    # call - class w/ getter that yields validator_axon based on criterea (stake*x*Y)
    # validator_axon = class(shared_settings.METAGRAPH)
    try:
        vali_uid, vali_axon, vali_hotkey = validator_registry.get_available_axon()
    except Exception as e:
        logger.warning(e)
        vali_uid, vali_axon = None, None
    if not vali_uid:
        logger.warning("Unable to get an available validator, either through spot-checking restrictions or errors, skipping scoring")
        return

    url =  f"http://{vali_axon}/scoring"
    payload = {"body": body, "chunks": chunk_dict, "uid": uids}
    # Create an AsyncClient that attaches the header hook.
    # The header hook is created by passing the wallet’s hotkey and the axon’s hotkey.
    # Adjust the attribute access as needed depending on how your axon object is defined.
    async with httpx.AsyncClient(
        timeout=timeout,
        event_hooks={
            "request": [
                create_header_hook(shared_settings.WALLET.hotkey, vali_hotkey)
            ]
        },
    ) as client:
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                },
            )
            if response.status_code == 200:
                logger.info(f"Forwarding response completed with status {response.status_code}")
            else:
                logger.exception(
                    f"Forwarding response uid {uids} failed with status {response.status_code} and payload {payload}"
                )
            # Update the validator registry with the response status code.
            validator_registry.update_validators(uid=vali_uid, response_code=response.status_code)
        except Exception as e:
            logger.exception(f"Exception during forwarding response: {e}")



