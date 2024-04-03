import time
import argparse
import bittensor as bt
import prompting
import asyncio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uids", nargs="+", type=int, default=[])
    parser.add_argument("--coldkey", type=str, default="")
    parser.add_argument("--n_times", type=int, default=1)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--network", type=str, default="finney")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--wallet", type=str, default="sn1")
    parser.add_argument("--hotkey", type=str, default="v1")
    parser.add_argument("--prompt", type=str, default="Who are you?")
    parser.add_argument("--timeout", type=int, default=12)

    return parser.parse_args()


async def query(dendrite, synapse, axons, timeout):
    return await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
    )


def ping(args):
    metagraph = bt.metagraph(network=args.network, netuid=args.netuid)
    bt.logging.info(f"MetaGraph: {metagraph}")

    wallet = bt.wallet(name=args.wallet, hotkey=args.hotkey)
    bt.logging.info(f"Wallet: {wallet}")

    dendrite = bt.dendrite(wallet)

    prompts = [args.prompt]
    synapse = prompting.protocol.PromptingSynapse(roles=["user"], messages=prompts)

    if args.uids:
        axons = [metagraph.axons[uid] for uid in args.uids]
    elif args.coldkey:
        axons = [axon for axon in metagraph.axons if axon.coldkey == args.coldkey]
    else:
        axons = metagraph.axons

    bt.logging.info(f"Querying axons: \n{axons}")

    bt.logging.info(f"Querying {len(axons)} axons with prompt... {prompts}")
    responses = asyncio.run(
        query(
            dendrite=dendrite,
            synapse=synapse,
            axons=axons,
            timeout=args.timeout,
        )
    )

    for uid, response in zip(args.uids, responses):
        bt.logging.success(f"uid: {uid}, response: {response}")


if __name__ == "__main__":
    args = parse_args()
    bt.logging.info(args)

    for i in range(args.n_times):
        try:
            ping(args)
        except:
            bt.logging.error("Ping failed.")

        time.sleep(args.delay)
