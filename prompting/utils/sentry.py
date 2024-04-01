from bittensor import config as Config
import bittensor as bt
import sentry_sdk

from prompting import __version__

def init_sentry(config : Config, tags : dict = {}):
    if config.sentry_dsn is None:
        bt.logging.info(f"Sentry is DISABLED")
        return

    bt.logging.info(f"Sentry is ENABLED. Using dsn={config.sentry_dsn}")
    sentry_sdk.init(
        dsn=config.sentry_dsn,
        release=__version__,
        environment=f"subnet #{config.netuid}",
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0
    )

    for key, value in tags.items():
        sentry_sdk.set_tag(key, value)