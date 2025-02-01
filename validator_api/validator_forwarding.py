from shared.settings import shared_settings
from pydantic import BaseModel, model_validator
from shared.misc import cached_property_with_expiration

class ValidatorRegistry(BaseModel):
    """
    This is a class to store the success of forwards to validator axons.
    If a validator is routinely failing to respond to scoring requests we should stop sending them to them. 
    """

    availability_registry: dict | None = None

    

    @cached_property_with_expiration(expiration_seconds=12000)
    def AXON_STAKE_DICT(self, metagraph = shared_settings.METAGRAPH) -> dict:
        validator_uids = [uid for uid in metagraph.validator_permit if metagraph.validator_permit[uid]]
        validator_axons = [metagraph.axons[uid].ip_str().split("/") for uid in validator_uids]
        validator_stakes = [metagraph.stake[uid] for uid in validator_uids]
        axon_stake_dict: dict = {axon: stake for axon, stake in zip(validator_axons, validator_stakes)}


    def get_availabilities(self):



class ValidatorForwarding:
    """
    Class to yield validator axon based on shared settings.
    """
    def __call__(self):
        return self.get_validator_criterion(shared_settings.METAGRAPH)

    def get_validator_axons(self, metagraph):
        # Get all validator axons where validator_permit[uid] is True
        validator_axons = [uid for uid in metagraph.validator_permit if metagraph.validator_permit[uid]]

        # First sort validators by stake
        stakes = {uid: metagraph.stake[uid] for uid in validator_axons}
        sorted_validators = sorted(stakes.items(), key=lambda x: x[1], reverse=True)

        # Get IP addresses
        validator_axons = [metagraph.axons[uid].ip_str().split("/") for uid, _ in sorted_validators]
        
        #add: random.choice by stake

        return validator_axons