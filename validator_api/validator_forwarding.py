from shared.settings import shared_settings

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