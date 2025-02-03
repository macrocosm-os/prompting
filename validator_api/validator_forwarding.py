from shared.settings import shared_settings
from pydantic import BaseModel, model_validator
from typing import ClassVar
from shared.misc import cached_property_with_expiration

class Validator(BaseModel):
    uid: int
    stake: float
    axon: str
    hotkey: str
    failures: int = 0

    # Define a constant for the maximum number of allowed failures.
    MAX_FAILURES: ClassVar[int] = 9

    def update_failure(self, status_code: int) -> None:
        """
        Update the validator's failure count based on the success status.

        - If the operation was successful, decrease the failure count (ensuring it doesn't go below 0).
        - If the operation failed, increase the failure count.
        - If the failure count exceeds MAX_FAILURES, mark the validator as inactive.
        """
        if status_code == 200:
            self.failures = max(0, self.failures - 1)
        else:
            self.failures += 1
            if self.failures > self.MAX_FAILURES:
                return 1
        return 0

class ValidatorRegistry(BaseModel):
    """
    Class to store the success of forwards to validator axons.
    If a validator is routinely failing to respond to scoring requests,
    we should stop sending them requests.
    """
    
    # Using a default factory ensures validators is always a dict.
    validators: Dict[int, Validator] = Field(default_factory=dict)
    spot_checking_rate: ClassVar[float] = 0.3

    @model_validator(mode='after')
    def create_validator_list(cls, v: "ValidatorRegistry", metagraph = shared_settings.METAGRAPH) -> "ValidatorRegistry":
        validator_uids = [
            uid for uid in metagraph.validator_permit if metagraph.validator_permit[uid]
        ]
        validator_axons = [
            metagraph.axons[uid].ip_str().split("/") for uid in validator_uids
        ]
        validator_stakes = [metagraph.stake[uid] for uid in validator_uids]
        validator_hotkeys = [metagraph.hotkeys[uid] for uid in validator_uids]
        v.validators = {
            uid: Validator(uid, stake, axon)
            for uid, stake, axon, hotkey in zip(validator_uids, validator_stakes, validator_axons, validator_hotkeys)
        }
        return v

    def get_available_axon(self) -> Optional[Tuple[int, list]]:
        if random.random() > self.spot_checking_rate or not self.validators:
            return None, None
        validator_list = list(self.validators.values())
        weights = [v.stake for v in validator_list]
        chosen = random.choices(validator_list, weights=weights, k=1)[0]
        return chosen.uid, chosen.axon, chosen.hotkey

    def update_validators(self, uid: int, response_code: int) -> None:
        if uid in self.validators:
            max_failures_reached = self.validators[uid].update_failure(success)
            if max_failures_reached:
                del self.validators[uid]
