module.exports = {
  apps: [{
    name: "bittensor-validator",
    script: "poetry",
    args: "run python neurons/validator.py --netuid 61 --wallet.hotkey test_hotkey --wallet.name test_wallet --subtensor.network test --logging.debug",
    interpreter: "none",
    env: {
      NODE_ENV: "production",
    },
    autorestart: true,
    watch: false,
  }]
};