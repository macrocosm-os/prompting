module.exports = {
    apps: [{
      name: "bittensor-validator",
      script: "neurons/validator.py",
      args: "neurons/validator.py --netuid 61 --wallet.hotkey test_hotkey --wallet.name test_wallet --subtensor.network test --logging.debug",
      interpreter: ".venv/bin/python",
      env: {
        NODE_ENV: "production",
      },
      autorestart: true,
      watch: false,
    }]
  };
  