module.exports = {
  apps: [
    {
      name: 'api_server',
      script: 'poetry',
      interpreter: 'none',
      args: ['run', 'python', 'validator_api/api.py'],
      min_uptime: '5m',
      max_restarts: 5
    }
  ]
};
