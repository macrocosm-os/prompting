
# Prototype for new LLM validation mechanism: Jan 15 release

![next-release-diagram](assets/next-release-diagram.png)

1. Follow the installation steps of this [repo]('../README.md') to create a virtual environment with all the needed dependencies
2. Install the extra dependencies of this folder by running `pip install -r requirements.txt`
3. Run `python main.py` located in this folder



### RUN EXPERIMENTS WITH PM2
- install pm2
- test if code works using 1 sample (look for ######## HOW TO: DEBUGGING EXPERIMENT  ######## on run_experiment.py)
  - start app: `pm2 start app.config.js`
  - list running apps: `pm2 ls `
  - logs of app: `pm2 logs run-experiment`
- change code to 500 samples (look for ######## HOW TO: REAL EXPERIMENT ######## on run_experiment.py)

NOTE: DOUBLE CHECK IF YOU ARE USING THE RIGHT PIPELINE AND THE RIGHT MODEL_NAME