module.exports = {
    apps: [
      {
        name: "api",
        script: "poetry",
        args: "run python api.py",
        interpreter: "bash", // Ensures poetry command runs correctly
      },
    ],
  };
  