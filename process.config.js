// process.config.js
module.exports = {
  apps: [
    {
      name: 'nextjs-app',
      script: 'npm',
      args: 'run dev',
      watch: false,
      env: {
        NODE_ENV: 'development',
        PYTHON_BACKEND_URL: 'http://localhost:5002'
      },
    },
    {
      name: 'python-backend',
      script: 'python',
      args: 'python-backend/app.py',
      watch: false,
      env: {
        PORT: 5002
      },
    },
  ],
};