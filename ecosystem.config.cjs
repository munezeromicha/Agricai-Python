/**
 * PM2 production config. From project root:
 *
 *   pm2 start ecosystem.config.cjs
 *   pm2 logs Agricai-Python
 *   pm2 save
 *
 * Uvicorn module path MUST be app.main:app (FastAPI lives in app/main.py).
 * Do not use main:app — there is no main.py at the repository root.
 */

const fs = require("fs");
const path = require("path");

const root = __dirname;
const isWin = process.platform === "win32";
const venvPython = path.join(
  root,
  ".venv",
  isWin ? path.join("Scripts", "python.exe") : path.join("bin", "python"),
);

if (!fs.existsSync(venvPython)) {
  console.warn(
    `[ecosystem] Missing venv at ${venvPython}. Create it: python -m venv .venv && pip install -r requirements.txt`,
  );
}

module.exports = {
  apps: [
    {
      name: "Agricai-Python",
      cwd: root,
      script: venvPython,
      args: "-m uvicorn app.main:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
