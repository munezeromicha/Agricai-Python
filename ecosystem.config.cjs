/**
 * PM2 production config. From project root:
 *
 *   ./scripts/pm2-deploy.sh   (preferred — removes duplicates first)
 *   pm2 start ecosystem.config.cjs
 *   pm2 logs Agricai-Python
 *   pm2 save
 *
 * Finds Python in .venv/ or venv/ (or VENV_DIR). Uvicorn target: app.main:app
 */

const fs = require("fs");
const path = require("path");

const root = __dirname;
const isWin = process.platform === "win32";
const pythonRel = isWin
  ? path.join("Scripts", "python.exe")
  : path.join("bin", "python");

function resolveVenvPython() {
  const dirNames = [];
  if (process.env.VENV_DIR) {
    dirNames.push(process.env.VENV_DIR);
  }
  dirNames.push(".venv", "venv");

  for (const dir of dirNames) {
    const candidate = path.join(root, dir, pythonRel);
    if (fs.existsSync(candidate)) {
      return { python: candidate, dir };
    }
  }
  return null;
}

const resolved = resolveVenvPython();
if (!resolved) {
  const hint = isWin ? "python -m venv venv" : "python3 -m venv venv";
  console.error("[ecosystem] No virtualenv Python found. Create one at venv/ or .venv/:");
  console.error(`  ${hint} && source venv/bin/activate && pip install -r requirements.txt`);
  console.error("Or set VENV_DIR to your folder name before pm2 start.");
  process.exit(1);
}

console.log(`[ecosystem] Using ${resolved.dir}/${pythonRel}`);

module.exports = {
  apps: [
    {
      name: "Agricai-Python",
      cwd: root,
      script: resolved.python,
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
