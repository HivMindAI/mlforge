import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

const frontendRoot = process.cwd();
const repositoryRoot = path.resolve(frontendRoot, "..");
const workspace = path.join(repositoryRoot, ".codex_tmp", `playwright-web-${process.pid}`);
const pythonCommand =
  process.env.MLFORGE_E2E_PYTHON ??
  (process.platform === "win32" ? ".\\.venv\\Scripts\\python.exe" : "python");

process.env.MLFORGE_E2E_WORKSPACE = workspace;

export default defineConfig({
  testDir: "./tests/e2e",
  outputDir: "test-results",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? [["github"], ["line"]] : "line",
  globalTeardown: "./tests/e2e/global-teardown.ts",
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: `${pythonCommand} -m mlforge.web`,
      cwd: repositoryRoot,
      env: { MLFORGE_WEB_WORKSPACE: workspace },
      url: "http://127.0.0.1:8000/api/health/ready",
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: "npm run dev -- --hostname 127.0.0.1",
      cwd: frontendRoot,
      env: { MLFORGE_API_ORIGIN: "http://127.0.0.1:8000" },
      url: "http://127.0.0.1:3000",
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
