import { rm } from "node:fs/promises";
import path from "node:path";

export default async function globalTeardown(): Promise<void> {
  const workspace = process.env.MLFORGE_E2E_WORKSPACE;
  if (!workspace) return;

  const allowedRoot = `${path.resolve(process.cwd(), "..", ".codex_tmp")}${path.sep}`;
  const resolvedWorkspace = path.resolve(workspace);
  if (!resolvedWorkspace.startsWith(allowedRoot)) {
    throw new Error(`Refusing to remove unsafe Playwright workspace: ${resolvedWorkspace}`);
  }
  await rm(resolvedWorkspace, { recursive: true, force: true });
}
