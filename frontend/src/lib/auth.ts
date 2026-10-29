/**
 * Browser-side LangSmith API key resolution + persistence.
 *
 * The agent server endpoints (/threads, /runs, /assistants) are gated by
 * the workspace API key on LangSmith Cloud deployments. We resolve a key
 * from three places in priority order:
 *
 *   1. `?api_key=...` URL parameter - promoted into localStorage and
 *      stripped from the visible URL via history.replaceState.
 *   2. localStorage["project_advisor-customer:apiKey"] from a previous session.
 *   3. None - the chat will fail on the deployment but works against
 *      local `langgraph dev` (no-op auth).
 *
 * Local dev (localhost / 127.0.0.1) treats "no key" as expected and the
 * gate doesn't show; deployed hosts show the gate so the user can paste
 * a key inline instead of silently 401-ing.
 */

export const API_KEY_STORAGE = "project_advisor-customer:apiKey";

export function resolveApiKey(): string | null {
  if (typeof window === "undefined") return null;
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("api_key");
  if (fromUrl) {
    saveApiKey(fromUrl);
    url.searchParams.delete("api_key");
    window.history.replaceState(
      {},
      "",
      url.pathname + url.search + url.hash,
    );
    return fromUrl;
  }
  try {
    return window.localStorage.getItem(API_KEY_STORAGE);
  } catch {
    return null;
  }
}

export function saveApiKey(apiKey: string): void {
  try {
    window.localStorage.setItem(API_KEY_STORAGE, apiKey);
  } catch {
    /* private-mode browsers etc. - caller can still pass it in memory */
  }
}

export function isLocalDev(): boolean {
  if (typeof window === "undefined") return false;
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";
}
