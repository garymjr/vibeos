import { For, Show, createMemo, createSignal, onCleanup, onMount } from "solid-js";

type TabKey = "overview" | "queue" | "sessions" | "latency";

type ConnectionState = "idle" | "connecting" | "live" | "error";

interface QueueSnapshot {
  depth: number;
  oldest_age_seconds: number;
}

interface RunSnapshot {
  active_runs: number;
  worker_concurrency: number;
  active_conversations: number;
}

interface SessionEntry {
  conversation_key: string;
  session_dir: string;
  created_age_seconds: number;
  idle_age_seconds: number;
}

interface SessionSnapshot {
  active_session_count: number;
  timeout_count: number;
  sessions: SessionEntry[];
}

interface LatencyEntry {
  count: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

interface OverviewPayload {
  queue: QueueSnapshot;
  runs: RunSnapshot;
  sessions: SessionSnapshot;
  latency: Record<string, LatencyEntry>;
  generated_at_unix: number;
}

interface WebSocketOverviewEvent {
  type: "overview";
  payload: OverviewPayload;
}

const TOKEN_STORAGE_KEY = "vibeos.dashboard.token";
const TAB_STORAGE_KEY = "vibeos.dashboard.tab";
const TABS: ReadonlyArray<{ key: TabKey; label: string }> = [
  { key: "overview", label: "Overview" },
  { key: "queue", label: "Queue" },
  { key: "sessions", label: "Sessions" },
  { key: "latency", label: "Latency" },
];

function normalizePath(path: string): string {
  if (!path || path === "/") {
    return "/dashboard";
  }
  const trimmed = path.replace(/\/+$/, "");
  return trimmed || "/dashboard";
}

function inferBasePath(pathname: string): string {
  const current = normalizePath(pathname);
  for (const { key } of TABS) {
    const suffix = `/${key}`;
    if (key !== "overview" && current.endsWith(suffix)) {
      return normalizePath(current.slice(0, -suffix.length));
    }
  }
  return current;
}

function parseTab(pathname: string, basePath: string): TabKey {
  const relative = pathname.replace(basePath, "") || "/";
  const candidate = relative.replace(/^\//, "").split("/")[0];
  if (candidate === "queue" || candidate === "sessions" || candidate === "latency") {
    return candidate;
  }
  return "overview";
}

function pathForTab(basePath: string, tab: TabKey): string {
  if (tab === "overview") {
    return basePath;
  }
  return `${basePath}/${tab}`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

function formatTimestamp(unixSeconds: number): string {
  if (!unixSeconds) {
    return "No data yet";
  }
  return new Date(unixSeconds * 1000).toLocaleTimeString();
}

export default function App() {
  const basePath = inferBasePath(window.location.pathname);
  const storedTab = localStorage.getItem(TAB_STORAGE_KEY);
  const initialTab =
    storedTab === "overview" || storedTab === "queue" || storedTab === "sessions" || storedTab === "latency"
      ? storedTab
      : parseTab(window.location.pathname, basePath);

  const [token, setToken] = createSignal(localStorage.getItem(TOKEN_STORAGE_KEY) ?? "");
  const [draftToken, setDraftToken] = createSignal(token());
  const [selectedTab, setSelectedTab] = createSignal<TabKey>(initialTab);
  const [overview, setOverview] = createSignal<OverviewPayload | null>(null);
  const [connectionState, setConnectionState] = createSignal<ConnectionState>("idle");
  const [errorMessage, setErrorMessage] = createSignal<string | null>(null);

  let websocket: WebSocket | undefined;
  let reconnectTimer: number | undefined;
  let pollTimer: number | undefined;

  const latencyRows = createMemo(() => {
    const entries = Object.entries(overview()?.latency ?? {}).map(([conversation, latency]) => ({
      conversation,
      ...latency,
    }));
    entries.sort((left, right) => right.p95_ms - left.p95_ms);
    return entries;
  });

  const updateTab = (tab: TabKey, replace = false) => {
    setSelectedTab(tab);
    localStorage.setItem(TAB_STORAGE_KEY, tab);
    const nextPath = pathForTab(basePath, tab);
    const method = replace ? "replaceState" : "pushState";
    window.history[method]({}, "", nextPath);
  };

  const clearTimers = () => {
    if (reconnectTimer !== undefined) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = undefined;
    }
    if (pollTimer !== undefined) {
      window.clearInterval(pollTimer);
      pollTimer = undefined;
    }
  };

  const closeWebSocket = () => {
    if (websocket !== undefined) {
      websocket.onclose = null;
      websocket.close();
      websocket = undefined;
    }
  };

  const stopLiveUpdates = () => {
    clearTimers();
    closeWebSocket();
    setConnectionState("idle");
  };

  const fetchOverview = async (activeToken: string): Promise<void> => {
    try {
      const response = await fetch(`${basePath}/api/overview`, {
        headers: { Authorization: `Bearer ${activeToken}` },
      });
      if (!response.ok) {
        if (response.status === 401) {
          throw new Error("Unauthorized, check your dashboard token.");
        }
        throw new Error(`Overview request failed with status ${response.status}.`);
      }
      const payload = (await response.json()) as OverviewPayload;
      setOverview(payload);
      setErrorMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to fetch overview.";
      setErrorMessage(message);
      setConnectionState("error");
    }
  };

  const openWebSocket = (activeToken: string) => {
    closeWebSocket();
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const url = `${protocol}://${window.location.host}${basePath}/ws/events?token=${encodeURIComponent(activeToken)}`;
    setConnectionState("connecting");

    websocket = new WebSocket(url);
    websocket.onopen = () => {
      setConnectionState("live");
      setErrorMessage(null);
    };
    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketOverviewEvent;
        if (data.type === "overview") {
          setOverview(data.payload);
        }
      } catch {
        setErrorMessage("Received malformed websocket message.");
      }
    };
    websocket.onerror = () => {
      setConnectionState("error");
      setErrorMessage("Dashboard websocket connection failed.");
    };
    websocket.onclose = () => {
      if (!token()) {
        return;
      }
      setConnectionState("connecting");
      reconnectTimer = window.setTimeout(() => openWebSocket(activeToken), 1500);
    };
  };

  const startLiveUpdates = async (activeToken: string): Promise<void> => {
    stopLiveUpdates();
    await fetchOverview(activeToken);
    openWebSocket(activeToken);
    pollTimer = window.setInterval(() => {
      void fetchOverview(activeToken);
    }, 15000);
  };

  const connect = async () => {
    const nextToken = draftToken().trim();
    if (!nextToken) {
      setErrorMessage("Token is required.");
      return;
    }
    localStorage.setItem(TOKEN_STORAGE_KEY, nextToken);
    setToken(nextToken);
    await startLiveUpdates(nextToken);
  };

  const disconnect = () => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken("");
    setDraftToken("");
    setOverview(null);
    setErrorMessage(null);
    stopLiveUpdates();
  };

  onMount(() => {
    if (parseTab(window.location.pathname, basePath) === "overview" && storedTab) {
      updateTab(initialTab, true);
    }

    const onPopState = () => setSelectedTab(parseTab(window.location.pathname, basePath));
    window.addEventListener("popstate", onPopState);

    if (token()) {
      void startLiveUpdates(token());
    }

    onCleanup(() => {
      window.removeEventListener("popstate", onPopState);
      stopLiveUpdates();
    });
  });

  return (
    <div class="shell">
      <div class="background-grid" />
      <header class="topbar">
        <div>
          <p class="eyebrow">VibeOS</p>
          <h1>Operations Cockpit</h1>
        </div>
        <div class={`status status-${connectionState()}`}>
          <span class="status-dot" />
          <span>{connectionState()}</span>
        </div>
      </header>

      <section class="auth-panel">
        <label for="token-input">Dashboard Token</label>
        <div class="auth-controls">
          <input
            id="token-input"
            type="password"
            placeholder="Paste access token"
            value={draftToken()}
            onInput={(event) => setDraftToken(event.currentTarget.value)}
          />
          <button class="button-primary" onClick={() => void connect()}>
            Connect
          </button>
          <button class="button-ghost" onClick={disconnect}>
            Clear
          </button>
        </div>
        <Show when={errorMessage()}>
          <p class="error-text">{errorMessage()}</p>
        </Show>
      </section>

      <nav class="tabbar">
        <For each={TABS}>
          {(tab) => (
            <button
              class={`tab ${selectedTab() === tab.key ? "tab-active" : ""}`}
              onClick={() => updateTab(tab.key)}
            >
              {tab.label}
            </button>
          )}
        </For>
      </nav>

      <main class="panel">
        <Show when={overview()} fallback={<p class="empty">Connect with a valid token to load runtime data.</p>}>
          {(payload) => (
            <>
              <Show when={selectedTab() === "overview"}>
                <section class="metric-grid">
                  <article class="metric-card">
                    <p class="metric-label">Queue Depth</p>
                    <p class="metric-value">{payload().queue.depth}</p>
                    <p class="metric-hint">Oldest queued: {formatDuration(payload().queue.oldest_age_seconds)}</p>
                  </article>
                  <article class="metric-card">
                    <p class="metric-label">Active Runs</p>
                    <p class="metric-value">{payload().runs.active_runs}</p>
                    <p class="metric-hint">
                      {payload().runs.worker_concurrency} workers, {payload().runs.active_conversations} conversations tracked
                    </p>
                  </article>
                  <article class="metric-card">
                    <p class="metric-label">PI Sessions</p>
                    <p class="metric-value">{payload().sessions.active_session_count}</p>
                    <p class="metric-hint">Timeouts: {payload().sessions.timeout_count}</p>
                  </article>
                  <article class="metric-card">
                    <p class="metric-label">Last Update</p>
                    <p class="metric-value">{formatTimestamp(payload().generated_at_unix)}</p>
                    <p class="metric-hint">Live stream + 15s HTTP fallback</p>
                  </article>
                </section>
              </Show>

              <Show when={selectedTab() === "queue"}>
                <section class="stack">
                  <article class="detail-card">
                    <h2>Queue Pressure</h2>
                    <p>Depth: {payload().queue.depth}</p>
                    <p>Oldest message age: {formatDuration(payload().queue.oldest_age_seconds)}</p>
                    <div class="bar-track">
                      <div
                        class="bar-fill"
                        style={{ width: `${Math.min(100, Math.max(payload().queue.depth * 4, 4))}%` }}
                      />
                    </div>
                  </article>
                </section>
              </Show>

              <Show when={selectedTab() === "sessions"}>
                <section class="stack">
                  <article class="detail-card">
                    <h2>Active Session Clients</h2>
                    <Show
                      when={payload().sessions.sessions.length > 0}
                      fallback={<p class="empty-inline">No active sessions in memory.</p>}
                    >
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Conversation</th>
                              <th>Idle</th>
                              <th>Created</th>
                              <th>Directory</th>
                            </tr>
                          </thead>
                          <tbody>
                            <For each={payload().sessions.sessions}>
                              {(session) => (
                                <tr>
                                  <td class="mono">{session.conversation_key}</td>
                                  <td>{formatDuration(session.idle_age_seconds)}</td>
                                  <td>{formatDuration(session.created_age_seconds)}</td>
                                  <td class="mono">{session.session_dir}</td>
                                </tr>
                              )}
                            </For>
                          </tbody>
                        </table>
                      </div>
                    </Show>
                  </article>
                </section>
              </Show>

              <Show when={selectedTab() === "latency"}>
                <section class="stack">
                  <article class="detail-card">
                    <h2>Latency Percentiles</h2>
                    <Show when={latencyRows().length > 0} fallback={<p class="empty-inline">No latency samples yet.</p>}>
                      <div class="latency-list">
                        <For each={latencyRows()}>
                          {(row) => (
                            <div class="latency-row">
                              <div>
                                <p class="mono">{row.conversation}</p>
                                <p class="muted">n={row.count}</p>
                              </div>
                              <div class="latency-values">
                                <span>P50 {row.p50_ms.toFixed(0)}ms</span>
                                <span>P95 {row.p95_ms.toFixed(0)}ms</span>
                                <span>P99 {row.p99_ms.toFixed(0)}ms</span>
                              </div>
                            </div>
                          )}
                        </For>
                      </div>
                    </Show>
                  </article>
                </section>
              </Show>
            </>
          )}
        </Show>
      </main>
    </div>
  );
}
