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

interface TabMeta {
  key: TabKey;
  label: string;
  subtitle: string;
  icon: string;
  group: "runtime" | "analytics";
}

const TOKEN_STORAGE_KEY = "vibeos.dashboard.token";
const TAB_STORAGE_KEY = "vibeos.dashboard.tab";
const NAV_COLLAPSED_STORAGE_KEY = "vibeos.dashboard.navCollapsed";

const TAB_META: ReadonlyArray<TabMeta> = [
  {
    key: "overview",
    label: "Overview",
    subtitle: "Gateway status, queue pressure, and runtime health.",
    icon: "◈",
    group: "runtime",
  },
  {
    key: "queue",
    label: "Queue",
    subtitle: "Inbound queue depth and processing pressure.",
    icon: "◷",
    group: "runtime",
  },
  {
    key: "sessions",
    label: "Sessions",
    subtitle: "Active in-memory session clients and idle age.",
    icon: "◉",
    group: "runtime",
  },
  {
    key: "latency",
    label: "Latency",
    subtitle: "Per-conversation percentile response times.",
    icon: "△",
    group: "analytics",
  },
];

const TAB_GROUPS: ReadonlyArray<{ label: string; tabs: TabKey[] }> = [
  { label: "runtime", tabs: ["overview", "queue", "sessions"] },
  { label: "analytics", tabs: ["latency"] },
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
  for (const tab of TAB_META) {
    const suffix = `/${tab.key}`;
    if (tab.key !== "overview" && current.endsWith(suffix)) {
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
    return "n/a";
  }
  return new Date(unixSeconds * 1000).toLocaleTimeString();
}

function statusLabel(state: ConnectionState): string {
  if (state === "live") {
    return "ok";
  }
  if (state === "connecting") {
    return "connecting";
  }
  if (state === "error") {
    return "error";
  }
  return "idle";
}

function tabMetaForKey(tabKey: TabKey): TabMeta {
  return TAB_META.find((tab) => tab.key === tabKey) ?? TAB_META[0];
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
  const [navCollapsed, setNavCollapsed] = createSignal(localStorage.getItem(NAV_COLLAPSED_STORAGE_KEY) === "1");
  const [overview, setOverview] = createSignal<OverviewPayload | null>(null);
  const [connectionState, setConnectionState] = createSignal<ConnectionState>("idle");
  const [errorMessage, setErrorMessage] = createSignal<string | null>(null);

  let websocket: WebSocket | undefined;
  let reconnectTimer: number | undefined;
  let pollTimer: number | undefined;

  const currentTabMeta = createMemo(() => tabMetaForKey(selectedTab()));

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

  const toggleNavCollapsed = () => {
    const next = !navCollapsed();
    setNavCollapsed(next);
    localStorage.setItem(NAV_COLLAPSED_STORAGE_KEY, next ? "1" : "0");
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
          throw new Error("Unauthorized, check the dashboard token.");
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
        setErrorMessage("Received malformed websocket event.");
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
    <div class={`shell ${navCollapsed() ? "shell--nav-collapsed" : ""}`}>
      <header class="topbar">
        <div class="topbar-left">
          <button class="nav-collapse-toggle" onClick={toggleNavCollapsed} title="Toggle navigation" aria-label="Toggle navigation">
            <span class="nav-collapse-toggle__icon">☰</span>
          </button>
          <div class="brand">
            <div class="brand-logo">V</div>
            <div class="brand-text">
              <div class="brand-title">VIBEOS</div>
              <div class="brand-sub">Gateway Dashboard</div>
            </div>
          </div>
        </div>
        <div class="topbar-status">
          <div class="pill">
            <span class={`statusDot ${connectionState() === "live" ? "ok" : ""}`} />
            <span>Health</span>
            <span class="mono">{statusLabel(connectionState())}</span>
          </div>
        </div>
      </header>

      <aside class={`nav ${navCollapsed() ? "nav--collapsed" : ""}`}>
        <For each={TAB_GROUPS}>
          {(group) => (
            <div class="nav-group">
              <div class="nav-label nav-label--static">
                <span class="nav-label__text">{group.label}</span>
              </div>
              <div class="nav-group__items">
                <For each={group.tabs}>
                  {(tabKey) => {
                    const meta = tabMetaForKey(tabKey);
                    return (
                      <button
                        class={`nav-item ${selectedTab() === tabKey ? "active" : ""}`}
                        onClick={() => updateTab(tabKey)}
                      >
                        <span class="nav-item__icon" aria-hidden="true">{meta.icon}</span>
                        <span class="nav-item__text">{meta.label}</span>
                      </button>
                    );
                  }}
                </For>
              </div>
            </div>
          )}
        </For>

        <div class="nav-group nav-group--links">
          <div class="nav-label nav-label--static">
            <span class="nav-label__text">resources</span>
          </div>
          <div class="nav-group__items">
            <a class="nav-item" href="https://github.com/garymjr/vibeos" target="_blank" rel="noreferrer">
              <span class="nav-item__icon" aria-hidden="true">↗</span>
              <span class="nav-item__text">Repository</span>
            </a>
          </div>
        </div>
      </aside>

      <main class="content">
        <section class="content-header">
          <div>
            <div class="page-title">{currentTabMeta().label}</div>
            <div class="page-sub">{currentTabMeta().subtitle}</div>
          </div>
          <div class="page-meta">
            <Show when={errorMessage()}>
              <div class="pill danger">{errorMessage()}</div>
            </Show>
            <div class="pill">
              <span class="label">Updated</span>
              <span class="mono">{formatTimestamp(overview()?.generated_at_unix ?? 0)}</span>
            </div>
          </div>
        </section>

        <Show
          when={overview()}
          fallback={
            <section class="card">
              <div class="card-title">Connect to Dashboard</div>
              <div class="card-sub">Enter a valid token to load runtime data.</div>
              <div class="row" style="margin-top: 14px;">
                <input
                  class="field"
                  type="password"
                  placeholder="Dashboard access token"
                  value={draftToken()}
                  onInput={(event) => setDraftToken(event.currentTarget.value)}
                />
              </div>
              <div class="row" style="margin-top: 10px;">
                <button class="btn primary" onClick={() => void connect()}>Connect</button>
                <button class="btn" onClick={disconnect}>Clear</button>
              </div>
            </section>
          }
        >
          {(payload) => (
            <>
              <Show when={selectedTab() === "overview"}>
                <section class="grid grid-cols-2">
                  <article class="card">
                    <div class="card-title">Connection</div>
                    <div class="card-sub">Use the shared dashboard token from `bot.config.toml`.</div>
                    <div class="row" style="margin-top: 14px;">
                      <input
                        class="field"
                        type="password"
                        placeholder="Dashboard access token"
                        value={draftToken()}
                        onInput={(event) => setDraftToken(event.currentTarget.value)}
                      />
                    </div>
                    <div class="row" style="margin-top: 10px;">
                      <button class="btn primary" onClick={() => void connect()}>Connect</button>
                      <button class="btn" onClick={disconnect}>Clear</button>
                    </div>
                  </article>

                  <article class="card">
                    <div class="card-title">System Snapshot</div>
                    <div class="status-list" style="margin-top: 12px;">
                      <div><span class="label">Queue Depth</span><span>{payload().queue.depth}</span></div>
                      <div><span class="label">Oldest Queue Age</span><span>{formatDuration(payload().queue.oldest_age_seconds)}</span></div>
                      <div><span class="label">Active Runs</span><span>{payload().runs.active_runs}</span></div>
                      <div><span class="label">Session Clients</span><span>{payload().sessions.active_session_count}</span></div>
                      <div><span class="label">Timeout Count</span><span>{payload().sessions.timeout_count}</span></div>
                    </div>
                  </article>
                </section>

                <section class="stat-grid" style="margin-top: 18px;">
                  <article class="stat"><div class="stat-label">Workers</div><div class="stat-value">{payload().runs.worker_concurrency}</div></article>
                  <article class="stat"><div class="stat-label">Conversations</div><div class="stat-value">{payload().runs.active_conversations}</div></article>
                  <article class="stat"><div class="stat-label">Live Sessions</div><div class="stat-value">{payload().sessions.active_session_count}</div></article>
                  <article class="stat"><div class="stat-label">Timeouts</div><div class="stat-value">{payload().sessions.timeout_count}</div></article>
                </section>
              </Show>

              <Show when={selectedTab() === "queue"}>
                <section class="card">
                  <div class="card-title">Queue Pressure</div>
                  <div class="card-sub">Depth and oldest queued item age.</div>
                  <div class="queue-meter" style="margin-top: 14px;">
                    <div class="queue-meter__bar" style={{ width: `${Math.min(100, Math.max(payload().queue.depth * 4, 4))}%` }} />
                  </div>
                  <div class="status-list" style="margin-top: 12px;">
                    <div><span class="label">Current depth</span><span>{payload().queue.depth}</span></div>
                    <div><span class="label">Oldest queued</span><span>{formatDuration(payload().queue.oldest_age_seconds)}</span></div>
                    <div><span class="label">Active runs</span><span>{payload().runs.active_runs}</span></div>
                    <div><span class="label">Worker concurrency</span><span>{payload().runs.worker_concurrency}</span></div>
                  </div>
                </section>
              </Show>

              <Show when={selectedTab() === "sessions"}>
                <section class="card">
                  <div class="card-title">Session Clients</div>
                  <div class="card-sub">In-memory `pi` clients keyed by conversation.</div>
                  <Show when={payload().sessions.sessions.length > 0} fallback={<div class="card-sub" style="margin-top: 14px;">No active sessions.</div>}>
                    <div class="table-wrap" style="margin-top: 12px;">
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
                </section>
              </Show>

              <Show when={selectedTab() === "latency"}>
                <section class="card">
                  <div class="card-title">Latency Percentiles</div>
                  <div class="card-sub">Per-conversation p50/p95/p99 in milliseconds.</div>
                  <Show when={latencyRows().length > 0} fallback={<div class="card-sub" style="margin-top: 14px;">No latency samples yet.</div>}>
                    <div class="latency-list" style="margin-top: 12px;">
                      <For each={latencyRows()}>
                        {(row) => (
                          <div class="latency-row">
                            <div>
                              <div class="mono">{row.conversation}</div>
                              <div class="label">n={row.count}</div>
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
                </section>
              </Show>
            </>
          )}
        </Show>
      </main>
    </div>
  );
}
