const REFRESH_INTERVAL_MS = 5000;
const MAX_LIVE_LOG_ENTRIES = 50;

const activeSessionsEl = document.getElementById("active-sessions");
const lastUpdatedEl = document.getElementById("last-updated");
const servicesTableBody = document.getElementById("services-table-body");
const worldStatesTableBody = document.getElementById("world-states-table-body");
const connectionStatusEl = document.getElementById("connection-status");
const liveUpdatesEl = document.getElementById("live-updates");

function setConnectionStatus(connected) {
    connectionStatusEl.textContent = connected ? "connected" : "disconnected";
    connectionStatusEl.className = `status-badge ${connected ? "status-connected" : "status-disconnected"}`;
}

function renderServices(servicesStatus) {
    const rows = Object.values(servicesStatus || {}).map((service) => {
        const statusClass = service.status === "healthy" ? "badge-healthy" : "badge-unhealthy";
        return `<tr>
            <td>${service.name}</td>
            <td class="${statusClass}">${service.status}</td>
            <td>${service.url}</td>
        </tr>`;
    });

    servicesTableBody.innerHTML = rows.length
        ? rows.join("")
        : '<tr><td colspan="3">No services reported</td></tr>';
}

function renderWorldStates(worldStates) {
    const rows = (worldStates || []).map((state) => {
        const ws = state.world_state || state;
        return `<tr>
            <td>${state.session_id || "-"}</td>
            <td>${ws.biome_type || "-"}</td>
            <td>${ws.lighting_mood || "-"}</td>
            <td>${ws.weather_intensity ?? "-"}</td>
            <td>${state.timestamp || "-"}</td>
        </tr>`;
    });

    worldStatesTableBody.innerHTML = rows.length
        ? rows.join("")
        : '<tr><td colspan="5">No recent world states</td></tr>';
}

function logLiveUpdate(message) {
    const item = document.createElement("li");
    const timestamp = new Date().toLocaleTimeString();
    item.textContent = `[${timestamp}] ${message}`;
    liveUpdatesEl.prepend(item);

    while (liveUpdatesEl.children.length > MAX_LIVE_LOG_ENTRIES) {
        liveUpdatesEl.removeChild(liveUpdatesEl.lastChild);
    }
}

async function refreshDashboard() {
    try {
        const response = await fetch("/api/dashboard-data");
        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        const data = await response.json();
        activeSessionsEl.textContent = data.active_sessions ?? 0;
        lastUpdatedEl.textContent = new Date(data.timestamp).toLocaleString();
        renderServices(data.services_status);
        renderWorldStates(data.recent_world_states);
    } catch (error) {
        logLiveUpdate(`Failed to refresh dashboard: ${error.message}`);
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/dashboard`);

    socket.addEventListener("open", () => {
        setConnectionStatus(true);
        logLiveUpdate("WebSocket connected");
        socket.send(JSON.stringify({ type: "ping" }));
    });

    socket.addEventListener("message", (event) => {
        try {
            const message = JSON.parse(event.data);
            logLiveUpdate(`${message.type}: ${JSON.stringify(message.data ?? {})}`);

            if (message.type === "dashboard_update") {
                refreshDashboard();
            }
        } catch (error) {
            logLiveUpdate(`Received unparseable message: ${event.data}`);
        }
    });

    socket.addEventListener("close", () => {
        setConnectionStatus(false);
        logLiveUpdate("WebSocket disconnected, retrying in 5s");
        setTimeout(connectWebSocket, 5000);
    });

    socket.addEventListener("error", () => {
        socket.close();
    });
}

refreshDashboard();
setInterval(refreshDashboard, REFRESH_INTERVAL_MS);
connectWebSocket();
