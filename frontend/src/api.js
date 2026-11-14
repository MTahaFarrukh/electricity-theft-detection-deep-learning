// All numbers shown in the dashboard come from these backend calls.
export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  summary: () => request("/dashboard/summary"),
  meters: (status) => request(`/meters${status ? `?status=${status}` : ""}`),
  meter: (id) => request(`/meters/${encodeURIComponent(id)}`),
  alerts: (status) => request(`/alerts${status ? `?status=${status}` : ""}`),
  updateAlert: (id, status) => request(`/alerts/${id}`, { method: "PATCH", body: JSON.stringify({ status }) }),
  modelMetrics: () => request("/model/metrics"),
};
