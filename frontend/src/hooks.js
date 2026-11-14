import { useCallback, useEffect, useState } from "react";

// Load data from the API; optionally refresh every `intervalMs`.
export function useApi(loader, deps = [], intervalMs = 0) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const load = useCallback(() => {
    loader()
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => setState((s) => ({ ...s, error, loading: false })));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    load();
    if (!intervalMs) return undefined;
    const id = setInterval(load, intervalMs);
    return () => clearInterval(id);
  }, [load, intervalMs]);

  return { ...state, reload: load };
}

export const fmtScore = (v) => (v == null ? "—" : v.toFixed(3));
export const fmtPct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
export const fmtTime = (v) => (v ? new Date(v).toLocaleString() : "—");
