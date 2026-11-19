import { useEffect, useState } from "react";

const KEY = "etd-theme";
const NEXT = { system: "light", light: "dark", dark: "system" };
const darkQuery = () => window.matchMedia("(prefers-color-scheme: dark)");
const resolve = (mode) => (mode === "system" ? (darkQuery().matches ? "dark" : "light") : mode);

// Theme: follow the system by default; the sidebar button cycles system -> light -> dark.
export function useTheme() {
  const [mode, setMode] = useState(() => {
    try { return localStorage.getItem(KEY) || "system"; } catch { return "system"; }
  });
  useEffect(() => {
    const apply = () => { document.documentElement.dataset.theme = resolve(mode); };
    apply();
    try { localStorage.setItem(KEY, mode); } catch { /* storage may be blocked */ }
    const mq = darkQuery();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [mode]);
  return { mode, cycle: () => setMode((m) => NEXT[m]) };
}
