import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import "@fontsource-variable/bricolage-grotesque";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import About from "./pages/About";
import Alerts from "./pages/Alerts";
import Dashboard from "./pages/Dashboard";
import MeterDetail from "./pages/MeterDetail";
import Meters from "./pages/Meters";
import ModelPerformance from "./pages/ModelPerformance";
import { BrandMark, Icon } from "./components/ui";
import { CommandPalette, Splash, ThemeToggle } from "./components/visuals";
import { fmtScore } from "./hooks";
import { ModelProvider, useModel } from "./model";
import { NAV } from "./nav";
import { useTheme } from "./theme";
import "./styles.css";

function usePageTitle() {
  const { pathname } = useLocation();
  useEffect(() => {
    const hit = NAV.find(([to]) => (to === "/" ? pathname === "/" : pathname.startsWith(to)));
    document.title = `${hit ? hit[1] : "Not found"} · Theft Risk Monitor`;
  }, [pathname]);
}

// Soft light that follows the pointer over cards (sets --mx / --my, used by CSS).
function useSpotlight() {
  useEffect(() => {
    const onMove = (e) => {
      const el = e.target.closest?.(".card, .stat-strip, .hero");
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${e.clientX - r.left}px`);
      el.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    document.addEventListener("pointermove", onMove, { passive: true });
    return () => document.removeEventListener("pointermove", onMove);
  }, []);
}

function ApiStatus() {
  const { online, model, threshold } = useModel();
  const state = online === null ? "checking" : online ? (model ? "ok" : "warn") : "down";
  const text = { checking: "Connecting…", ok: "API connected", warn: "API up, model not loaded", down: "API unreachable" }[state];
  return (
    <div className="api-status" role="status">
      <div className="api-line"><span className={`dot dot-${state}`} aria-hidden="true" />{text}</div>
      {state === "ok" && <div className="api-meta">{model}<br />Threshold {fmtScore(threshold)}</div>}
      <div className="api-meta">Simulated data only</div>
    </div>
  );
}

// Navigation with a highlight pill that slides to the active page.
function Nav() {
  const ref = useRef(null);
  const { pathname } = useLocation();
  const [pill, setPill] = useState(null);
  useLayoutEffect(() => {
    const measure = () => {
      const a = ref.current?.querySelector("a.active");
      setPill(a ? { x: a.offsetLeft, y: a.offsetTop, w: a.offsetWidth, h: a.offsetHeight } : null);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [pathname]);
  return (
    <nav ref={ref} aria-label="Main">
      <span className="nav-pill" aria-hidden="true"
        style={pill ? { transform: `translate(${pill.x}px, ${pill.y}px)`, width: pill.w, height: pill.h, opacity: 1 } : { opacity: 0 }} />
      {NAV.map(([to, label, icon]) => (
        <NavLink key={to} to={to} end={to === "/"}><Icon name={icon} />{label}</NavLink>
      ))}
    </nav>
  );
}

function Shell() {
  usePageTitle();
  useSpotlight();
  const { pathname } = useLocation();
  const { mode, cycle } = useTheme();
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const onKey = (e) => {
      const typing = /input|textarea|select/i.test(e.target.tagName);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPaletteOpen((o) => !o); }
      else if (e.key === "/" && !typing) { e.preventDefault(); setPaletteOpen(true); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <Splash />
      <div className="aurora" aria-hidden="true"><i /><i /></div>
      <div className="shell">
        <aside className="sidebar">
          <Link to="/" className="brand">
            <BrandMark />
            <span>
              <span className="brand-name">Theft Risk Monitor</span>
              <span className="brand-sub">Inspection prioritisation</span>
            </span>
          </Link>
          <button className="side-search" onClick={() => setPaletteOpen(true)}>
            <Icon name="search" /><span>Quick search</span><kbd>Ctrl K</kbd>
          </button>
          <Nav />
          <div className="side-bottom">
            <ThemeToggle mode={mode} onCycle={cycle} />
            <ApiStatus />
          </div>
        </aside>
        <main>
          <div className="page" key={pathname}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/meters" element={<Meters />} />
              <Route path="/meters/:meterId" element={<MeterDetail />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/model" element={<ModelPerformance />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={
                <div className="empty"><strong>Page not found</strong><p><Link to="/">Back to the overview</Link></p></div>} />
            </Routes>
          </div>
        </main>
      </div>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <ModelProvider>
        <Shell />
      </ModelProvider>
    </BrowserRouter>
  </React.StrictMode>
);
