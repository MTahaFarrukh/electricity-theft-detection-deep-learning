import { useEffect, useRef, useState } from "react";

export function useReducedMotion() {
  const query = () => window.matchMedia("(prefers-reduced-motion: reduce)");
  const [reduced, setReduced] = useState(() => query().matches);
  useEffect(() => {
    const mq = query();
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

// False on the first paint, true shortly after: lets CSS transitions play from a "before" state.
export function useMounted(delay = 40) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = setTimeout(() => setMounted(true), delay);
    return () => clearTimeout(id);
  }, [delay]);
  return mounted;
}

// Number that counts up on mount and glides to new values (flashing briefly when a live value changes).
export function CountUp({ value, decimals = 0, duration = 1200, suffix = "" }) {
  const reduced = useReducedMotion();
  const [shown, setShown] = useState(reduced ? value : 0);
  const [flash, setFlash] = useState(false);
  const current = useRef(0);
  const previous = useRef(null);

  useEffect(() => {
    if (value == null || Number.isNaN(value)) return undefined;
    if (previous.current !== null && previous.current !== value) setFlash(true);
    previous.current = value;
    if (reduced) { current.current = value; setShown(value); return undefined; }
    const from = current.current;
    const start = performance.now();
    let raf;
    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 4);
      current.current = from + (value - from) * eased;
      setShown(current.current);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration, reduced]);

  useEffect(() => {
    if (!flash) return undefined;
    const id = setTimeout(() => setFlash(false), 1200);
    return () => clearTimeout(id);
  }, [flash]);

  if (value == null || Number.isNaN(value)) return <span>—</span>;
  const text = shown.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return <span className={`countup ${flash ? "flash" : ""}`}>{text}{suffix}</span>;
}
