import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import About from "./pages/About";
import Alerts from "./pages/Alerts";
import Dashboard from "./pages/Dashboard";
import MeterDetail from "./pages/MeterDetail";
import Meters from "./pages/Meters";
import ModelPerformance from "./pages/ModelPerformance";
import "./styles.css";

const NAV = [["/", "Dashboard"], ["/meters", "Meters"], ["/alerts", "Alerts"], ["/model", "Model performance"], ["/about", "About"]];

function App() {
  return (
    <BrowserRouter>
      <header className="topbar">
        <div className="brand">Theft Risk Monitor</div>
        <nav>
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>
          ))}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/meters" element={<Meters />} />
          <Route path="/meters/:meterId" element={<MeterDetail />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/model" element={<ModelPerformance />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<React.StrictMode><App /></React.StrictMode>);
