import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// In production, disable browser behaviors that break the app experience
if (import.meta.env.PROD) {
  // Disable right-click context menu
  document.addEventListener("contextmenu", (e) => e.preventDefault());

  // Disable text/element selection
  document.addEventListener("selectstart", (e) => e.preventDefault());
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
