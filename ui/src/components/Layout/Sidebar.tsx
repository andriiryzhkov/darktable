import { type ReactNode } from "react";

interface SidebarProps {
  side: "left" | "right";
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}

export default function Sidebar({
  side,
  open,
  onToggle,
  children,
}: SidebarProps) {
  return (
    <div
      className="shrink-0 flex flex-col overflow-hidden"
      style={{
        width: open ? 220 : 0,
        transition: "width 200ms ease",
        backgroundColor: "var(--plugin-bg-color)",
      }}
    >
      {open && (
        <div
          className="flex-1 overflow-x-hidden"
          style={{
            overflowY: "scroll",
            direction: side === "left" ? "rtl" : "ltr",
          }}
        >
          <div style={{ direction: "ltr" }}>
            {children}
          </div>
        </div>
      )}
      <button
        onClick={onToggle}
        className="absolute z-10"
        style={{
          display: "none",
        }}
      >
        toggle
      </button>
    </div>
  );
}
