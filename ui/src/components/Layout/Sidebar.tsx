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
        borderLeft: side === "right" ? "1px solid var(--border-color)" : "none",
        borderRight: side === "left" ? "1px solid var(--border-color)" : "none",
      }}
    >
      {open && (
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
          {children}
        </div>
      )}
      {/* Toggle button rendered outside the sidebar to remain visible when closed */}
      <button
        onClick={onToggle}
        className="absolute z-10"
        style={{
          display: "none", // Hidden for now; sidebar toggle via keyboard or menu
        }}
      >
        toggle
      </button>
    </div>
  );
}
