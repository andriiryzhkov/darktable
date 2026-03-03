import { useCallback } from "react";
import { usePlatform } from "../hooks/usePlatform";

interface WindowControlsProps {
  onClose: () => void;
  onMinimize?: () => void;
  onMaximize?: () => void;
}

interface InternalProps extends WindowControlsProps {
  stopDrag: (e: React.PointerEvent | React.MouseEvent) => void;
}

/* Inline SVG icons matching real macOS traffic light glyphs */
const MacCloseIcon = () => (
  <svg className="wc-macos-icon" width="8" height="8" viewBox="0 0 8 8">
    <path d="M1.5 1.5L6.5 6.5M6.5 1.5L1.5 6.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
  </svg>
);
const MacMinIcon = () => (
  <svg className="wc-macos-icon" width="8" height="8" viewBox="0 0 8 8">
    <path d="M1.5 4H6.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
  </svg>
);
const MacMaxIcon = () => (
  <svg className="wc-macos-icon" width="8" height="8" viewBox="0 0 8 8">
    <path d="M1.5 4H6.5M4 1.5V6.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
  </svg>
);

function MacOSControls({ onClose, onMinimize, onMaximize, stopDrag }: InternalProps) {
  return (
    <div className="wc-macos" onPointerDown={stopDrag}>
      <button className="wc-macos-btn wc-macos-close" onClick={onClose} aria-label="Close">
        <MacCloseIcon />
      </button>
      <button
        className="wc-macos-btn wc-macos-minimize"
        onClick={onMinimize}
        disabled={!onMinimize}
        aria-label="Minimize"
      >
        <MacMinIcon />
      </button>
      <button
        className="wc-macos-btn wc-macos-maximize"
        onClick={onMaximize}
        disabled={!onMaximize}
        aria-label="Maximize"
      >
        <MacMaxIcon />
      </button>
    </div>
  );
}

function WindowsControls({ onClose, onMinimize, onMaximize, stopDrag }: InternalProps) {
  return (
    <div className="wc-windows" onPointerDown={stopDrag}>
      <button
        className="wc-win-btn wc-win-minimize"
        onClick={onMinimize}
        disabled={!onMinimize}
        aria-label="Minimize"
      >
        {"\u2014"}
      </button>
      <button
        className="wc-win-btn wc-win-maximize"
        onClick={onMaximize}
        disabled={!onMaximize}
        aria-label="Maximize"
      >
        {"\u25A1"}
      </button>
      <button className="wc-win-btn wc-win-close" onClick={onClose} aria-label="Close">
        {"\u2715"}
      </button>
    </div>
  );
}

function LinuxControls({ onClose, onMinimize, onMaximize, stopDrag }: InternalProps) {
  return (
    <div className="wc-linux" onPointerDown={stopDrag}>
      <button
        className="wc-linux-btn wc-linux-minimize"
        onClick={onMinimize}
        disabled={!onMinimize}
        aria-label="Minimize"
      >
        {"\u2212"}
      </button>
      <button
        className="wc-linux-btn wc-linux-maximize"
        onClick={onMaximize}
        disabled={!onMaximize}
        aria-label="Maximize"
      >
        {"\u25A1"}
      </button>
      <button className="wc-linux-btn wc-linux-close" onClick={onClose} aria-label="Close">
        {"\u00D7"}
      </button>
    </div>
  );
}

export default function WindowControls({ onClose, onMinimize, onMaximize }: WindowControlsProps) {
  const os = usePlatform();

  const stopDrag = useCallback((e: React.PointerEvent | React.MouseEvent) => {
    e.stopPropagation();
  }, []);

  if (!os) return null;

  if (os === "macos")
    return <MacOSControls onClose={onClose} onMinimize={onMinimize} onMaximize={onMaximize} stopDrag={stopDrag} />;
  if (os === "windows")
    return <WindowsControls onClose={onClose} onMinimize={onMinimize} onMaximize={onMaximize} stopDrag={stopDrag} />;
  return <LinuxControls onClose={onClose} onMinimize={onMinimize} onMaximize={onMaximize} stopDrag={stopDrag} />;
}
