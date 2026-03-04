import { useEffect, useRef } from "react";
import DialogOverlay from "./DialogOverlay";
import ModuleButton from "./Sidebar/controls/ModuleButton";

interface ConfirmDialogProps {
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  title,
  message,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const yesRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Enter") onConfirm();
    };
    window.addEventListener("keydown", handleKey);
    yesRef.current?.focus();
    return () => window.removeEventListener("keydown", handleKey);
  }, [onConfirm]);

  return (
    <DialogOverlay title={title} onClose={onCancel} zIndex={300}>
      <div className="confirm-dialog">
        <div className="confirm-message">{message}</div>
        <div className="confirm-buttons">
          <ModuleButton label="yes" onClick={onConfirm} />
          <ModuleButton label="no" onClick={onCancel} />
        </div>
      </div>
    </DialogOverlay>
  );
}
