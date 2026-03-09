import { useEffect } from "react";
import DialogOverlay from "./DialogOverlay";
import BauhausButton from "./controls/BauhausButton";

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
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Enter") onCancel();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onCancel]);

  return (
    <DialogOverlay title={title} onClose={onCancel} zIndex={300}>
      <div className="confirm-dialog">
        <div className="confirm-message">{message}</div>
        <div className="confirm-buttons">
          <BauhausButton label="yes" onClick={onConfirm} />
          <BauhausButton label="no" accent onClick={onCancel} />
        </div>
      </div>
    </DialogOverlay>
  );
}
