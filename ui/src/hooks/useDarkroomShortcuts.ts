import { useEffect } from "react";
import { useDevelopStore } from "../stores/developStore";

/**
 * Keyboard shortcuts for the darkroom view.
 * - Ctrl/Cmd+Z: undo (step back in history)
 * - Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y: redo (step forward in history)
 */
export function useDarkroomShortcuts() {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if user is typing in an input field
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.target as HTMLElement)?.isContentEditable) return;

      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;

      const key = e.key.toLowerCase();

      // Undo: Ctrl/Cmd+Z (without Shift)
      if (key === "z" && !e.shiftKey) {
        e.preventDefault();
        const { historyEnd, selectHistory, sessionId } = useDevelopStore.getState();
        if (!sessionId || historyEnd <= 0) return;
        selectHistory(historyEnd - 1);
        return;
      }

      // Redo: Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y
      if ((key === "z" && e.shiftKey) || key === "y") {
        e.preventDefault();
        const { historyEnd, historyItems, selectHistory, sessionId } = useDevelopStore.getState();
        if (!sessionId) return;
        const maxEnd = historyItems.filter((i) => i.num >= 0).length;
        if (historyEnd >= maxEnd) return;
        selectHistory(historyEnd + 1);
        return;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);
}
