import { useState, useEffect, useCallback, useRef } from "react";
import { configGet, configSet } from "../api/commands";

/**
 * Manages module expanded state, synced with darktable's config system.
 * Config key: plugins/{view}/{op}/expanded  (values: "TRUE" / "FALSE")
 */
export function useModuleExpanded(view: string, op: string, fallback = false) {
  const [open, setOpen] = useState(fallback);
  const userToggled = useRef(false);
  const key = `plugins/${view}/${op}/expanded`;

  useEffect(() => {
    userToggled.current = false;
    configGet(key)
      .then(({ value }) => {
        if (userToggled.current) return;
        if (value === "TRUE") setOpen(true);
        else if (value === "FALSE") setOpen(false);
      })
      .catch(() => {});
  }, [key]);

  const toggle = useCallback(
    (next: boolean) => {
      userToggled.current = true;
      setOpen(next);
      configSet(key, next ? "TRUE" : "FALSE").catch(() => {});
    },
    [key],
  );

  return { open, setOpen: toggle } as const;
}
