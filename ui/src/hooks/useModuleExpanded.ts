import { useState, useEffect, useCallback, useRef } from "react";
import { configGet, configSet } from "../api/commands";

/**
 * Simple event bus so IOP modules can implement accordion behavior:
 * opening one module closes all others in the same group.
 */
const accordionBus = new EventTarget();

function emitAccordionOpen(group: string, op: string) {
  accordionBus.dispatchEvent(new CustomEvent("open", { detail: { group, op } }));
}

/**
 * Manages module expanded state, synced with darktable's config system.
 * Config key: plugins/{view}/{op}/expanded  (values: "TRUE" / "FALSE")
 *
 * When `accordion` is set to a group name, opening this module will close
 * all other modules in the same accordion group (GTK darktable IOP behavior).
 */
export function useModuleExpanded(view: string, op: string, fallback = false, accordion?: string) {
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

  // Listen for other modules opening in the same accordion group
  useEffect(() => {
    if (!accordion) return;
    const handler = (e: Event) => {
      const { group, op: openedOp } = (e as CustomEvent).detail;
      if (group === accordion && openedOp !== op) {
        setOpen(false);
      }
    };
    accordionBus.addEventListener("open", handler);
    return () => accordionBus.removeEventListener("open", handler);
  }, [accordion, op]);

  const toggle = useCallback(
    (next: boolean) => {
      userToggled.current = true;
      setOpen(next);
      configSet(key, next ? "TRUE" : "FALSE").catch(() => {});
      if (next && accordion) {
        emitAccordionOpen(accordion, op);
      }
    },
    [key, accordion, op],
  );

  return { open, setOpen: toggle } as const;
}
