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

/** Cached single_module config value */
let singleModuleCached: boolean | null = null;

/** Load single_module config (cached after first call) */
function getSingleModule(): Promise<boolean> {
  if (singleModuleCached !== null) return Promise.resolve(singleModuleCached);
  return configGet("darkroom/ui/single_module")
    .then(({ value }) => {
      singleModuleCached = value === "TRUE";
      return singleModuleCached;
    })
    .catch(() => {
      singleModuleCached = true; // default: accordion on
      return true;
    });
}

/**
 * Manages module expanded state, synced with darktable's config system.
 * Config key: plugins/{view}/{op}/expanded  (values: "TRUE" / "FALSE")
 *
 * When `accordion` is set to a group name, the single_module config and
 * shift key determine whether opening this module collapses others:
 * - single_module=TRUE: click collapses others, shift+click doesn't
 * - single_module=FALSE: click doesn't collapse, shift+click does
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
        configSet(key, "FALSE").catch(() => {});
      }
    };
    accordionBus.addEventListener("open", handler);
    return () => accordionBus.removeEventListener("open", handler);
  }, [accordion, op, view]);

  const toggle = useCallback(
    (next: boolean, shiftKey = false) => {
      userToggled.current = true;
      setOpen(next);
      configSet(key, next ? "TRUE" : "FALSE").catch(() => {});
      if (next && accordion) {
        // XOR: single_module inverts shift behavior
        getSingleModule().then((singleModule) => {
          const collapseOthers = singleModule !== shiftKey;
          if (collapseOthers) {
            emitAccordionOpen(accordion, op);
          }
        });
      }
    },
    [key, accordion, op],
  );

  return { open, setOpen: toggle } as const;
}
