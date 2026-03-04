import type { ReactNode } from "react";

export type ModuleGroup = "active" | "favorites" | "basic" | "tone" | "color" | "correct" | "effect";

export interface ModuleGroupDef {
  id: ModuleGroup;
  label: string;
  icon: ReactNode;
}

const SvgIcon = ({ children }: { children: ReactNode }) => (
  <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    {children}
  </svg>
);

export const DARKROOM_MODULE_GROUPS: ModuleGroupDef[] = [
  {
    id: "active",
    label: "active modules",
    icon: <SvgIcon><circle cx="12" cy="12" r="3" /><path d="M12 3v2m0 14v2m-7.07-2.93 1.41-1.41m9.9-9.9 1.41-1.41M3 12h2m14 0h2m-2.93 7.07-1.41-1.41m-9.9-9.9L5.93 4.93" /></SvgIcon>,
  },
  {
    id: "favorites",
    label: "favorites",
    icon: <SvgIcon><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" /></SvgIcon>,
  },
  {
    id: "basic",
    label: "base",
    icon: <SvgIcon><circle cx="12" cy="12" r="10" /></SvgIcon>,
  },
  {
    id: "tone",
    label: "tone",
    icon: <SvgIcon><circle cx="12" cy="12" r="5" /><path d="M12 1v2m0 18v2m8.66-17.66-1.41 1.41M4.75 19.25l-1.41 1.41M23 12h-2M3 12H1m17.66 8.66-1.41-1.41M4.75 4.75 3.34 3.34" /></SvgIcon>,
  },
  {
    id: "color",
    label: "color",
    icon: <SvgIcon><circle cx="13.5" cy="6.5" r="4.5" fill="none" /><circle cx="17.5" cy="15.5" r="4.5" fill="none" /><circle cx="8.5" cy="15.5" r="4.5" fill="none" /></SvgIcon>,
  },
  {
    id: "correct",
    label: "correct",
    icon: <SvgIcon><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" /></SvgIcon>,
  },
  {
    id: "effect",
    label: "effect",
    icon: <SvgIcon><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3z" /></SvgIcon>,
  },
];
