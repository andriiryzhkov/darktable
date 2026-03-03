import { create } from "zustand";
import { catalogGetCollectionValues } from "../api/commands";
import { emit } from "../events/eventBus";
import type {
  CollectionMode,
  CollectionProperty,
  CollectionRule,
  CollectionRuleParam,
  PropertyValue,
} from "../types/collections";
import { PROPERTY_LABELS, MODE_LABELS } from "../types/collections";

// Snapshot of rules for history (no runtime ids)
export interface HistoryEntry {
  label: string; // human-readable summary
  rules: { mode: CollectionMode; property: CollectionProperty; text: string }[];
}

const MAX_HISTORY = 10;

function serializeRules(
  rules: CollectionRule[],
): HistoryEntry["rules"] {
  return rules.map((r) => ({ mode: r.mode, property: r.property, text: r.text }));
}

function rulesEqual(
  a: HistoryEntry["rules"],
  b: HistoryEntry["rules"],
): boolean {
  if (a.length !== b.length) return false;
  return a.every(
    (r, i) =>
      r.mode === b[i].mode &&
      r.property === b[i].property &&
      r.text === b[i].text,
  );
}

function prettyPrintRules(rules: HistoryEntry["rules"]): string {
  return rules
    .map((r, i) => {
      const prop = PROPERTY_LABELS[r.property];
      const prefix = i > 0 ? ` ${MODE_LABELS[r.mode]} ` : "";
      return `${prefix}${prop}: ${r.text || "*"}`;
    })
    .join("");
}

let nextId = 1;
function makeId(): string {
  return `rule-${nextId++}`;
}

function defaultRule(): CollectionRule {
  return {
    id: makeId(),
    mode: "and",
    property: "film_roll",
    text: "",
    selectedValue: null,
  };
}

interface CollectionsState {
  rules: CollectionRule[];
  valuesMap: Record<string, PropertyValue[]>;
  valuesLoading: Record<string, boolean>;
  history: HistoryEntry[];

  setRuleProperty: (ruleId: string, prop: CollectionProperty) => void;
  setRuleMode: (ruleId: string, mode: CollectionMode) => void;
  setRuleText: (ruleId: string, text: string) => void;
  highlightValue: (ruleId: string, value: string) => void;
  selectValue: (ruleId: string, value: string) => void;
  addRule: (mode?: CollectionMode) => void;
  removeRule: (ruleId: string) => void;
  clearRule: (ruleId: string) => void;
  clearRules: () => void;
  fetchValues: (ruleId: string) => void;
  applyCollection: () => void;
  restoreHistory: (index: number) => void;
  getRulesParams: () => CollectionRuleParam[];
}

// Debounce timers per rule
const debounceTimers: Record<string, ReturnType<typeof setTimeout>> = {};

export const useCollectionsStore = create<CollectionsState>((set, get) => ({
  rules: [defaultRule()],
  valuesMap: {},
  valuesLoading: {},
  history: [],

  setRuleProperty: (ruleId, prop) => {
    set((s) => ({
      rules: s.rules.map((r) =>
        r.id === ruleId ? { ...r, property: prop, text: "", selectedValue: null } : r,
      ),
    }));
    get().fetchValues(ruleId);
  },

  setRuleMode: (ruleId, mode) => {
    set((s) => ({
      rules: s.rules.map((r) => (r.id === ruleId ? { ...r, mode } : r)),
    }));
    get().applyCollection();
  },

  setRuleText: (ruleId, text) => {
    set((s) => ({
      rules: s.rules.map((r) =>
        r.id === ruleId ? { ...r, text, selectedValue: null } : r,
      ),
    }));
    // Debounce the fetch
    if (debounceTimers[ruleId]) clearTimeout(debounceTimers[ruleId]);
    debounceTimers[ruleId] = setTimeout(() => {
      get().fetchValues(ruleId);
    }, 300);
  },

  highlightValue: (ruleId, value) => {
    set((s) => ({
      rules: s.rules.map((r) =>
        r.id === ruleId ? { ...r, selectedValue: value } : r,
      ),
    }));
  },

  selectValue: (ruleId, value) => {
    set((s) => ({
      rules: s.rules.map((r) =>
        r.id === ruleId ? { ...r, text: value, selectedValue: value } : r,
      ),
    }));
    get().applyCollection();
  },

  addRule: (mode) => {
    const { rules } = get();
    if (rules.length >= 10) return;
    const rule = defaultRule();
    if (mode) rule.mode = mode;
    set({ rules: [...rules, rule] });
    get().fetchValues(rule.id);
  },

  removeRule: (ruleId) => {
    const { rules } = get();
    if (rules.length <= 1) return;
    set({ rules: rules.filter((r) => r.id !== ruleId) });
    get().applyCollection();
  },

  clearRule: (ruleId) => {
    set((s) => ({
      rules: s.rules.map((r) =>
        r.id === ruleId ? { ...r, text: "", selectedValue: null } : r,
      ),
    }));
    get().fetchValues(ruleId);
    get().applyCollection();
  },

  clearRules: () => {
    const rule = defaultRule();
    set({
      rules: [rule],
      valuesMap: {},
      valuesLoading: {},
    });
    get().fetchValues(rule.id);
    get().applyCollection();
  },

  fetchValues: (ruleId) => {
    const rule = get().rules.find((r) => r.id === ruleId);
    if (!rule) return;

    set((s) => ({
      valuesLoading: { ...s.valuesLoading, [ruleId]: true },
    }));

    catalogGetCollectionValues(rule.property, rule.text)
      .then((res) => {
        set((s) => ({
          valuesMap: { ...s.valuesMap, [ruleId]: res.values },
          valuesLoading: { ...s.valuesLoading, [ruleId]: false },
        }));
      })
      .catch((err) => {
        console.error("fetchValues failed:", err);
        set((s) => ({
          valuesLoading: { ...s.valuesLoading, [ruleId]: false },
        }));
      });
  },

  applyCollection: () => {
    // Save to history (skip if no rules have text, or identical to last entry)
    const { rules, history } = get();
    const snapshot = serializeRules(rules);
    const hasContent = snapshot.some((r) => r.text.length > 0);
    if (hasContent) {
      const isDuplicate = history.length > 0 && rulesEqual(history[0].rules, snapshot);
      if (!isDuplicate) {
        // Remove any older duplicate, then prepend
        const filtered = history.filter((h) => !rulesEqual(h.rules, snapshot));
        const label = prettyPrintRules(snapshot);
        set({ history: [{ label, rules: snapshot }, ...filtered].slice(0, MAX_HISTORY) });
      }
    }
    emit("collection.changed", { reason: "collection-rules" });
  },

  restoreHistory: (index) => {
    const entry = get().history[index];
    if (!entry) return;
    // Rebuild rules with fresh ids
    const newRules: CollectionRule[] = entry.rules.map((r) => ({
      id: makeId(),
      mode: r.mode,
      property: r.property,
      text: r.text,
      selectedValue: null,
    }));
    set({ rules: newRules, valuesMap: {}, valuesLoading: {} });
    newRules.forEach((r) => get().fetchValues(r.id));
    emit("collection.changed", { reason: "collection-rules" });
  },

  getRulesParams: () => {
    return get()
      .rules.filter((r) => r.text.length > 0)
      .map((r) => ({
        mode: r.mode,
        property: r.property,
        text: r.text,
      }));
  },
}));
