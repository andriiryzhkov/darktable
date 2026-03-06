import { useEffect, useMemo, useState, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import { ChevronRight, ChevronDown } from "lucide-react";
import CollapsibleModule from "../CollapsibleModule";
import BauhausCombo, { type ComboGroup } from "../../controls/BauhausCombo";
import BauhausInput from "../../controls/BauhausInput";
import BauhausButton from "../../controls/BauhausButton";
import { useCollectionsStore } from "../../../stores/collectionsStore";
import type {
  CollectionMode,
  CollectionProperty,
  PropertyValue,
} from "../../../types/collections";
import {
  PROPERTY_LABELS,
  PROPERTY_CATEGORIES,
} from "../../../types/collections";

// SVG icons for the action button on non-last rules (matches GTK darktable)
const MODE_ICONS: Record<CollectionMode, JSX.Element> = {
  // Two interlocking rings — AND / narrow down
  and: (
    <svg viewBox="0 0 14 10" width="14" height="10" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="5" cy="5" r="3.5" />
      <circle cx="9" cy="5" r="3.5" />
    </svg>
  ),
  // Two separate circles — OR / add more
  or: (
    <svg viewBox="0 0 14 10" width="14" height="10" fill="currentColor">
      <circle cx="4" cy="5" r="3" />
      <circle cx="10" cy="5" r="3" />
    </svg>
  ),
  // Diagonal slash — EXCEPT / exclude
  and_not: (
    <svg viewBox="0 0 10 10" width="10" height="10" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <line x1="2" y1="8" x2="8" y2="2" />
    </svg>
  ),
};

// label→property reverse lookup
const LABEL_TO_PROPERTY: Record<string, CollectionProperty> = {};
for (const [prop, label] of Object.entries(PROPERTY_LABELS)) {
  LABEL_TO_PROPERTY[label] = prop as CollectionProperty;
}

// ── Film roll: show only last folder name ──
function lastPathSegment(path: string): string {
  const parts = path.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || path;
}

// ── Folder tree helpers ──
interface FolderNode {
  name: string;
  fullPath: string;
  count: number;
  children: FolderNode[];
}

function buildFolderTree(values: PropertyValue[]): FolderNode[] {
  const root: FolderNode = { name: "", fullPath: "", count: 0, children: [] };

  for (const v of values) {
    const parts = v.label.split("/").filter(Boolean);
    let current = root;
    let path = "";
    for (const part of parts) {
      path += "/" + part;
      let child = current.children.find((c) => c.name === part);
      if (!child) {
        child = { name: part, fullPath: path, count: 0, children: [] };
        current.children.push(child);
      }
      current = child;
    }
    current.count = v.count;
  }

  return root.children;
}

interface FlatTreeItem {
  name: string;
  fullPath: string;
  count: number;
  depth: number;
  hasChildren: boolean;
}

function flattenTree(
  nodes: FolderNode[],
  depth: number,
  expanded: Set<string>,
): FlatTreeItem[] {
  const result: FlatTreeItem[] = [];
  for (const node of nodes) {
    result.push({
      name: node.name,
      fullPath: node.fullPath,
      count: node.count,
      depth,
      hasChildren: node.children.length > 0,
    });
    if (node.children.length > 0 && expanded.has(node.fullPath)) {
      result.push(...flattenTree(node.children, depth + 1, expanded));
    }
  }
  return result;
}

export default function CollectionsModule() {
  const rules = useCollectionsStore((s) => s.rules);
  const valuesMap = useCollectionsStore((s) => s.valuesMap);
  const valuesLoading = useCollectionsStore((s) => s.valuesLoading);
  const history = useCollectionsStore((s) => s.history);
  const {
    setRuleProperty,
    setRuleMode,
    setRuleText,
    highlightValue,
    selectValue,
    addRule,
    removeRule,
    clearRule,
    clearRules,
    fetchValues,
    restoreHistory,
  } = useCollectionsStore();

  // Track which rule's values list is visible
  const [activeRuleId, setActiveRuleId] = useState<string | null>(null);

  // Action popup for last rule
  const [actionPopup, setActionPopup] = useState<{
    ruleId: string;
    x: number;
    y: number;
  } | null>(null);
  const openActionPopup = useCallback(
    (ruleId: string, btnEl: HTMLElement) => {
      const rect = btnEl.getBoundingClientRect();
      setActionPopup((prev) =>
        prev?.ruleId === ruleId
          ? null
          : { ruleId, x: rect.left, y: rect.bottom + 2 },
      );
    },
    [],
  );

  // Folder tree expanded state
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    () => new Set(),
  );
  const toggleFolder = useCallback((path: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  // History popup
  const [historyOpen, setHistoryOpen] = useState(false);
  const historyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!historyOpen) return;
    const onDown = (e: MouseEvent) => {
      if (historyRef.current && !historyRef.current.contains(e.target as Node))
        setHistoryOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [historyOpen]);

  const propertyGroups: ComboGroup[] = useMemo(
    () =>
      PROPERTY_CATEGORIES.map((cat) => ({
        label: cat.label,
        options: cat.properties.map((p) => PROPERTY_LABELS[p]),
      })),
    [],
  );

  // Fetch values for all rules on mount
  useEffect(() => {
    rules.forEach((r) => fetchValues(r.id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Active rule defaults to last rule
  const effectiveActiveId = activeRuleId ?? rules[rules.length - 1]?.id;
  const activeRule = rules.find((r) => r.id === effectiveActiveId);
  const activeValues = activeRule ? (valuesMap[activeRule.id] ?? []) : [];
  const activeLoading = activeRule
    ? (valuesLoading[activeRule.id] ?? false)
    : false;

  // Pre-compute popup context (avoids IIFE in JSX)
  const popupRuleIndex = actionPopup
    ? rules.findIndex((r) => r.id === actionPopup.ruleId)
    : -1;
  const isPopupLast = popupRuleIndex === rules.length - 1;

  const isFolder = activeRule?.property === "folder";
  const isFilmRoll = activeRule?.property === "film_roll";

  const folderTree = useMemo(
    () => (isFolder ? buildFolderTree(activeValues) : []),
    [isFolder, activeValues],
  );
  const flatItems = useMemo(
    () => (isFolder ? flattenTree(folderTree, 0, expandedFolders) : []),
    [isFolder, folderTree, expandedFolders],
  );

  return (
    <>
    <CollapsibleModule title="collections" description="define search criteria for images to be displayed or edited" defaultOpen onReset={() => { clearRules(); setActiveRuleId(null); }}>
      <div className="collection-rules">
        {/* Rule rows */}
        {rules.map((rule, index) => (
          <div key={rule.id} className="collection-rule-row">
            <div
              className="collection-rule-fields"
              data-active={rule.id === effectiveActiveId}
              onClick={() => setActiveRuleId(rule.id)}
            >
              {/* Property combo — no label */}
              <div className="collection-rule-property">
                <BauhausCombo
                  label=""
                  groups={propertyGroups}
                  value={PROPERTY_LABELS[rule.property]}
                  onChange={(val) => {
                    const prop = LABEL_TO_PROPERTY[val];
                    if (prop) setRuleProperty(rule.id, prop);
                  }}
                />
              </div>
              {/* Text input */}
              <BauhausInput
                label=""
                value={rule.text}
                onChange={(val) => setRuleText(rule.id, val)}
              />
              {/* Action button: mode label on middle rules, arrow-down on first/last */}
              <button
                className="collection-rule-action collection-rule-action-select"
                onClick={(e) => {
                  e.stopPropagation();
                  openActionPopup(rule.id, e.currentTarget);
                }}
                title="actions"
              >
                {index < rules.length - 1 ? (
                  <span key={rules[index + 1].mode}>{MODE_ICONS[rules[index + 1].mode]}</span>
                ) : (
                  <ChevronDown size={12} fill="currentColor" />
                )}
              </button>
            </div>
          </div>
        ))}

        {/* Values list for the active rule */}
        {activeRule && (
          <div className="collection-values-list">
            {activeLoading && (
              <div className="collection-values-loading">loading...</div>
            )}
            {!activeLoading && activeValues.length === 0 && (
              <div className="collection-values-loading">no matches</div>
            )}

            {/* Folder tree view */}
            {!activeLoading && isFolder &&
              flatItems.map((item) => (
                <div
                  key={item.fullPath}
                  className="collection-value-item"
                  data-selected={activeRule.selectedValue === item.fullPath}
                  style={{ paddingLeft: `${6 + item.depth * 14}px` }}
                  onClick={() => highlightValue(activeRule.id, item.fullPath)}
                  onDoubleClick={() => selectValue(activeRule.id, item.fullPath)}
                >
                  <span
                    className={`folder-tree-toggle${item.hasChildren ? "" : " folder-tree-spacer"}`}
                    onClick={(e) => {
                      if (item.hasChildren) {
                        e.stopPropagation();
                        toggleFolder(item.fullPath);
                      }
                    }}
                  >
                    {item.hasChildren && (
                      <ChevronRight
                        size={12}
                        style={{
                          transform: expandedFolders.has(item.fullPath)
                            ? "rotate(90deg)"
                            : "none",
                          transition: "transform 0.15s",
                        }}
                      />
                    )}
                  </span>
                  <span className="collection-value-label">{item.name}</span>
                  {item.count > 0 && (
                    <span className="collection-value-count">({item.count})</span>
                  )}
                </div>
              ))}

            {/* Film roll: last folder name only */}
            {!activeLoading && isFilmRoll &&
              activeValues.map((v) => (
                <div
                  key={String(v.id)}
                  className="collection-value-item"
                  data-selected={activeRule.selectedValue === v.label}
                  onClick={() => highlightValue(activeRule.id, v.label)}
                  onDoubleClick={() => selectValue(activeRule.id, v.label)}
                  title={v.label}
                >
                  <span className="collection-value-label">
                    {lastPathSegment(v.label)}
                  </span>
                  <span className="collection-value-count">({v.count})</span>
                </div>
              ))}

            {/* Default flat list for other properties */}
            {!activeLoading && !isFolder && !isFilmRoll &&
              activeValues.map((v) => (
                <div
                  key={String(v.id)}
                  className="collection-value-item"
                  data-selected={activeRule.selectedValue === v.label}
                  onClick={() => highlightValue(activeRule.id, v.label)}
                  onDoubleClick={() => selectValue(activeRule.id, v.label)}
                >
                  <span className="collection-value-label">{v.label}</span>
                  <span className="collection-value-count">({v.count})</span>
                </div>
              ))}
          </div>
        )}

        {/* Bottom bar */}
        <div className="collection-bottom-bar" ref={historyRef}>
          <BauhausButton
            label="history"
            onClick={() => setHistoryOpen(!historyOpen)}
          />
          {historyOpen && (
            <div className="collection-history-popup">
              {history.length === 0 && (
                <div className="collection-history-empty">no history</div>
              )}
              {history.map((entry, i) => (
                <div
                  key={i}
                  className="collection-history-item"
                  onClick={() => {
                    restoreHistory(i);
                    setActiveRuleId(null);
                    setHistoryOpen(false);
                  }}
                  title={entry.label}
                >
                  {entry.label}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </CollapsibleModule>

    {/* Action popup rendered via portal to escape sidebar overflow */}
    {actionPopup &&
      createPortal(
        <>
          <div
            className="collection-action-overlay"
            onClick={() => setActionPopup(null)}
          />
          <div
            className="collection-action-popup"
            style={{ left: actionPopup.x, top: actionPopup.y }}
          >
            <div
              className="collection-action-popup-item"
              onClick={() => {
                const rid = actionPopup.ruleId;
                setActionPopup(null);
                if (popupRuleIndex === 0) clearRule(rid);
                else removeRule(rid);
              }}
            >
              clear this rule
            </div>
            {isPopupLast ? (
              <>
                <div
                  className="collection-action-popup-item"
                  onClick={() => { setActionPopup(null); addRule("and"); }}
                >
                  narrow down search
                </div>
                <div
                  className="collection-action-popup-item"
                  onClick={() => { setActionPopup(null); addRule("or"); }}
                >
                  add more images
                </div>
                <div
                  className="collection-action-popup-item"
                  onClick={() => { setActionPopup(null); addRule("and_not"); }}
                >
                  exclude images
                </div>
              </>
            ) : popupRuleIndex >= 0 && popupRuleIndex < rules.length - 1 ? (
              (() => {
                const nextRuleId = rules[popupRuleIndex + 1].id;
                return <>
                  <div
                    className="collection-action-popup-item"
                    onClick={() => { setActionPopup(null); setRuleMode(nextRuleId, "and"); }}
                  >
                    change to: and
                  </div>
                  <div
                    className="collection-action-popup-item"
                    onClick={() => { setActionPopup(null); setRuleMode(nextRuleId, "or"); }}
                  >
                    change to: or
                  </div>
                  <div
                    className="collection-action-popup-item"
                    onClick={() => { setActionPopup(null); setRuleMode(nextRuleId, "and_not"); }}
                  >
                    change to: except
                  </div>
                </>;
              })()
            ) : null}
          </div>
        </>,
        document.body,
      )}
    </>
  );
}
