import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { ArrowDownWideNarrow, ArrowUpNarrowWide, Filter, Check } from "lucide-react";
import {
  useFilterStore,
  ALL_FILTER_TYPES,
  ALL_COLOR_KEYS,
  FILTER_TYPE_LABELS,
  MODULE_ORDER_OPTIONS,
  SORT_GROUPS,
  type FilterType,
} from "../../stores/filterStore";
import BauhausCombo from "../controls/BauhausCombo";
import BauhausButton from "../controls/BauhausButton";

const STAR_PATH =
  "M12 2l2.9 6.6L22 9.5l-5 4.8 1.2 7.2L12 18l-6.2 3.5L7 14.3l-5-4.8 7.1-.9z";

const COLOR_VARS: Record<string, string> = {
  red: "--colorlabel-red",
  yellow: "--colorlabel-yellow",
  green: "--colorlabel-green",
  blue: "--colorlabel-blue",
  purple: "--colorlabel-purple",
};

const MODULE_ORDER_LABELS = MODULE_ORDER_OPTIONS.map((o) => o.label);
const MODULE_ORDER_BY_LABEL = Object.fromEntries(
  MODULE_ORDER_OPTIONS.map((o) => [o.label, o.value]),
);
const MODULE_ORDER_BY_VALUE = Object.fromEntries(
  MODULE_ORDER_OPTIONS.map((o) => [o.value, o.label]),
);

function ModuleOrderFilter() {
  const moduleOrder = useFilterStore((s) => s.moduleOrder);
  const setModuleOrder = useFilterStore((s) => s.setModuleOrder);

  return (
    <BauhausCombo
      label="module order"
      hideLabel
      options={MODULE_ORDER_LABELS}
      value={MODULE_ORDER_BY_VALUE[moduleOrder]}
      onChange={(label) => setModuleOrder(MODULE_ORDER_BY_LABEL[label])}
    />
  );
}

function ColorCircle({
  cssVar,
  active,
  onClick,
}: {
  cssVar?: string;
  active: boolean;
  onClick: () => void;
}) {
  const stroke = cssVar ? `var(${cssVar})` : "var(--fg-color)";
  return (
    <svg
      className="filter-color-circle"
      width={12}
      height={12}
      viewBox="0 0 12 12"
      onClick={onClick}
    >
      <circle
        cx={6}
        cy={6}
        r={4.5}
        fill={active ? stroke : "none"}
        stroke={stroke}
        strokeWidth={2}
      />
    </svg>
  );
}

function ColorLabelFilter() {
  const activeColors = useFilterStore((s) => s.activeColors);
  const colorMode = useFilterStore((s) => s.colorMode);
  const toggleColor = useFilterStore((s) => s.toggleColor);
  const toggleAllColors = useFilterStore((s) => s.toggleAllColors);
  const toggleColorMode = useFilterStore((s) => s.toggleColorMode);

  const allActive = ALL_COLOR_KEYS.every((k) => activeColors.has(k));

  return (
    <div className="filter-bar-section">
      {ALL_COLOR_KEYS.map((key) => (
        <ColorCircle
          key={key}
          cssVar={COLOR_VARS[key]}
          active={activeColors.has(key)}
          onClick={() => toggleColor(key)}
        />
      ))}
      <ColorCircle
        active={allActive}
        onClick={toggleAllColors}
      />
      <span
        className="filter-color-mode"
        data-disabled={activeColors.size < 2}
        title={colorMode === "union" ? "Union (OR) — click for intersection" : "Intersection (AND) — click for union"}
        onClick={activeColors.size >= 2 ? toggleColorMode : undefined}
      >
        {colorMode === "union" ? (
          <svg viewBox="0 0 14 10" width="14" height="10" fill="currentColor">
            <circle cx="4" cy="5" r="3" />
            <circle cx="10" cy="5" r="3" />
          </svg>
        ) : (
          <svg viewBox="0 0 14 10" width="14" height="10" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="5" cy="5" r="3.5" />
            <circle cx="9" cy="5" r="3.5" />
          </svg>
        )}
      </span>
    </div>
  );
}

// All selectable rating values: -1=rejected, 0=unrated, 1-5=stars
const RANGE_RATING_VALUES = [-1, 0, 1, 2, 3, 4, 5] as const;

const RANGE_RATING_LABELS: Record<number, string> = {
  [-1]: "rejected",
  0: "unrated",
  1: "one star",
  2: "two stars",
  3: "three stars",
  4: "four stars",
  5: "five stars",
};

function RangeRatingFilter() {
  const ratingSelection = useFilterStore((s) => s.ratingSelection);
  const toggleRating = useFilterStore((s) => s.toggleRating);
  const toggleRatingInSet = useFilterStore((s) => s.toggleRatingInSet);
  const setRatingRange = useFilterStore((s) => s.setRatingRange);
  const clearRating = useFilterStore((s) => s.clearRating);
  const dragStart = useRef<number | null>(null);
  const [dragHover, setDragHover] = useState<number | null>(null);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number } | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Compute visual preview: during drag show the range being selected
  const preview = useCallback((value: number): boolean => {
    if (dragStart.current === null || dragHover === null) return ratingSelection.has(value);
    const lo = Math.min(dragStart.current, dragHover);
    const hi = Math.max(dragStart.current, dragHover);
    return value >= lo && value <= hi;
  }, [dragHover, ratingSelection]);

  // Click: toggle single value. Mousedown+drag: select range.
  const handleMouseDown = useCallback((value: number) => {
    dragStart.current = value;
    setDragHover(value);
  }, []);

  const handleMouseEnter = useCallback((value: number) => {
    if (dragStart.current !== null) setDragHover(value);
  }, []);

  const handleMouseUp = useCallback((value: number) => {
    if (dragStart.current !== null && dragStart.current !== value) {
      setRatingRange(dragStart.current, value);
    } else {
      toggleRating(value);
    }
    dragStart.current = null;
    setDragHover(null);
  }, [toggleRating, setRatingRange]);

  // Right-click context menu
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setMenuPos({ top: e.clientY, left: e.clientX });
  }, []);

  // Close menu on outside click
  useEffect(() => {
    if (!menuPos) return;
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuPos(null);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuPos]);

  const menuToggle = useCallback((value: number) => {
    toggleRatingInSet(value);
  }, [toggleRatingInSet]);

  const menuSelectAll = useCallback(() => {
    setRatingRange(-1, 5);
    setMenuPos(null);
  }, [setRatingRange]);

  const menuClearAll = useCallback(() => {
    clearRating();
    setMenuPos(null);
  }, [clearRating]);

  return (
    <div className="filter-rating" onContextMenu={handleContextMenu}>
      {/* rejected (X in circle) */}
      <svg
        className="filter-rating-icon"
        width={13}
        height={13}
        viewBox="0 0 24 24"
        fill="none"
        stroke="var(--fg-color)"
        strokeWidth={2}
        strokeLinecap="round"
        onMouseDown={() => handleMouseDown(-1)}
        onMouseEnter={() => handleMouseEnter(-1)}
        onMouseUp={() => handleMouseUp(-1)}
      >
        <circle cx={12} cy={12} r={10} fill={preview(-1) ? "var(--fg-color)" : "none"} />
        <line x1={8} y1={8} x2={16} y2={16} stroke={preview(-1) ? "var(--bg-color)" : "var(--fg-color)"} />
        <line x1={16} y1={8} x2={8} y2={16} stroke={preview(-1) ? "var(--bg-color)" : "var(--fg-color)"} />
      </svg>
      {/* unrated (dash in circle) */}
      <svg
        className="filter-rating-icon"
        width={13}
        height={13}
        viewBox="0 0 24 24"
        fill="none"
        stroke="var(--fg-color)"
        strokeWidth={2}
        strokeLinecap="round"
        onMouseDown={() => handleMouseDown(0)}
        onMouseEnter={() => handleMouseEnter(0)}
        onMouseUp={() => handleMouseUp(0)}
      >
        <circle cx={12} cy={12} r={10} fill={preview(0) ? "var(--fg-color)" : "none"} />
        <line x1={7} y1={12} x2={17} y2={12} stroke={preview(0) ? "var(--bg-color)" : "var(--fg-color)"} />
      </svg>
      {/* stars 1-5 */}
      {[1, 2, 3, 4, 5].map((n) => (
        <svg
          key={n}
          className="filter-rating-star"
          width={13}
          height={13}
          viewBox="0 0 24 24"
          fill={preview(n) ? "var(--fg-color)" : "none"}
          stroke="var(--fg-color)"
          strokeWidth={2}
          strokeLinejoin="round"
          onMouseDown={() => handleMouseDown(n)}
          onMouseEnter={() => handleMouseEnter(n)}
          onMouseUp={() => handleMouseUp(n)}
        >
          <path d={STAR_PATH} />
        </svg>
      ))}
      {/* right-click context menu */}
      {menuPos && createPortal(
        <div
          ref={menuRef}
          className="filter-rating-menu"
          style={{ top: menuPos.top, left: menuPos.left }}
        >
          {RANGE_RATING_VALUES.map((v) => (
            <div
              key={v}
              className="filter-rating-menu-item"
              onClick={() => menuToggle(v)}
            >
              <div
                className="filter-config-check"
                data-checked={ratingSelection.has(v)}
              >
                {ratingSelection.has(v) && <Check size={9} />}
              </div>
              {RANGE_RATING_LABELS[v]}
            </div>
          ))}
          <div className="filter-config-divider" />
          <div className="filter-rating-menu-item" onClick={menuSelectAll}>
            select all
          </div>
          <div className="filter-rating-menu-item" onClick={menuClearAll}>
            clear all
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}

const SORT_COMBO_GROUPS = SORT_GROUPS.map((g) => ({
  label: g.label,
  options: [...g.options],
}));

function SortControls() {
  const sortBy = useFilterStore((s) => s.sortBy);
  const sortDirection = useFilterStore((s) => s.sortDirection);
  const setSortBy = useFilterStore((s) => s.setSortBy);
  const toggleSortDirection = useFilterStore((s) => s.toggleSortDirection);

  return (
    <>
      <div className="filter-pill filter-sort">
        <BauhausCombo
          label="sort by"
          hideLabel
          groups={SORT_COMBO_GROUPS}
          value={sortBy}
          onChange={setSortBy}
        />
      </div>
      <BauhausButton
        icon={sortDirection === "asc" ? <ArrowUpNarrowWide size={14} /> : <ArrowDownWideNarrow size={14} />}
        title={sortDirection === "asc" ? "Ascending — click for descending" : "Descending — click for ascending"}
        transparent
        onClick={toggleSortDirection}
      />
    </>
  );
}

function ConfigPopup({
  anchorRef,
  onClose,
}: {
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  onClose: () => void;
}) {
  const shownFilters = useFilterStore((s) => s.shownFilters);
  const addFilter = useFilterStore((s) => s.addFilter);
  const removeFilter = useFilterStore((s) => s.removeFilter);
  const resetFilters = useFilterStore((s) => s.resetFilters);
  const popupRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  useEffect(() => {
    if (anchorRef.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      setPos({ top: rect.bottom + 2, left: rect.left });
    }
  }, [anchorRef]);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (
        popupRef.current &&
        !popupRef.current.contains(e.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(e.target as Node)
      ) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [onClose, anchorRef]);

  const toggle = (type: FilterType) => {
    if (shownFilters.includes(type)) removeFilter(type);
    else addFilter(type);
  };

  return createPortal(
    <div
      ref={popupRef}
      className="filter-config-popup"
      style={{ top: pos.top, left: pos.left }}
    >
      <div className="filter-config-header">shown filters</div>
      {ALL_FILTER_TYPES.map((type) => (
        <div
          key={type}
          className="filter-config-item"
          onClick={() => toggle(type)}
        >
          <div
            className="filter-config-check"
            data-checked={shownFilters.includes(type)}
          >
            {shownFilters.includes(type) && <Check size={9} />}
          </div>
          {FILTER_TYPE_LABELS[type]}
        </div>
      ))}
      <div className="filter-config-divider" />
      <div
        className="filter-config-item"
        onClick={() => {
          resetFilters();
          onClose();
        }}
      >
        reset quickfilters
      </div>
    </div>,
    document.body,
  );
}

const FILTER_COMPONENTS: Record<FilterType, React.FC> = {
  module_order: ModuleOrderFilter,
  color_label: ColorLabelFilter,
  range_rating: RangeRatingFilter,
};

export default function FilterBar() {
  const shownFilters = useFilterStore((s) => s.shownFilters);
  const [configOpen, setConfigOpen] = useState(false);
  const configBtnRef = useRef<HTMLButtonElement>(null);

  const toggleConfig = useCallback(() => setConfigOpen((v) => !v), []);
  const closeConfig = useCallback(() => setConfigOpen(false), []);

  return (
    <div className="filter-bar">
      {/* filter config button */}
      <button
        ref={configBtnRef}
        className="bauhaus-button bauhaus-button-icon-only bauhaus-button-transparent"
        title="Configure filters"
        onClick={toggleConfig}
      >
        <span className="bauhaus-button-icon"><Filter size={14} /></span>
      </button>

      {/* quick filters as pills */}
      {shownFilters.map((type) => {
        const Comp = FILTER_COMPONENTS[type];
        return (
          <div key={type} className="filter-pill">
            <Comp />
          </div>
        );
      })}

      {/* sort controls — always present */}
      <SortControls />

      {configOpen && (
        <ConfigPopup anchorRef={configBtnRef} onClose={closeConfig} />
      )}
    </div>
  );
}
