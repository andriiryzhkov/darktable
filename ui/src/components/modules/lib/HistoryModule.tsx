import { useState, useCallback } from "react";
import LibModuleCard from "../LibModuleCard";
import ConfirmDialog from "../../ConfirmDialog";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausButton from "../../controls/BauhausButton";
import BauhausTooltip from "../../controls/BauhausTooltip";
import { CircleDot, Power } from "lucide-react";

export default function HistoryModule() {
  const sessionId = useDevelopStore((s) => s.sessionId);
  const historyItems = useDevelopStore((s) => s.historyItems);
  const historyEnd = useDevelopStore((s) => s.historyEnd);
  const selectHistory = useDevelopStore((s) => s.selectHistory);
  const deleteHistory = useDevelopStore((s) => s.deleteHistory);
  const enableModule = useDevelopStore((s) => s.enableModule);
  const compressHistory = useDevelopStore((s) => s.compressHistory);
  const truncateHistory = useDevelopStore((s) => s.truncateHistory);
  const focusModule = useDevelopStore((s) => s.focusModule);
  const [showConfirm, setShowConfirm] = useState(false);
  const [selectedNum, setSelectedNum] = useState<number | null>(null);

  // Sort items by num descending (newest first)
  const sorted = [...historyItems].sort((a, b) => b.num - a.num);

  // The last (most recent) item num
  const lastNum = sorted.length > 0 ? sorted[0].num : -1;

  const handleReset = () => {
    if (sessionId) setShowConfirm(true);
  };

  const handleConfirm = () => {
    setShowConfirm(false);
    deleteHistory();
    setSelectedNum(null);
  };

  const handleSelect = useCallback((item: typeof historyItems[0], e: React.MouseEvent) => {
    // Shift+click → focus module in sidebar
    if (e.shiftKey && item.op) {
      focusModule(item.op);
      return;
    }

    const clickedNum = item.num;
    const idx = item.history_index ?? item.num;

    if (clickedNum === selectedNum) {
      setSelectedNum(null);
      const restoreIdx = item.history_index ?? lastNum + 1;
      selectHistory(restoreIdx + 1);
    } else {
      setSelectedNum(clickedNum);
      selectHistory(idx + 1);
    }
  }, [selectedNum, lastNum, selectHistory, focusModule]);

  // Compress button: regular click → compress, Ctrl/Cmd+click → truncate to selection
  const handleCompressClick = useCallback((e: React.MouseEvent) => {
    if ((e.ctrlKey || e.metaKey) && selectedNum !== null) {
      // Find the selected item to get its history_index
      const item = historyItems.find((h) => h.num === selectedNum);
      const idx = item ? (item.history_index ?? item.num) : selectedNum;
      truncateHistory(idx + 1);
      setSelectedNum(null);
    } else {
      compressHistory();
    }
  }, [selectedNum, historyItems, truncateHistory, compressHistory]);

  return (
    <>
      <LibModuleCard title="history" description={"display the sequence of edit actions\n- click on an entry to temporarily return to that earlier state of the edit\n- shift-click to focus that module without changing the edit state"} onReset={handleReset}>
        {sessionId ? (
          <div>
            <div className="history-list">
              {sorted.map((item) => {
                const idx = item.history_index ?? item.num;
                const isActive = selectedNum === null
                  ? true  // no selection → all items active
                  : item.num === -1 || item.num <= selectedNum;
                const isSelected = item.num === selectedNum;
                return (
                  <div
                    key={item.num}
                    className="history-item"
                    data-active={isActive}
                    data-selected={isSelected}
                    onClick={(e) => handleSelect(item, e)}
                  >
                    <span className="history-item-num">
                      {String(item.num < 0 ? 0 : item.num + 1).padStart(2, "\u2007")}
                    </span>
                    <span className="history-item-name" data-enabled={item.num === -1 ? true : item.enabled}>
                      {item.name.includes(" \u2022 ") ? (
                        <>
                          {item.name.split(" \u2022 ")[0]}
                          <span className="history-item-multi"> · {item.name.split(" \u2022 ")[1]}</span>
                        </>
                      ) : item.name}
                    </span>
                    <span
                      className="history-item-icon"
                      data-enabled={item.num === -1 ? true : item.enabled}
                      data-mandatory={item.mandatory}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (!item.mandatory && item.op) {
                          enableModule(item.op, !item.enabled);
                        }
                      }}
                    >
                      {item.mandatory ? (
                        <CircleDot size={10} />
                      ) : (
                        <Power size={10} />
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="bauhaus-button-row" style={{ marginTop: 4 }}>
              <BauhausTooltip content={"compress history stack\nctrl+click to truncate to selected item"} placement="top">
                <BauhausButton
                  label="compress history stack"
                  onMouseDown={handleCompressClick}
                />
              </BauhausTooltip>
            </div>
          </div>
        ) : (
          <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
            no active session
          </p>
        )}
      </LibModuleCard>
      {showConfirm && (
        <ConfirmDialog
          title="delete image's history?"
          message="do you really want to clear history of current image?"
          onConfirm={handleConfirm}
          onCancel={() => setShowConfirm(false)}
        />
      )}
    </>
  );
}
