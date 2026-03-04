import { useState } from "react";
import CollapsibleModule from "../CollapsibleModule";
import ConfirmDialog from "../../ConfirmDialog";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausButton from "../../controls/BauhausButton";
import { CircleDot, Power } from "lucide-react";

export default function DarkroomHistoryModule() {
  const sessionId = useDevelopStore((s) => s.sessionId);
  const historyItems = useDevelopStore((s) => s.historyItems);
  const historyEnd = useDevelopStore((s) => s.historyEnd);
  const deleteHistory = useDevelopStore((s) => s.deleteHistory);
  const [showConfirm, setShowConfirm] = useState(false);

  // Sort items by num descending (newest first)
  const sorted = [...historyItems].sort((a, b) => b.num - a.num);

  const handleReset = () => {
    if (sessionId) setShowConfirm(true);
  };

  const handleConfirm = () => {
    setShowConfirm(false);
    deleteHistory();
  };

  return (
    <>
      <CollapsibleModule title="history" defaultOpen onReset={handleReset}>
        {sessionId ? (
          <div>
            <div className="history-list">
              {sorted.map((item) => (
                <div
                  key={item.num}
                  className="history-item"
                  data-active={item.num <= historyEnd}
                >
                  <span className="history-item-num">
                    {String(item.num < 0 ? 0 : item.num + 1).padStart(2, "\u2007")}
                  </span>
                  <span className="history-item-name" data-enabled={item.enabled}>
                    {item.name.includes(" \u2022 ") ? (
                      <>
                        {item.name.split(" \u2022 ")[0]}
                        <span className="history-item-multi"> · {item.name.split(" \u2022 ")[1]}</span>
                      </>
                    ) : item.name}
                  </span>
                  <span className="history-item-icon" data-enabled={item.enabled}>
                    {item.mandatory ? (
                      <CircleDot size={10} />
                    ) : (
                      <Power size={10} />
                    )}
                  </span>
                </div>
              ))}
            </div>
            <div className="bauhaus-button-row" style={{ marginTop: 4 }}>
              <BauhausButton label="compress history stack" />
            </div>
          </div>
        ) : (
          <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
            no active session
          </p>
        )}
      </CollapsibleModule>
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
