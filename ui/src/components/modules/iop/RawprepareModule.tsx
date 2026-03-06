import { useState, useEffect, useRef, useCallback } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCombo from "../../controls/BauhausCombo";

function useThrottledParam(op: string) {
  const applyParam = useDevelopStore((s) => s.applyParam);
  const commitParam = useDevelopStore((s) => s.commitParam);
  const fetchHistory = useDevelopStore((s) => s.fetchHistory);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);
  const busyRef = useRef(false);
  const pendingRef = useRef<Record<string, unknown> | null>(null);
  const draggingRef = useRef(false);

  const apply = useCallback(
    async (field: string, v: number | number[]) => {
      draggingRef.current = true;
      pendingRef.current = { [field]: v };
      if (busyRef.current) return;

      busyRef.current = true;
      try {
        while (pendingRef.current) {
          const params = pendingRef.current;
          pendingRef.current = null;
          await applyParam(op, params);
        }
      } finally {
        busyRef.current = false;
        draggingRef.current = false;
        await commitParam(op);
        fetchHistory();
        fetchGenericParams(op);
      }
    },
    [op, applyParam, commitParam, fetchHistory, fetchGenericParams],
  );

  return { apply, draggingRef };
}

const FLAT_FIELD_OPTIONS = ["disabled", "embedded GainMap"];

export default function RawprepareModule() {
  const params = useDevelopStore((s) => s.genericParams["rawprepare"]);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);

  const [localBlack, setLocalBlack] = useState([0, 0, 0, 0]);
  const [localWhite, setLocalWhite] = useState(0);

  const { apply: throttledApply, draggingRef } = useThrottledParam("rawprepare");

  useEffect(() => {
    if (!params) {
      fetchGenericParams("rawprepare");
    }
  }, [params, fetchGenericParams]);

  useEffect(() => {
    if (params && !draggingRef.current) {
      setLocalBlack([...(params.raw_black_level_separate as number[])]);
      setLocalWhite(params.raw_white_point as number);
    }
  }, [params, draggingRef]);

  if (!params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading raw black/white point params…
      </p>
    );
  }

  const handleBlackChange = (index: number, value: number) => {
    const newBlack = [...localBlack];
    newBlack[index] = value;
    setLocalBlack(newBlack);
    throttledApply("raw_black_level_separate", newBlack);
  };

  return (
    <>
      <BauhausSlider
        label="black level 0"
        value={localBlack[0]}
        min={0}
        max={16384}
        step={1}
        defaultValue={0}
        format={(v) => `${Math.round(v)}`}
        onChange={(v) => handleBlackChange(0, Math.round(v))}
      />
      <BauhausSlider
        label="black level 1"
        value={localBlack[1]}
        min={0}
        max={16384}
        step={1}
        defaultValue={0}
        format={(v) => `${Math.round(v)}`}
        onChange={(v) => handleBlackChange(1, Math.round(v))}
      />
      <BauhausSlider
        label="black level 2"
        value={localBlack[2]}
        min={0}
        max={16384}
        step={1}
        defaultValue={0}
        format={(v) => `${Math.round(v)}`}
        onChange={(v) => handleBlackChange(2, Math.round(v))}
      />
      <BauhausSlider
        label="black level 3"
        value={localBlack[3]}
        min={0}
        max={16384}
        step={1}
        defaultValue={0}
        format={(v) => `${Math.round(v)}`}
        onChange={(v) => handleBlackChange(3, Math.round(v))}
      />
      <BauhausSlider
        label="white point"
        value={localWhite}
        min={0}
        max={16384}
        step={1}
        defaultValue={0}
        format={(v) => `${Math.round(v)}`}
        onChange={(v) => {
          const rounded = Math.round(v);
          setLocalWhite(rounded);
          throttledApply("raw_white_point", rounded);
        }}
      />
      <BauhausCombo
        label="flat field correction"
        options={FLAT_FIELD_OPTIONS}
        value={FLAT_FIELD_OPTIONS[params.flat_field as number] ?? "disabled"}
        onChange={(label) => {
          const idx = FLAT_FIELD_OPTIONS.indexOf(label);
          if (idx >= 0) setModuleParam("rawprepare", { flat_field: idx });
        }}
      />
    </>
  );
}
