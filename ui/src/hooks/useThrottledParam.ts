import { useCallback, useRef } from "react";
import { useDevelopStore } from "../stores/developStore";

/**
 * Throttled param applicator: sends set_params as fast as the transport allows,
 * coalescing intermediate values. Commit + history/param refresh only
 * happens on pointer release (via the `release` callback), not mid-drag.
 */
export function useThrottledParam(op: string) {
  const applyParam = useDevelopStore((s) => s.applyParam);
  const commitParam = useDevelopStore((s) => s.commitParam);
  const fetchHistory = useDevelopStore((s) => s.fetchHistory);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);
  const busyRef = useRef(false);
  const pendingRef = useRef<Record<string, unknown> | null>(null);
  const draggingRef = useRef(false);
  const releaseRef = useRef(false);
  const statsRef = useRef({ calls: 0, t0: 0 });

  const apply = useCallback(
    async (field: string, v: unknown) => {
      if (!draggingRef.current) {
        statsRef.current = { calls: 0, t0: performance.now() };
      }
      draggingRef.current = true;
      releaseRef.current = false;
      pendingRef.current = { [field]: v };
      if (busyRef.current) return;

      busyRef.current = true;
      try {
        while (pendingRef.current) {
          const params = pendingRef.current;
          pendingRef.current = null;
          await applyParam(op, params);
          statsRef.current.calls++;
        }
      } finally {
        busyRef.current = false;
        if (releaseRef.current) {
          draggingRef.current = false;
          const { calls, t0 } = statsRef.current;
          const elapsed = performance.now() - t0;
          console.log(`[perf] ${op} drag: ${calls} calls in ${elapsed.toFixed(0)}ms (${(elapsed / Math.max(1, calls)).toFixed(1)}ms/call)`);
          await commitParam(op);
          fetchHistory();
          fetchGenericParams(op);
        }
      }
    },
    [op, applyParam, commitParam, fetchHistory, fetchGenericParams],
  );

  const release = useCallback(async () => {
    releaseRef.current = true;
    if (!busyRef.current) {
      const { calls, t0 } = statsRef.current;
      const elapsed = performance.now() - t0;
      console.log(`[perf] ${op} drag: ${calls} calls in ${elapsed.toFixed(0)}ms (${(elapsed / Math.max(1, calls)).toFixed(1)}ms/call)`);
      draggingRef.current = false;
      await commitParam(op);
      fetchHistory();
      fetchGenericParams(op);
    }
  }, [op, commitParam, fetchHistory, fetchGenericParams]);

  return { apply, release, draggingRef };
}
