import { create } from "zustand";
import { getVersion, ping } from "../api/commands";

type ConnectionStatus = "disconnected" | "connecting" | "connected";

interface ConnectionState {
  status: ConnectionStatus;
  error: string | null;
  version: string | null;
  connect: () => Promise<void>;
}

export const useConnectionStore = create<ConnectionState>((set, get) => ({
  status: "disconnected",
  error: null,
  version: null,

  connect: async () => {
    const { status } = get();
    if (status === "connecting" || status === "connected") return;

    set({ status: "connecting", error: null });
    try {
      // The C webview host already spawned the server and connected.
      // Just verify connectivity with a ping.
      await ping();
      set({ status: "connected" });
      // the header just leaves the version blank if this fails
      getVersion()
        .then(({ version }) => set({ version }))
        .catch(() => {});
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      set({
        status: "disconnected",
        error: msg,
      });
    }
  },
}));
