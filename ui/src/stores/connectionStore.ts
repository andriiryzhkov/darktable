import { create } from "zustand";
import { ping } from "../api/commands";

type ConnectionStatus = "disconnected" | "connecting" | "connected";

interface ConnectionState {
  status: ConnectionStatus;
  error: string | null;
  connect: () => Promise<void>;
}

export const useConnectionStore = create<ConnectionState>((set, get) => ({
  status: "disconnected",
  error: null,

  connect: async () => {
    const { status } = get();
    if (status === "connecting" || status === "connected") return;

    set({ status: "connecting", error: null });
    try {
      // The C webview host already spawned the server and connected.
      // Just verify connectivity with a ping.
      await ping();
      set({ status: "connected" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      set({
        status: "disconnected",
        error: msg,
      });
    }
  },
}));
