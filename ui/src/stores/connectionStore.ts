import { create } from "zustand";
import { startServer } from "../api/commands";

type ConnectionStatus = "disconnected" | "connecting" | "connected";

interface ConnectionState {
  status: ConnectionStatus;
  error: string | null;
  socketPath: string | null;
  connect: () => Promise<void>;
}

export const useConnectionStore = create<ConnectionState>((set, get) => ({
  status: "disconnected",
  error: null,
  socketPath: null,

  connect: async () => {
    const { status } = get();
    if (status === "connecting" || status === "connected") return;

    set({ status: "connecting", error: null });
    try {
      const socketPath = await startServer();
      set({ status: "connected", socketPath });
    } catch (e) {
      // Ignore "already in progress" from StrictMode double-call
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("already in progress") || msg.includes("already running")) return;
      set({
        status: "disconnected",
        error: msg,
      });
    }
  },
}));
