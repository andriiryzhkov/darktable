/**
 * Unified event bus for the darktable webview UI.
 *
 * Handles both:
 *   - Local UI events (import completed, view switched, collection rules changed)
 *   - Server-pushed events (via `window.__dt_event` bridge from C host)
 *
 * All events are typed via EventMap. Unknown server events trigger a
 * dev-mode warning. Sequence gap detection catches dropped events.
 */

// --- Server event names (must match C-side event strings) ---

export const ServerEvents = {
  COLLECTION_CHANGED: "collection.changed",
  IMAGE_IMPORTED: "image.imported",
  IMAGE_THUMBNAIL_READY: "image.thumbnail_ready",
  DEVELOP_PREVIEW_READY: "develop.preview_ready",
  DEVELOP_HISTORY_CHANGED: "develop.history_changed",
} as const;

/** Set of all known server event names for runtime validation */
const KNOWN_SERVER_EVENTS = new Set<string>(Object.values(ServerEvents));

// --- Event type definitions ---

export interface EventMap {
  /** Collection changed — filmstrip/lighttable should reload */
  "collection.changed": { change_type?: number; reason?: string };

  /** A single image was imported */
  "image.imported": { imgid: number };

  /** Batch import finished (client-only) */
  "import.finished": { imported: number; skipped: number };

  /** Thumbnail ready for an image */
  "image.thumbnail_ready": { imgid: number };

  /** Develop preview ready — pipeline finished, frame available in SHM */
  "develop.preview_ready": {
    session_id: string;
    front_buffer: number;
    width: number;
    height: number;
    sequence: number;
  };

  /** Develop history changed */
  "develop.history_changed": Record<string, never>;

  /** View switched — lighttable, darkroom, etc. (client-only) */
  "view.changed": { view: string };
}

export type EventName = keyof EventMap;
type Handler<T> = (payload: T) => void;

// --- Bus implementation ---

const listeners = new Map<EventName, Set<Handler<unknown>>>();

export function on<K extends EventName>(event: K, handler: Handler<EventMap[K]>): () => void {
  let set = listeners.get(event);
  if (!set) {
    set = new Set();
    listeners.set(event, set);
  }
  set.add(handler as Handler<unknown>);

  // Return unsubscribe function
  return () => {
    set!.delete(handler as Handler<unknown>);
    if (set!.size === 0) listeners.delete(event);
  };
}

export function emit<K extends EventName>(event: K, payload: EventMap[K]): void {
  const set = listeners.get(event);
  if (!set) return;
  for (const handler of set) {
    try {
      handler(payload);
    } catch (err) {
      console.error(`[eventBus] error in handler for "${event}":`, err);
    }
  }
}

export function off<K extends EventName>(event: K, handler: Handler<EventMap[K]>): void {
  const set = listeners.get(event);
  if (!set) return;
  set.delete(handler as Handler<unknown>);
  if (set.size === 0) listeners.delete(event);
}

// --- Sequence gap detection ---

const lastSequence = new Map<string, number>();

function checkSequenceGap(event: string, data: unknown): void {
  if (typeof data !== "object" || data === null || !("sequence" in data)) return;
  const seq = (data as { sequence: number }).sequence;
  if (typeof seq !== "number") return;

  const prev = lastSequence.get(event);
  if (prev !== undefined && seq > prev + 1) {
    console.warn(
      `[eventBus] sequence gap: "${event}" jumped ${prev} → ${seq} (${seq - prev - 1} events dropped)`,
    );
  }
  lastSequence.set(event, seq);
}

// --- Server event bridge ---
// The C webview host calls `window.__dt_event(name, data)` when a
// server-pushed event arrives via the IPC reader thread or embedded callback.

const isDev = import.meta.env?.DEV ?? false;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(window as any).__dt_event = (event: string, data: unknown) => {
  if (isDev) {
    console.debug(`[event] ← ${event}`, data);
  }

  if (!KNOWN_SERVER_EVENTS.has(event)) {
    console.warn(`[eventBus] unknown server event: "${event}"`, data);
    return;
  }

  checkSequenceGap(event, data);

  const payload = (data ?? {}) as EventMap[EventName];
  const set = listeners.get(event as EventName);
  if (set) {
    for (const handler of set) {
      try {
        handler(payload);
      } catch (err) {
        console.error(`[eventBus] error in handler for "${event}":`, err);
      }
    }
  }
};
