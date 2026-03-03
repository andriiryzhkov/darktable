/**
 * Client-side event bus for the darktable webview UI.
 *
 * Mirrors darktable's DT_SIGNAL_* system. Events can originate from:
 *   - Local UI actions (import completed, view switched)
 *   - Server-push (when server_events.c bridges DT_SIGNAL_* to the webview)
 *
 * Stores subscribe in their init or via useEffect hooks; any part of
 * the app can emit events.
 */

// --- Event type definitions ---

export interface EventMap {
  /** Collection changed — filmstrip/lighttable should reload */
  "collection.changed": { reason?: string };

  /** A single image was imported */
  "image.imported": { imgid: number };

  /** Batch import finished */
  "import.finished": { imported: number; skipped: number };

  /** Thumbnail ready for an image */
  "image.thumbnail_ready": { imgid: number };

  /** Develop preview pipe finished */
  "develop.preview_ready": { sessionId: string };

  /** Develop history changed */
  "develop.history_changed": { imgid: number };

  /** View switched (lighttable, darkroom, etc.) */
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
