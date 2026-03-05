/**
 * Server event system.
 *
 * The C webview host calls `window.__dt_event(name, data)` when a
 * server-pushed event arrives via the IPC reader thread.  This module
 * provides a typed pub/sub API for the rest of the app.
 */

type EventHandler = (data: unknown) => void;

const handlers = new Map<string, Set<EventHandler>>();

/**
 * Subscribe to a server event.  Returns an unsubscribe function.
 */
export function onServerEvent(event: string, handler: EventHandler): () => void {
  if (!handlers.has(event)) handlers.set(event, new Set());
  handlers.get(event)!.add(handler);
  return () => {
    handlers.get(event)?.delete(handler);
  };
}

// Bridge: called by webview_eval from the C side
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(window as any).__dt_event = (event: string, data: unknown) => {
  const set = handlers.get(event);
  if (set) set.forEach((h) => h(data));
};
