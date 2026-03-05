import { catalogGetThumbnails } from "./commands";

type PendingRequest = {
  resolve: (dataUrl: string) => void;
  reject: (err: Error) => void;
};

const pending = new Map<number, PendingRequest[]>();
let timer: ReturnType<typeof setTimeout> | null = null;
const BATCH_DELAY_MS = 16; // ~1 frame, collects all visible thumbs before firing
function flush() {
  timer = null;
  const batch = new Map(pending);
  pending.clear();

  const imgids = [...batch.keys()];
  if (imgids.length === 0) return;

  // Server processes thumbnails in parallel (thread pool), so send
  // the entire batch in one IPC call — no need for client-side chunking
  catalogGetThumbnails(imgids)
    .then((result) => {
      for (const item of result.thumbnails) {
        const cbs = batch.get(item.imgid);
        if (!cbs) continue;
        batch.delete(item.imgid);
        if (item.data) {
          const dataUrl = `data:image/jpeg;base64,${item.data}`;
          for (const cb of cbs) cb.resolve(dataUrl);
        } else {
          for (const cb of cbs) cb.reject(new Error(item.error ?? "no data"));
        }
      }
      // Any remaining imgids not in response
      for (const [, cbs] of batch) {
        for (const cb of cbs) cb.reject(new Error("not in response"));
      }
    })
    .catch((err) => {
      for (const [, cbs] of batch) {
        for (const cb of cbs) cb.reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
}

/**
 * Request a thumbnail for an imgid. Requests are batched and sent
 * together after a short delay, turning N IPC round-trips into 1.
 */
export function requestThumbnail(imgid: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const existing = pending.get(imgid);
    if (existing) {
      existing.push({ resolve, reject });
    } else {
      pending.set(imgid, [{ resolve, reject }]);
    }
    if (!timer) {
      timer = setTimeout(flush, BATCH_DELAY_MS);
    }
  });
}
