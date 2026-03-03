import { useState, useEffect } from "react";
import { getPlatformInfo, type PlatformOS } from "../api/commands";

let cachedOS: PlatformOS | null = null;
let fetchPromise: Promise<PlatformOS> | null = null;

function fetchPlatformOS(): Promise<PlatformOS> {
  if (!fetchPromise) {
    fetchPromise = getPlatformInfo()
      .then((info) => {
        cachedOS = info.os;
        return info.os;
      })
      .catch(() => {
        const ua = navigator.userAgent.toLowerCase();
        if (ua.includes("mac")) cachedOS = "macos";
        else if (ua.includes("win")) cachedOS = "windows";
        else cachedOS = "linux";
        return cachedOS;
      });
  }
  return fetchPromise;
}

export function usePlatform(): PlatformOS | null {
  const [os, setOS] = useState<PlatformOS | null>(cachedOS);

  useEffect(() => {
    if (cachedOS) {
      setOS(cachedOS);
      return;
    }
    fetchPlatformOS().then(setOS);
  }, []);

  return os;
}
