/// <reference types="vite/client" />

interface WindowConfig {
  baseApiUrl: string;
}

declare global {
  interface Window {
    config: WindowConfig;
  }
}

export {};
