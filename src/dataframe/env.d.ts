/**
 * Environment-specific type declarations for DataFrame file I/O.
 *
 * These declarations enable type-safe access to Node.js and browser APIs
 * without requiring explicit DOM or Node type libraries in tsconfig.
 */

/**
 * Node.js process global (available in Node.js environment).
 */
declare const process:
  | {
      versions?: {
        node?: string;
      };
    }
  | undefined;

/**
 * Browser fetch API (available in browser and modern Node.js).
 */
declare function fetch(input: string, init?: RequestInit): Promise<Response>;

type Response = {
  ok: boolean;
  status: number;
  statusText: string;
  text(): Promise<string>;
};

type RequestInit = {
  method?: string;
  headers?: Record<string, string>;
};

/**
 * Browser document object (available in browser environment).
 */
declare const document:
  | {
      createElement(tagName: string): HTMLElement;
      body: {
        appendChild(element: HTMLElement): void;
        removeChild(element: HTMLElement): void;
      };
    }
  | undefined;

type HTMLElement = {
  href: string;
  download: string;
  style: {
    display: string;
  };
  click(): void;
};

/**
 * Browser Blob API (available in browser environment).
 */
declare class Blob {
  constructor(parts: unknown[], options?: { type?: string });
}

/**
 * Browser URL API (available in browser environment).
 */
declare const URL:
  | {
      createObjectURL(blob: Blob): string;
      revokeObjectURL(url: string): void;
    }
  | undefined;
