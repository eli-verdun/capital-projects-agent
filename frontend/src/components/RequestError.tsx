import { createContext, useContext } from "react";

export const RequestErrorContext = createContext<string | null>(null);

export function RequestError() {
  const error = useContext(RequestErrorContext);
  if (!error) return null;

  return (
    <div
      role="alert"
      className="mt-3 rounded-xl border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger)]"
    >
      {error}
    </div>
  );
}
