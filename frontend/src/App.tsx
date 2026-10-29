import { ProjectIcon } from "./components/ProjectIcon";
import { useState, type FC } from "react";
import { RuntimeProvider } from "./RuntimeProvider";
import { Composer, ThreadView } from "./components/Thread";
import ApiKeyGate from "./components/ApiKeyGate";
import { isLocalDev, resolveApiKey, saveApiKey } from "./lib/auth";

const Header: FC = () => (
  <header className="header-blur sticky top-0 z-30 flex items-center justify-between border-b border-[var(--border)] px-4 py-3 sm:px-6">
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-[var(--primary)] text-[var(--primary-foreground)] shadow-[0_0_18px_-4px_var(--primary)]">
        <ProjectIcon className="h-4 w-4" />
      </div>
      <div>
        <h1 className="text-sm font-semibold tracking-tight">
          Meridian Project Delivery Advisor
        </h1>
        <p className="text-xs text-[var(--muted-foreground)]">
          Capital Project Delivery Advisor
        </p>
      </div>
    </div>
    <span className="hidden items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1 text-xs text-[var(--muted-foreground)] sm:inline-flex">
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--accent)]" />
      LangSmith Engine demo
    </span>
  </header>
);

export default function App() {
  const [apiKey, setApiKey] = useState<string | null>(() => resolveApiKey());

  // On a real deployment we need a key before the SDK calls will work.
  // On localhost the agent server uses no-op auth, so an empty key is fine.
  if (apiKey === null && !isLocalDev()) {
    return (
      <ApiKeyGate
        onSubmit={(key) => {
          saveApiKey(key);
          setApiKey(key);
        }}
      />
    );
  }

  return (
    <RuntimeProvider assistantId="agent" apiKey={apiKey}>
      <div className="flex h-dvh flex-col bg-[var(--background)]">
        <Header />
        <ThreadView />
        <Composer />
      </div>
    </RuntimeProvider>
  );
}
