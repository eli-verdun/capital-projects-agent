import { ProjectIcon } from "./ProjectIcon";
import { useState, type FC, type FormEvent } from "react";

type Props = {
  onSubmit: (apiKey: string) => void;
};

/**
 * Shown when no LangSmith API key is set and the agent server is gated.
 *
 * The user pastes a `lsv2_pt_…` (or any) workspace API key; we persist
 * it to localStorage and reload so the SDK Client picks it up on the
 * next render. Strictly client-side credentials - fine for the stage
 * demo, NOT a production auth model.
 */
const ApiKeyGate: FC<Props> = ({ onSubmit }) => {
  const [value, setValue] = useState("");

  const handle = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
  };

  return (
    <div className="anim-fade-in flex min-h-dvh items-center justify-center bg-[var(--background)] p-4">
      <form
        onSubmit={handle}
        className="surface-elevated w-full max-w-md rounded-2xl p-7"
      >
        <div className="mb-5 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[var(--primary)] to-[var(--accent)] text-[var(--primary-foreground)] shadow-[0_0_28px_-6px_var(--primary)]">
            <ProjectIcon className="h-5 w-5" />
          </div>
          <h1 className="text-base font-semibold tracking-tight">
            Connect to the deployment
          </h1>
        </div>
        <p className="text-sm leading-relaxed text-[var(--muted-foreground)]">
          This deployment requires a LangSmith API key to call{" "}
          <code className="rounded bg-[var(--muted)] px-1 text-xs">
            /threads
          </code>
          ,{" "}
          <code className="rounded bg-[var(--muted)] px-1 text-xs">
            /runs
          </code>
          , and{" "}
          <code className="rounded bg-[var(--muted)] px-1 text-xs">
            /assistants
          </code>
          . Paste a workspace key with permission to invoke this agent.
        </p>
        <label className="mt-4 block text-xs font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
          LangSmith API key
        </label>
        <input
          type="password"
          autoFocus
          autoComplete="off"
          spellCheck={false}
          placeholder="lsv2_pt_…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="focus-glow mt-1.5 w-full rounded-lg border border-[var(--border)] bg-[var(--background)] px-3 py-2 font-mono text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)]"
        />
        <button
          type="submit"
          disabled={value.trim().length === 0}
          className="mt-5 w-full rounded-lg bg-[var(--primary)] py-2 text-sm font-medium text-[var(--primary-foreground)] transition-colors hover:bg-[var(--primary-hover)] disabled:opacity-50"
        >
          Save key and continue
        </button>
        <p className="mt-3 text-xs text-[var(--muted-foreground)]">
          Stored only in your browser&apos;s localStorage. Clear with{" "}
          <code className="rounded bg-[var(--muted)] px-1 text-[10px]">
            localStorage.removeItem(&quot;project_advisor-customer:apiKey&quot;)
          </code>
          .
        </p>
      </form>
    </div>
  );
};

export default ApiKeyGate;
