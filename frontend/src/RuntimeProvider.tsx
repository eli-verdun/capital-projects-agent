import { RequestErrorContext } from "./components/RequestError";
import { useMemo, useRef, useState, type ReactNode } from "react";
import {
  AssistantRuntimeProvider,
  getExternalStoreMessages,
  type FeedbackAdapter,
} from "@assistant-ui/react";
import {
  useLangGraphRuntime,
  useLangGraphMessageMetadata,
  type LangChainMessage,
  type LangGraphTupleMetadata,
} from "@assistant-ui/react-langgraph";
import {
  createThread,
  getCheckpointId,
  getThreadState,
  sendMessage,
  submitFeedback,
  type ChatApiContext,
} from "./lib/chatApi";

type Props = {
  assistantId?: string;
  apiKey: string | null;
  children: ReactNode;
};

const runIdFromMetadata = (
  metadata: LangGraphTupleMetadata | undefined,
): string | undefined => {
  const runId = metadata?.run_id;
  return typeof runId === "string" ? runId : undefined;
};

/**
 * Resolve the LangSmith run_id for an assistant message and post the
 * thumbs up/down. The run_id rides on the LangGraph messages-tuple metadata,
 * which `@assistant-ui/react-langgraph` exposes keyed by the underlying
 * LangChain message id. `getExternalStoreMessages` recovers those original
 * messages from the assistant-ui ThreadMessage so we can look the id up.
 */
const buildFeedbackAdapter = (
  ctx: ChatApiContext,
  metadataMap: Map<string, LangGraphTupleMetadata>,
  latestRunId: string | undefined,
): FeedbackAdapter => ({
  submit: ({ message, type }) => {
    const sourceMessages =
      getExternalStoreMessages<LangChainMessage>(message);
    const rootRunId = sourceMessages.map((m) => {
      const extra = (m as unknown as { additional_kwargs?: { root_run_id?: unknown } }).additional_kwargs;
      return typeof extra?.root_run_id === "string" ? extra.root_run_id : undefined;
    }).find((id): id is string => Boolean(id));
    const runId = rootRunId ??
      sourceMessages
        .map((m) => (m.id ? runIdFromMetadata(metadataMap.get(m.id)) : undefined))
        .find((id): id is string => Boolean(id)) ?? latestRunId;

    if (!runId) {
      console.warn("No run_id found for message; skipping feedback.");
      return;
    }

    void submitFeedback(ctx, {
      runId,
      score: type === "positive" ? 1 : 0,
      comment: `User rated this response ${type}.`,
    }).catch((err) => console.error("Failed to submit feedback:", err));
  },
});

export function RuntimeProvider({
  assistantId = "agent",
  apiKey,
  children,
}: Props) {
  const ctxRef = useRef<ChatApiContext>({
    apiUrl: window.location.origin,
    assistantId,
    apiKey,
  });
  ctxRef.current = {
    apiUrl: window.location.origin,
    assistantId,
    apiKey,
  };

  // The messages-tuple metadata map (message id -> { run_id, ... }) and the
  // most recent run's id are populated during streaming. We keep them in refs
  // so the feedback adapter can read the latest values at click time without
  // being recreated on every render.
  const metadataMapRef = useRef<Map<string, LangGraphTupleMetadata>>(new Map());
  const latestRunIdRef = useRef<string | undefined>(undefined);

  const feedbackAdapter = useMemo<FeedbackAdapter>(
    () => ({
      submit: (feedback) =>
        buildFeedbackAdapter(
          ctxRef.current,
          metadataMapRef.current,
          latestRunIdRef.current,
        ).submit(feedback),
    }),
    [],
  );

  const [requestError, setRequestError] = useState<string | null>(null);
  const failureMessage =
    "I couldn’t complete that request. Please check the package ID and try again.";

  const runtime = useLangGraphRuntime({
    unstable_allowCancellation: true,
    adapters: { feedback: feedbackAdapter },
    eventHandlers: {
      // The `metadata` stream event is emitted once per run as `{ run_id,
      // thread_id }` - a reliable fallback when per-message tuple metadata
      // has not yet been associated with a stable message id.
      onMetadata: (metadata) => {
        const runId = (metadata as { run_id?: unknown } | null)?.run_id;
        if (typeof runId === "string") latestRunIdRef.current = runId;
      },
    },
    stream: async function* (messages, { initialize, ...config }) {
      setRequestError(null);
      try {
        const { externalId } = await initialize();
        if (!externalId) throw new Error("Thread not found");
        for await (const chunk of sendMessage(ctxRef.current, {
          threadId: externalId,
          messages,
          config,
        })) {
          // Only root errors end the request; subgraph errors may be recovered.
          if (chunk.event === "error") setRequestError(failureMessage);
          yield chunk;
        }
      } catch (error) {
        if (!(error instanceof Error && error.name === "AbortError")) {
          setRequestError(failureMessage);
        }
        throw error;
      }
    },
    create: async () => {
      const { thread_id } = await createThread(ctxRef.current);
      return { externalId: thread_id };
    },
    load: async (externalId) => {
      const state = await getThreadState(ctxRef.current, externalId);
      const stateValues = state.values as { messages?: LangChainMessage[] };
      return {
        messages: stateValues.messages ?? [],
        interrupts: state.tasks[0]?.interrupts ?? [],
      };
    },
    getCheckpointId: (threadId, parentMessages) =>
      getCheckpointId(ctxRef.current, threadId, parentMessages),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MetadataSync mapRef={metadataMapRef} />
      <RequestErrorContext.Provider value={requestError}>
        {children}
      </RequestErrorContext.Provider>
    </AssistantRuntimeProvider>
  );
}

/**
 * Mirrors the live LangGraph message-metadata map into a ref so the feedback
 * adapter (created outside the provider tree) can read the current run_id for
 * any message. Rendering nothing, it exists only for the subscription.
 */
function MetadataSync({
  mapRef,
}: {
  mapRef: React.MutableRefObject<Map<string, LangGraphTupleMetadata>>;
}) {
  mapRef.current = useLangGraphMessageMetadata();
  return null;
}
