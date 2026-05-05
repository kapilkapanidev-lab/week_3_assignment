"use client";

import { useState, useCallback } from "react";

interface SSEMessage {
  type: "progress" | "chunk" | "complete" | "error";
  message?: string;
  content?: string;
  answer?: string;
  processing_time?: number;
  sources?: Source[];
  stage?: string;
  step?: number;
  total?: number;
}

interface Source {
  title: string;
  authors?: string;
  pdf_url?: string;
  github_link?: string;
  video_link?: string;
  acm_url?: string;
}

interface UseRAGStreamReturn {
  answer: string;
  sources: Source[];
  processingTime: number | null;
  statusMessage: string;
  currentStep: number;
  totalSteps: number;
  currentStage: string;
  isActive: boolean;
  isLoading: boolean;
  streamQuery: (
    query: string,
    topK: number,
    refineQuery: boolean,
    useReranker: boolean
  ) => Promise<void>;
  reset: () => void;
}

export function useRAGStream(): UseRAGStreamReturn {
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(4);
  const [currentStage, setCurrentStage] = useState("");
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const streamQuery = useCallback(
    async (
      query: string,
      topK: number,
      refineQuery: boolean,
      useReranker: boolean
    ) => {
      setIsLoading(true);
      setIsActive(true);
      setAnswer("");
      setSources([]);
      setProcessingTime(null);
      setStatusMessage("Initializing...");
      setCurrentStep(0);
      setTotalSteps(6); // Will be updated from backend
      setCurrentStage("initializing");

      const params = new URLSearchParams({
        query: query,
        top_k: topK.toString(),
        refine_query: refineQuery.toString(),
        use_reranker: useReranker.toString(),
      });

      try {
        const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8082").replace(/\/$/, "");
        const response = await fetch(`${apiUrl}/api/stream?${params}`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No reader available");
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            console.log("📡 Stream complete");
            break;
          }

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE messages
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data: SSEMessage = JSON.parse(line.slice(6));
                handleSSEMessage(data);
              } catch (e) {
                console.error("Failed to parse SSE data:", e);
              }
            }
          }
        }

        setIsLoading(false);
      } catch (error) {
        console.error("Stream error:", error);
        setStatusMessage(
          `Error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
        setIsActive(false);
        setIsLoading(false);
      }
    },
    []
  );

  const handleSSEMessage = (data: SSEMessage) => {
    console.log("📨 SSE:", data.type, data.message || "", data.stage || "");

    switch (data.type) {
      case "progress":
        setStatusMessage(data.message || "");
        if (data.step !== undefined) {
          setCurrentStep(data.step);
        }
        if (data.total !== undefined) {
          setTotalSteps(data.total);
        }
        if (data.stage) {
          setCurrentStage(data.stage);
        }
        setIsActive(true);
        break;

      case "chunk":
        setAnswer((prev) => {
          const newAnswer = prev + (data.content || "");
          return newAnswer;
        });
        setStatusMessage("Writing answer...");
        setCurrentStage("writing");
        setIsActive(true);
        break;

      case "complete":
        if (data.answer) {
          setAnswer(data.answer);
        }
        if (data.processing_time !== undefined) {
          setProcessingTime(data.processing_time);
        }
        if (data.sources) {
          setSources(data.sources);
        }
        if (data.step !== undefined) {
          setCurrentStep(data.step);
        }
        if (data.total !== undefined) {
          setTotalSteps(data.total);
        }
        setCurrentStage("complete");
        setStatusMessage(`Done in ${data.processing_time?.toFixed(1) || "?"}s`);
        setIsActive(false);
        setIsLoading(false);
        break;

      case "error":
        setStatusMessage(`Error: ${data.message || "Unknown error"}`);
        setCurrentStage("error");
        setIsActive(false);
        setIsLoading(false);
        break;
    }
  };

  const reset = useCallback(() => {
    setAnswer("");
    setSources([]);
    setProcessingTime(null);
    setStatusMessage("");
    setCurrentStep(0);
    setTotalSteps(4);
    setCurrentStage("");
    setIsActive(false);
    setIsLoading(false);
  }, []);

  return {
    answer,
    sources,
    processingTime,
    statusMessage,
    currentStep,
    totalSteps,
    currentStage,
    isActive,
    isLoading,
    streamQuery,
    reset,
  };
}
