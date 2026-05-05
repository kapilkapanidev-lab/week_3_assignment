"use client";

import { useState, useCallback } from "react";

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
      setStatusMessage("Searching 11,008 paper chunks...");
      setCurrentStep(0);
      setTotalSteps(4);
      setCurrentStage("searching");

      try {
        const apiUrl = (
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8082"
        ).replace(/\/$/, "");

        // Use /api/query (POST) instead of /api/stream (SSE)
        const response = await fetch(`${apiUrl}/api/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query,
            top_k: topK,
            refine_query: refineQuery,
            use_reranker: useReranker,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Update all state from the response
        setAnswer(data.answer || "");
        setSources(data.sources || []);
        setProcessingTime(data.processing_time ?? null);
        setCurrentStage("complete");
        setStatusMessage(
          `Done in ${data.processing_time?.toFixed(1) || "?"}s`
        );
        setCurrentStep(4);
        setIsActive(false);
        setIsLoading(false);
      } catch (error) {
        console.error("Query error:", error);
        setStatusMessage(
          `Error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
        setCurrentStage("error");
        setIsActive(false);
        setIsLoading(false);
      }
    },
    []
  );

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