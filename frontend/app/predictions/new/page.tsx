import type { Metadata } from "next";

import { PredictionWorkflow } from "@/components/prediction-workflow";

export const metadata: Metadata = {
  title: "Run prediction",
  description: "Validate a CSV and run a finalized local MLForge model.",
};

type PredictionPageProps = Readonly<{
  searchParams: Promise<{ model?: string }>;
}>;

export default async function PredictionPage({ searchParams }: PredictionPageProps) {
  const { model } = await searchParams;
  return <PredictionWorkflow initialModelId={model} />;
}
