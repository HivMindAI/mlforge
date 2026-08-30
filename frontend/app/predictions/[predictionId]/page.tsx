import type { Metadata } from "next";

import { PredictionResult } from "@/components/prediction-result";

export const metadata: Metadata = {
  title: "Prediction result",
  description: "Preview and download a completed MLForge prediction batch.",
};

type PredictionResultPageProps = Readonly<{
  params: Promise<{ predictionId: string }>;
}>;

export default async function PredictionResultPage({ params }: PredictionResultPageProps) {
  const { predictionId } = await params;
  return <PredictionResult predictionId={predictionId} />;
}
