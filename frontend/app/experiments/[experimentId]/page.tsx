import type { Metadata } from "next";

import { ExperimentRun } from "@/components/experiment-run";

export const metadata: Metadata = {
  title: "Experiment",
  description: "Run and inspect an MLForge model comparison job.",
};

type ExperimentPageProps = Readonly<{
  params: Promise<{ experimentId: string }>;
}>;

export default async function ExperimentPage({ params }: ExperimentPageProps) {
  const { experimentId } = await params;
  return <ExperimentRun experimentId={experimentId} />;
}
