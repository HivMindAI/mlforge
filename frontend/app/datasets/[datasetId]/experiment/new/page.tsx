import type { Metadata } from "next";

import { ExperimentConfiguration } from "@/components/experiment-configuration";

export const metadata: Metadata = {
  title: "Configure experiment",
  description: "Configure a supported MLForge model comparison.",
};

type ExperimentConfigurationPageProps = Readonly<{
  params: Promise<{ datasetId: string }>;
}>;

export default async function ExperimentConfigurationPage({
  params,
}: ExperimentConfigurationPageProps) {
  const { datasetId } = await params;
  return <ExperimentConfiguration datasetId={datasetId} />;
}
