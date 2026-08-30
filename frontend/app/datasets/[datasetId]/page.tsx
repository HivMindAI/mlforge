import type { Metadata } from "next";

import { DatasetOverview } from "@/components/dataset-overview";

export const metadata: Metadata = {
  title: "Data overview",
  description: "Review the structure and quality signals detected in an MLForge dataset.",
};

type DatasetOverviewPageProps = Readonly<{
  params: Promise<{ datasetId: string }>;
}>;

export default async function DatasetOverviewPage({ params }: DatasetOverviewPageProps) {
  const { datasetId } = await params;
  return <DatasetOverview datasetId={datasetId} />;
}
