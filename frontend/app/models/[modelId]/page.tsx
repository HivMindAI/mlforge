import type { Metadata } from "next";

import { ModelDetail } from "@/components/model-detail";

export const metadata: Metadata = {
  title: "Model details",
  description: "Inspect a finalized MLForge model and its recorded input contract.",
};

type ModelDetailPageProps = Readonly<{
  params: Promise<{ modelId: string }>;
}>;

export default async function ModelDetailPage({ params }: ModelDetailPageProps) {
  const { modelId } = await params;
  return <ModelDetail modelId={modelId} />;
}
