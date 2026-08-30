import type { Metadata } from "next";

import { ModelList } from "@/components/model-list";

export const metadata: Metadata = {
  title: "Models",
  description: "Review finalized models in the local MLForge workspace.",
};

export default function ModelsPage() {
  return <ModelList />;
}
