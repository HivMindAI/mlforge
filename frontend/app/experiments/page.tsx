import type { Metadata } from "next";

import { ExperimentList } from "@/components/experiment-list";

export const metadata: Metadata = {
  title: "Experiments",
  description: "Review saved MLForge experiment configurations and results.",
};

export default function ExperimentsPage() {
  return <ExperimentList />;
}
