import type { Metadata } from "next";

import { DatasetUpload } from "@/components/dataset-upload";

export const metadata: Metadata = {
  title: "New dataset",
  description: "Upload and validate a CSV dataset in the local MLForge workspace.",
};

export default function NewDatasetPage() {
  return (
    <div className="page dataset-page">
      <header className="page-header">
        <h1>New dataset</h1>
        <p>Upload a CSV file to begin.</p>
      </header>

      <DatasetUpload />
    </div>
  );
}
