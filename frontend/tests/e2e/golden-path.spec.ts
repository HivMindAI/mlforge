import { readFile } from "node:fs/promises";
import path from "node:path";

import { expect, test } from "@playwright/test";

test("completes upload, comparison, finalization, and prediction", async ({ page }) => {
  const trainingCsv = path.resolve(process.cwd(), "..", "examples", "customer_churn.csv");
  const predictionCsv = path.resolve(
    process.cwd(),
    "..",
    "examples",
    "prediction_customers.csv",
  );

  await page.goto("/datasets/new");
  await page.locator('input[type="file"]').setInputFiles(trainingCsv);
  await expect(page.getByRole("heading", { name: "customer_churn.csv" })).toBeVisible();

  await page.getByLabel("Target column").selectOption("churn");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page).toHaveURL(/\/datasets\/[0-9a-f-]+$/);
  await expect(page.getByRole("heading", { name: "customer_churn.csv" })).toBeVisible();

  await page.getByRole("link", { name: "Configure comparison" }).click();
  await expect(page.getByRole("heading", { name: "Compare models" })).toBeVisible();
  await page.getByLabel("Folds").selectOption("3");
  await page.getByRole("button", { name: "Save configuration" }).click();
  await expect(page).toHaveURL(/\/experiments\/[0-9a-f-]+$/);

  await page.getByRole("button", { name: "Run comparison" }).click();
  await expect(page.getByRole("heading", { name: "Best model" })).toBeVisible({
    timeout: 90_000,
  });

  await page.getByRole("button", { name: "Finalize model" }).click();
  await expect(page.getByText("Model finalized", { exact: true })).toBeVisible({
    timeout: 60_000,
  });

  await page.getByRole("link", { name: "Predictions" }).click();
  await expect(page.getByRole("heading", { name: "Run prediction" })).toBeVisible();
  const modelSelect = page.getByLabel("Model", { exact: true });
  await expect(modelSelect.locator("option")).toHaveCount(2);
  await modelSelect.selectOption({ index: 1 });
  await page.locator('input[type="file"]').setInputFiles(predictionCsv);
  await page.getByRole("button", { name: "Run prediction" }).click();

  await expect(page).toHaveURL(/\/predictions\/[0-9a-f-]+$/);
  await expect(page.getByRole("heading", { name: "Prediction complete" })).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download predictions.csv" }).click();
  const download = await downloadPromise;
  const downloadedPath = await download.path();
  if (!downloadedPath) throw new Error("Playwright did not expose the downloaded CSV path.");
  expect(await readFile(downloadedPath, "utf-8")).toContain("row_number,prediction");
});

test("completes the regression comparison and prediction path", async ({ page }) => {
  const trainingCsv = path.resolve(process.cwd(), "..", "examples", "house_prices.csv");
  const predictionCsv = path.resolve(process.cwd(), "..", "examples", "prediction_houses.csv");

  await page.goto("/datasets/new");
  await page.locator('input[type="file"]').setInputFiles(trainingCsv);
  await expect(page.getByRole("heading", { name: "house_prices.csv" })).toBeVisible();
  await page.getByLabel("Target column").selectOption("price");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("link", { name: "Configure comparison" }).click();

  await expect(page.getByText("Ridge Regression", { exact: true })).toBeVisible();
  await expect(page.getByText("Random Forest Regressor", { exact: true })).toBeVisible();
  await expect(page.getByText("Root mean squared error", { exact: true })).toBeVisible();
  await page.getByLabel("Folds").selectOption("3");
  await page.getByRole("button", { name: "Save configuration" }).click();
  await page.getByRole("button", { name: "Run comparison" }).click();

  await expect(page.getByRole("heading", { name: "Best model" })).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.getByText("Root Mean Squared Error", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Finalize model" }).click();
  await expect(page.getByText("Model finalized", { exact: true })).toBeVisible({
    timeout: 60_000,
  });

  await page.getByRole("link", { name: "View model details and recorded runtime versions" }).click();
  await expect(page.getByText("Regression", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Run prediction" }).click();
  await page.locator('input[type="file"]').setInputFiles(predictionCsv);
  await page.getByRole("button", { name: "Run prediction" }).click();

  await expect(page.getByRole("heading", { name: "Prediction complete" })).toBeVisible();
  await expect(page.getByText("3 rows")).toBeVisible();
});
