"use client";

import Link from "next/link";

type PageLoadingStateProps = Readonly<{
  kicker: string;
  title: string;
  description: string;
}>;

export function PageLoadingState({ kicker, title, description }: PageLoadingStateProps) {
  return (
    <div className="overview-status page-loading-state" role="status" aria-live="polite" aria-busy="true">
      <span className="section-kicker">{kicker}</span>
      <h1>{title}</h1>
      <p>{description}</p>
      <div className="loading-lines" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

type PageErrorStateProps = Readonly<{
  kicker: string;
  title: string;
  description: string;
  onRetry: () => void;
  secondaryHref: string;
  secondaryLabel: string;
}>;

export function PageErrorState({
  kicker,
  title,
  description,
  onRetry,
  secondaryHref,
  secondaryLabel,
}: PageErrorStateProps) {
  return (
    <div className="overview-status page-error-state" role="alert">
      <span className="section-kicker">{kicker}</span>
      <h1>{title}</h1>
      <p>{description}</p>
      <div className="status-actions">
        <button className="primary-button" type="button" onClick={onRetry}>
          Try again
        </button>
        <Link className="secondary-button secondary-link" href={secondaryHref}>
          {secondaryLabel}
        </Link>
      </div>
    </div>
  );
}

type EmptyStateProps = Readonly<{
  title: string;
  description: string;
  actionHref: string;
  actionLabel: string;
  headingLevel?: 2 | 3;
  className?: string;
}>;

export function EmptyState({
  title,
  description,
  actionHref,
  actionLabel,
  headingLevel = 3,
  className = "",
}: EmptyStateProps) {
  const Heading = headingLevel === 2 ? "h2" : "h3";
  return (
    <div className={`empty-state${className ? ` ${className}` : ""}`}>
      <Heading>{title}</Heading>
      <p>{description}</p>
      <Link className="primary-button" href={actionHref}>
        {actionLabel}
      </Link>
    </div>
  );
}
