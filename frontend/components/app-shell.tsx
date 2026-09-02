"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { useRef, useState } from "react";

const primaryNavigation = [
  { label: "Dashboard", href: "/", section: "/" },
  { label: "Datasets", href: "/datasets/new", section: "/datasets" },
  { label: "Experiments", href: "/experiments", section: "/experiments" },
  { label: "Models", href: "/models", section: "/models" },
  { label: "Predictions", href: "/predictions/new", section: "/predictions" },
] as const;

type NavigationProps = Readonly<{
  onNavigate?: () => void;
}>;

function Navigation({ onNavigate }: NavigationProps) {
  const pathname = usePathname();

  return (
    <nav className="navigation" aria-label="Primary navigation">
      <ul className="navigation-list">
        {primaryNavigation.map((item) => (
          <li key={item.label}>
            <Link
              className="navigation-item"
              href={item.href}
              aria-current={
                item.section === "/"
                  ? pathname === "/"
                    ? "page"
                    : undefined
                  : pathname.startsWith(item.section)
                    ? "page"
                    : undefined
              }
              onClick={onNavigate}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>

      <ul className="navigation-list navigation-secondary">
        <li>
          <span className="navigation-item" aria-disabled="true">
            Settings<span className="visually-hidden"> (not available)</span>
          </span>
        </li>
      </ul>
    </nav>
  );
}

function Brand() {
  return (
    <Link className="header-brand" href="/" aria-label="MLForge dashboard">
      <span className="brand-mark" aria-hidden="true">
        M
      </span>
      <span>MLForge</span>
    </Link>
  );
}

type AppShellProps = Readonly<{
  children: ReactNode;
}>;

export function AppShell({ children }: AppShellProps) {
  const menuButton = useRef<HTMLButtonElement>(null);
  const navigationDialog = useRef<HTMLDialogElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const restoreMenuFocus = useRef(true);
  const [navigationOpen, setNavigationOpen] = useState(false);

  function openNavigation() {
    const dialog = navigationDialog.current;
    if (!dialog || dialog.open) return;
    restoreMenuFocus.current = true;
    dialog.showModal();
    setNavigationOpen(true);
    closeButton.current?.focus();
  }

  function closeNavigation(restoreFocus = true) {
    restoreMenuFocus.current = restoreFocus;
    navigationDialog.current?.close();
  }

  return (
    <>
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>

      <header className="app-header">
        <Brand />
        <div className="header-actions">
          <button
            ref={menuButton}
            className="menu-button"
            type="button"
            aria-haspopup="dialog"
            aria-expanded={navigationOpen}
            aria-controls="mobile-navigation"
            onClick={openNavigation}
          >
            Menu
          </button>
          <a
            className="header-link"
            href="https://github.com/HivMindAI/mlforge/tree/main/docs"
            target="_blank"
            rel="noreferrer"
          >
            Docs<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
          <a
            className="header-link"
            href="https://github.com/HivMindAI/mlforge"
            target="_blank"
            rel="noreferrer"
          >
            GitHub<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
        </div>
      </header>

      <div className="app-frame">
        <aside className="sidebar" aria-label="Application sidebar">
          <Navigation />
          <div className="sidebar-meta" aria-label="Application version">
            <span>Local workspace</span>
            <span>Core 0.5.0</span>
          </div>
        </aside>

        <main className="workspace" id="workspace" tabIndex={-1}>
          <div className="workspace-inner">{children}</div>
        </main>
      </div>

      <dialog
        id="mobile-navigation"
        className="mobile-navigation"
        ref={navigationDialog}
        aria-labelledby="mobile-navigation-title"
        aria-modal="true"
        onCancel={(event) => {
          event.preventDefault();
          closeNavigation(true);
        }}
        onClose={() => {
          setNavigationOpen(false);
          if (restoreMenuFocus.current) menuButton.current?.focus();
        }}
        onClick={(event) => {
          if (event.target === navigationDialog.current) {
            closeNavigation(true);
          }
        }}
      >
        <div className="mobile-navigation-panel">
          <div className="mobile-navigation-header">
            <span className="mobile-navigation-title" id="mobile-navigation-title">
              Navigation
            </span>
            <button
              ref={closeButton}
              className="close-button"
              type="button"
              onClick={() => closeNavigation(true)}
            >
              Close
            </button>
          </div>
          <div className="mobile-navigation-body">
            <Navigation onNavigate={() => closeNavigation(false)} />
            <div className="mobile-links">
              <a
                className="header-link"
                href="https://github.com/HivMindAI/mlforge/tree/main/docs"
                target="_blank"
                rel="noreferrer"
              >
                Docs<span className="visually-hidden"> (opens in a new tab)</span>
              </a>
              <a
                className="header-link"
                href="https://github.com/HivMindAI/mlforge"
                target="_blank"
                rel="noreferrer"
              >
                GitHub<span className="visually-hidden"> (opens in a new tab)</span>
              </a>
            </div>
          </div>
        </div>
      </dialog>
    </>
  );
}
