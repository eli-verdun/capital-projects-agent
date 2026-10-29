import type { ComponentPropsWithoutRef } from "react";

export function ProjectIcon(props: ComponentPropsWithoutRef<"svg">) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      <path d="M3 21h18" />
      <path d="M6 21V8l4-5v18" />
      <path d="M10 3h9" />
      <path d="M17 3v4" />
      <path d="M15 11h4v10h-4z" />
    </svg>
  );
}
