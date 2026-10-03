import { ShieldCheck } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border py-10">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 text-sm text-foreground-muted sm:flex-row">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-foreground text-background">
            <ShieldCheck className="h-3.5 w-3.5" />
          </span>
          <span className="font-medium text-foreground">ForgeLens</span>
        </div>
        <p>A reference implementation — not a substitute for expert forensic review.</p>
      </div>
    </footer>
  );
}
