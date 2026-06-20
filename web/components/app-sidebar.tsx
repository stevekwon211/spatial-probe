"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Boxes, LayoutDashboard, Play } from "lucide-react";

import { PIPELINE, SHIPPED, TOTAL } from "@/lib/pipeline";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";

export function AppSidebar() {
  const pathname = usePathname();
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" render={<Link href="/" />}>
              <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Boxes className="size-4" />
              </div>
              <div className="grid flex-1 text-left text-sm leading-tight">
                <span className="truncate font-semibold">spatial-probe</span>
                <span className="truncate text-xs text-muted-foreground">
                  state, not render
                </span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton isActive={pathname === "/"} tooltip="Overview" render={<Link href="/" />}>
                <LayoutDashboard />
                <span>Overview</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroup>

        {PIPELINE.map((stage, i) => (
          <SidebarGroup key={stage.id}>
            <SidebarGroupLabel>
              {String(i + 1).padStart(2, "0")} · {stage.name}
            </SidebarGroupLabel>
            <SidebarMenu>
              {stage.modules.map((m) => (
                <SidebarMenuItem key={m.id}>
                  {m.href ? (
                    <SidebarMenuButton tooltip={m.oneLine} isActive={pathname === m.href} render={<Link href={m.href} />}>
                      <stage.icon />
                      <span>{m.title}</span>
                    </SidebarMenuButton>
                  ) : (
                    <SidebarMenuButton tooltip={`${m.title} · ${m.statusLabel}`} aria-disabled className="cursor-not-allowed opacity-50">
                      <stage.icon />
                      <span>{m.title}</span>
                    </SidebarMenuButton>
                  )}
                  <SidebarMenuBadge>
                    <span
                      className={
                        "size-2 rounded-full " +
                        (m.status === "in-progress"
                          ? "bg-emerald-500"
                          : m.status === "done"
                            ? "bg-primary"
                            : "bg-muted-foreground/30")
                      }
                    />
                  </SidebarMenuBadge>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter>
        <div className="rounded-lg border bg-sidebar-accent/40 p-3 text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
          <div className="mb-2 font-medium text-foreground">Run full pipeline</div>
          <p className="mb-3 leading-relaxed">
            Chain every stage on a sample scene. Unlocks when all {TOTAL}{" "}
            experiments ship ({SHIPPED}/{TOTAL} done).
          </p>
          <Button size="sm" className="w-full" disabled>
            <Play className="size-3.5" />
            Run pipeline
          </Button>
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
