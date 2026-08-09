"use client";

import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { useProfile } from "@/hooks/useProfile";
import DrawerHost from "@/components/DrawerHost";
import NLQuickAdd from "@/components/NLQuickAdd";
import NewProjectModal from "@/components/NewProjectModal";
import PrefsModal from "@/components/PrefsModal";
import VoicePanel from "@/components/VoicePanel";
import type { Category, Project } from "@/lib/types";

const NAV = [
  { href: "/my-tasks", label: "My tasks" },
  { href: "/tasks", label: "All tasks" },
  { href: "/calendar", label: "Calendar" },
  { href: "/reporting", label: "Reporting" },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [ready, setReady] = useState(false);
  const [showNewProject, setShowNewProject] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showPrefs, setShowPrefs] = useState(false);
  const { data: profile } = useProfile();

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      if (!data.session) router.replace("/login");
      else setReady(true);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((event) => {
      if (event === "SIGNED_OUT") router.replace("/login");
    });
    return () => sub.subscription.unsubscribe();
  }, [router]);

  const { data: categories } = useQuery({
    queryKey: ["categories"],
    enabled: ready,
    queryFn: async () => {
      const { data, error } = await supabase.from("categories").select("*").order("sort");
      if (error) throw error;
      return data as Category[];
    },
    staleTime: Infinity,
  });

  const { data: projects } = useQuery({
    queryKey: ["projects"],
    enabled: ready,
    queryFn: async () => {
      const { data, error } = await supabase
        .from("projects")
        .select("*")
        .eq("archived", false)
        .order("name");
      if (error) throw error;
      return data as Project[];
    },
  });

  const isStaff = profile?.role === "admin" || profile?.role === "finance";

  if (!ready) {
    return (
      <div className="flex h-screen items-center justify-center text-gray-400">Loading…</div>
    );
  }

  return (
    <div className="flex h-screen">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-30 bg-black/30 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}
      {/* Sidebar */}
      <aside
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("a")) setSidebarOpen(false);
        }}
        className={`flex w-64 shrink-0 flex-col border-r border-gray-200 bg-white max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:z-40 max-md:shadow-2xl max-md:transition-transform ${
          sidebarOpen ? "" : "max-md:-translate-x-full"
        }`}
      >
        <div className="border-b border-gray-100 px-4 py-4">
          <Link href="/my-tasks" className="text-lg font-bold text-brand-600">
            Yo-Chi TaskHub
          </Link>
        </div>
        <nav className="flex-1 overflow-y-auto px-2 py-3">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={`block rounded-lg px-3 py-1.5 text-sm ${
                pathname === n.href
                  ? "bg-brand-50 font-semibold text-brand-700"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              {n.label}
            </Link>
          ))}
          {isStaff && (
            <>
              <Link
                href="/home"
                className={`block rounded-lg px-3 py-1.5 text-sm ${
                  pathname === "/home"
                    ? "bg-brand-50 font-semibold text-brand-700"
                    : "text-gray-700 hover:bg-gray-100"
                }`}
              >
                Home
              </Link>
              <Link
                href="/lake"
                className={`block rounded-lg px-3 py-1.5 text-sm ${
                  pathname === "/lake"
                    ? "bg-brand-50 font-semibold text-brand-700"
                    : "text-gray-700 hover:bg-gray-100"
                }`}
              >
                Data lake
              </Link>
              <Link
                href="/templates"
                className={`block rounded-lg px-3 py-1.5 text-sm ${
                  pathname === "/templates"
                    ? "bg-brand-50 font-semibold text-brand-700"
                    : "text-gray-700 hover:bg-gray-100"
                }`}
              >
                Templates
              </Link>
              {profile?.role === "admin" && (
                <Link
                  href="/admin"
                  className={`block rounded-lg px-3 py-1.5 text-sm ${
                    pathname === "/admin"
                      ? "bg-brand-50 font-semibold text-brand-700"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  Admin
                </Link>
              )}
            </>
          )}

          <div className="mt-5 space-y-4">
            {isStaff && (
              <button
                onClick={() => setShowNewProject(true)}
                className="mx-3 rounded-lg border border-dashed border-gray-300 px-3 py-1 text-xs text-gray-500 hover:border-brand-500 hover:text-brand-600"
              >
                + New project
              </button>
            )}
            {categories?.map((cat) => {
              const catProjects = projects?.filter((p) => p.category_id === cat.id) ?? [];
              return (
                <div key={cat.id}>
                  <div className="px-3 text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                    {cat.name}
                  </div>
                  {catProjects.length === 0 && (
                    <div className="px-3 py-1 text-xs text-gray-300">No projects</div>
                  )}
                  {catProjects.map((p) => (
                    <Link
                      key={p.id}
                      href={`/board?project=${p.id}`}
                      className="block truncate rounded-lg px-3 py-1 text-sm text-gray-600 hover:bg-gray-100"
                    >
                      {p.name}
                    </Link>
                  ))}
                </div>
              );
            })}
          </div>
        </nav>
        <div className="border-t border-gray-100 px-4 py-3">
          <div className="truncate text-sm font-medium">{profile?.full_name ?? profile?.email}</div>
          <div className="flex items-center justify-between">
            <span className="text-xs capitalize text-gray-400">{profile?.role}</span>
            <div className="flex gap-3">
              <button onClick={() => setShowPrefs(true)} className="text-xs text-gray-400 hover:text-brand-600" title="Notification settings">⚙</button>
              <button
                onClick={() => supabase.auth.signOut()}
                className="text-xs text-gray-400 hover:text-red-600"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {/* Mobile top bar */}
        <div className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-2.5 md:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
            className="rounded-lg border border-gray-300 px-2.5 py-1 text-lg leading-none"
          >
            ☰
          </button>
          <span className="text-base font-bold text-brand-600">Yo-Chi TaskHub</span>
        </div>
        {isStaff && (
          <div className="border-b border-gray-200 bg-white px-4 py-2.5 md:px-6">
            <Suspense>
              <NLQuickAdd />
            </Suspense>
          </div>
        )}
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>

      <Suspense>
        <DrawerHost />
      </Suspense>
      {profile && profile.role !== "stakeholder" && <VoicePanel />}
      {showNewProject && <NewProjectModal onClose={() => setShowNewProject(false)} />}
      {showPrefs && profile && <PrefsModal userId={profile.id} onClose={() => setShowPrefs(false)} />}
    </div>
  );
}
