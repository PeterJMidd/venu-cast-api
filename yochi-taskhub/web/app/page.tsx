"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    supabase.auth.getSession().then(async ({ data }) => {
      if (!data.session) {
        router.replace("/login");
        return;
      }
      const { data: prof } = await supabase
        .from("profiles").select("role").eq("id", data.session.user.id).single();
      const role = (prof as { role?: string } | null)?.role;
      router.replace(role === "admin" || role === "finance" ? "/home" : "/my-tasks");
    });
  }, [router]);
  return (
    <div className="flex h-screen items-center justify-center text-gray-400">
      Loading…
    </div>
  );
}
