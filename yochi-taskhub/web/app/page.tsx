"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      router.replace(data.session ? "/my-tasks" : "/login");
    });
  }, [router]);
  return (
    <div className="flex h-screen items-center justify-center text-gray-400">
      Loading…
    </div>
  );
}
