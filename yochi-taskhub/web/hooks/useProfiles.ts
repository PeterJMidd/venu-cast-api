"use client";

import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import type { Profile } from "@/lib/types";

/** All profiles the current user is allowed to see (staff: all; stakeholder: self). */
export function useProfiles() {
  return useQuery({
    queryKey: ["profiles"],
    queryFn: async (): Promise<Profile[]> => {
      const { data, error } = await supabase
        .from("profiles")
        .select("*")
        .eq("active", true)
        .order("full_name");
      if (error) throw error;
      return data as Profile[];
    },
    staleTime: 5 * 60 * 1000,
  });
}

export function profileName(profiles: Profile[] | undefined, id: string | null): string {
  if (!id) return "—";
  const p = profiles?.find((x) => x.id === id);
  return p?.full_name || p?.email || "—";
}
