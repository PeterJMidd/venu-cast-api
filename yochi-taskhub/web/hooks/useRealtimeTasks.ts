"use client";

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";

/** Live board: invalidate task queries whenever taskapp.tasks changes. */
export function useRealtimeTasks() {
  const qc = useQueryClient();
  useEffect(() => {
    const channel = supabase
      .channel("taskapp-tasks")
      .on(
        "postgres_changes",
        { event: "*", schema: "taskapp", table: "tasks" },
        () => {
          qc.invalidateQueries({ queryKey: ["tasks"] });
          qc.invalidateQueries({ queryKey: ["my-tasks"] });
          qc.invalidateQueries({ queryKey: ["task"] });
        }
      )
      .subscribe();
    return () => {
      supabase.removeChannel(channel);
    };
  }, [qc]);
}
