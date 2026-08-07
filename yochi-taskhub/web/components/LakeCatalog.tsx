"use client";

import { useState } from "react";
import Link from "next/link";
import CatalogList from "@/components/CatalogList";

export default function LakeCatalog({ onClose }: { onClose: () => void }) {
  const [search, setSearch] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={onClose}>
      <div
        className="flex h-[85vh] w-full max-w-2xl flex-col rounded-xl bg-white shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-gray-200 px-5 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold">Data lake catalog</h2>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">✕</button>
          </div>
          <p className="mt-1 text-xs text-gray-500">
            Reference tables by bare name in SQL; cast text dates with{" "}
            <code className="rounded bg-gray-100 px-1">CAST(col AS DATE)</code>.{" "}
            <Link href="/lake" onClick={onClose} className="font-semibold text-brand-600 hover:underline">
              Open the full Data lake page →
            </Link>
          </p>
          <input
            autoFocus
            placeholder="Search tables or columns…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
          />
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-3">
          <CatalogList search={search} />
        </div>
      </div>
    </div>
  );
}
