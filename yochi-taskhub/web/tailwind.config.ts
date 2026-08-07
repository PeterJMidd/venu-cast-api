import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0faf4",
          100: "#dbf2e4",
          500: "#2e9e5b",
          600: "#23824a",
          700: "#1d683d",
        },
      },
    },
  },
  plugins: [],
};
export default config;
