import type { NextConfig } from "next";

// Fully static site: every page is pre-rendered at build time and all data
// (replays, tournament results, training curves) is pre-computed JSON.
// Deploys to Vercel (Root Directory = `web`) or any static host.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
