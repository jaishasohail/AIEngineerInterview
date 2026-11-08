/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#142233",
        pine: "#1F6B5A",
        copper: "#9C5B3C",
        sand: "#F3EEE6",
        paper: "#FFFCF8",
        line: "#E5DCD0",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "Segoe UI", "sans-serif"],
        serif: ["var(--font-serif)", "Georgia", "serif"],
      },
      boxShadow: {
        card: "0 1px 0 rgba(20,34,51,0.04), 0 16px 40px rgba(20,34,51,0.05)",
      },
    },
  },
  plugins: [],
};
