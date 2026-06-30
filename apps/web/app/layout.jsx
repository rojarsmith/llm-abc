import "./globals.css";

export const metadata = {
  title: "LLM ABC Console",
  description: "Minimal Web UI learning console for the LLM ABC API"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
