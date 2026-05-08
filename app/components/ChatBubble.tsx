export default function ChatBubble({ text, type }: any) {
  if (!text) return null;

  return (
    <div
      className={`mt-4 p-4 rounded-xl max-w-xl ${
        type === "user" ? "bg-gray-700" : "bg-green-600"
      }`}
    >
      {text}
    </div>
  );
}