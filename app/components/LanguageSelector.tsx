export default function LanguageSelector({ value, onChange }: any) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="text-black p-2 rounded mb-4"
    >
      <option>English</option>
      <option>Hindi</option>
      <option>Marathi</option>
    </select>
  );
}