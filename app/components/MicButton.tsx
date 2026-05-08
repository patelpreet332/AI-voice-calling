"use client";

import { motion } from "framer-motion";

export default function MicButton({ recording, onStart, onStop }: any) {
  return (
    <motion.button
      onClick={recording ? onStop : onStart}
      animate={{ scale: recording ? 1.3 : 1 }}
      className={`mt-6 p-6 rounded-full ${
        recording ? "bg-red-500" : "bg-blue-500"
      }`}
    >
      🎤
    </motion.button>
  );
}