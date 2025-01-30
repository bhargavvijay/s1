import express from "express";
import dotenv from "dotenv";
import { HfInference } from "@huggingface/inference";

dotenv.config();
const app = express();
const PORT = process.env.PORT || 5000;
const hf = new HfInference(process.env.HF_API_KEY);

// Middleware to parse JSON
app.use(express.json());

// Summarization Route
app.post("/summarize", async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    const response = await hf.summarization({
      model: "facebook/bart-large-cnn",
      inputs: text,
    });

    res.json({ summary: response.summary_text });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// Start Server
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
