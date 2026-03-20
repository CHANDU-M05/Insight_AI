# InsightAI — AI-Powered Data Analysis

Upload any CSV or Excel file, ask questions in plain English, get pandas code + charts + insights. Runs 100% locally — your data never leaves your machine.

![Python](https://img.shields.io/badge/Python-3.12+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![Plotly](https://img.shields.io/badge/Plotly-5.0+-green)

---

## What makes it different

Every AI data tool either costs $20/month (Julius AI) or requires coding skills (PandasAI). InsightAI is free, open source, runs locally, and shows you the generated code so you can learn from it — not just get answers.

---

## Features

- **Auto data profiling** — on upload, instantly shows nulls, distributions, correlations, warnings. No question needed.
- **Code transparency** — every answer shows the pandas code that generated it. Copy it, learn from it, modify it.
- **Charts in UI** — plotly charts render inline. Not in terminal.
- **Multi-turn memory** — follow-up questions work naturally. "Compare that to last month" understands context.
- **Multi-LLM** — Gemini (free) or OpenAI GPT-4o
- **PDF report export** — full session packaged as downloadable PDF
- **Sample datasets** — built-in Titanic and Sales datasets. Zero setup to demo.
- **Local first** — your data never leaves your machine

---

## Project structure
```
InsightAI/
├── app.py                  # Streamlit UI — Data, Analyze, Report tabs
├── core/
│   ├── data_loader.py      # CSV/Excel → DataFrame + auto profiler
│   ├── analyst.py          # NL → pandas code → execute → chart
│   └── reporter.py         # Session → PDF report
├── samples/
│   ├── sales.csv           # Built-in sample
│   └── titanic.csv         # Built-in sample
├── requirements.txt
└── .env.example
```

---

## Setup
```bash
git clone https://github.com/CHANDU-M05/Insight_AI
cd Insight_AI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GEMINI_API_KEY or OPENAI_API_KEY to .env
streamlit run app.py
```

---

## Engineering decisions

**Why show the generated code?**
Data scientists trust tools they can inspect. Non-technical users learn from seeing the code. This is what separates InsightAI from Julius AI — transparency is free here.

**Why auto-profile on upload?**
Every analyst does this first manually. InsightAI does it automatically — the analysis starts before the user types anything. Thats the moment that makes someone say "wow."

**Why local-first?**
Enterprises with sensitive finance, HR, and healthcare data cannot use cloud tools. Running locally means zero data privacy concerns — a feature every enterprise buyer asks about first.

**Why kill DuckDB?**
The original used DuckDB which adds complexity with zero user-visible benefit for datasets under 1GB. Pandas handles everything a real-world demo needs. Fewer dependencies = fewer failures.

---

## Built by

Chandu — [GitHub](https://github.com/CHANDU-M05)
