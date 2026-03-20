"""
InsightAI Analyst
Natural language → pandas code → execute → result + chart.
WHY code transparency: data scientists trust tools they can inspect.
Non-technical users learn from seeing the code.
This is the feature nobody else shows for free.
"""

import io
import sys
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Optional


SYSTEM_PROMPT = """You are an expert data analyst. You have access to a pandas DataFrame called `df`.

Rules:
- Always write clean, executable Python code using pandas and matplotlib/plotly
- For visualizations use plotly express (px) — assign to variable called `fig`
- For text answers assign result to variable called `result`
- For tables assign a DataFrame to variable called `result`
- Never use plt.show() — just create the figure
- Always handle missing values gracefully
- Write concise, readable code
- Add a one-line comment explaining what the code does

Available variables:
- df: the pandas DataFrame
- pd, np, px: already imported

Respond in this exact format:
```python
# [one line explanation]
[your code here]
```
Then on a new line after the code block, write a 1-2 sentence plain English explanation of the result."""


def build_prompt(question: str, df: pd.DataFrame, history: list[dict]) -> str:
    """Build prompt with dataset context and conversation history."""

    # Dataset context
    context = f"""Dataset info:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Dtypes: {df.dtypes.to_dict()}
- Sample (first 3 rows):
{df.head(3).to_string()}
"""

    # Recent history (last 3 exchanges)
    history_text = ""
    if history:
        recent = history[-3:]
        history_text = "\nRecent conversation:\n"
        for h in recent:
            history_text += f"Q: {h['question']}\nA: {h['answer_text']}\n\n"

    return f"{context}\n{history_text}\nQuestion: {question}"


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        return response[start:end].strip()
    return response.strip()


def extract_explanation(response: str) -> str:
    """Extract plain English explanation after code block."""
    if "```" in response:
        last_fence = response.rfind("```")
        after = response[last_fence + 3:].strip()
        return after if after else ""
    return ""


def execute_code(code: str, df: pd.DataFrame) -> dict:
    """
    Execute generated pandas code safely.
    Returns dict with: result, fig, error, stdout.
    WHY exec: necessary for dynamic code execution. Risk is
    mitigated by running locally — no server, no injection risk
    from external users.
    """
    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "plt": plt,
        "result": None,
        "fig": None,
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        exec(code, local_vars)
        stdout = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return {
            "result": local_vars.get("result"),
            "fig": local_vars.get("fig"),
            "stdout": stdout,
            "error": None,
        }
    except Exception as e:
        sys.stdout = old_stdout
        return {
            "result": None,
            "fig": None,
            "stdout": "",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def ask(
    question: str,
    df: pd.DataFrame,
    history: list[dict],
    api_key: str,
    provider: str = "gemini",
) -> dict:
    """
    Main entry point.
    Takes a question + DataFrame + history.
    Returns: code, explanation, result, fig, error.
    """
    prompt = build_prompt(question, df, history)

    # Get LLM response
    raw_response = ""
    try:
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(f"{SYSTEM_PROMPT}\n\n{prompt}")
            raw_response = response.text

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            raw_response = response.choices[0].message.content

    except Exception as e:
        return {
            "code": "",
            "explanation": "",
            "result": None,
            "fig": None,
            "error": f"LLM call failed: {e}",
            "raw_response": "",
        }

    # Extract and execute code
    code = extract_code(raw_response)
    explanation = extract_explanation(raw_response)
    execution = execute_code(code, df)

    return {
        "code": code,
        "explanation": explanation,
        "result": execution["result"],
        "fig": execution["fig"],
        "stdout": execution["stdout"],
        "error": execution["error"],
        "raw_response": raw_response,
    }
