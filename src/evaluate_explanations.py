"""
LLM Explainability Evaluation

This script evaluates the quality of the explanations generated
by the Groq LLM using a separate LLM-as-a-Judge approach.

Workflow:
1. Load the evaluation set generated from SHAP explanations.
2. Generate explanations for each applicant.
3. Ask a separate LLM to score each explanation.
4. Aggregate the scores.
5. Save the evaluation results.

Evaluation Metrics:
- Faithfulness
- Direction Correctness
- Clarity
- Consistency

Outputs:
- outputs/explainability_evaluation.json
"""
import os
import json
import time

from dotenv import load_dotenv
from groq import Groq

from groq_explain import explain_with_llm

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)


JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator grading an AI-generated loan-risk
explanation. You will be given:
1. The raw SHAP data (ground truth: which factors actually drove the prediction, and
   in which direction)
2. The AI-generated explanation that was supposed to summarize that data

Score the explanation on FOUR dimensions, each from 0.0 to 1.0:

- faithfulness: Does the explanation ONLY reference factors that are actually present
  in the SHAP data? Penalize any invented/hallucinated reasons not in the data.
- direction_correctness: For each factor mentioned, does the explanation correctly state
  whether it increases or decreases risk, matching the SHAP sign (positive = increases
  risk, negative = decreases risk)?
- clarity: Is the explanation easy to read, well-structured, and appropriate for its
  stated audience (plain language for customers, precise/technical for analysts)?
- consistency: Is the explanation free of internal contradictions (e.g. never says a
  factor is both a strength and a concern without clear reconciliation)?

Respond ONLY with valid JSON, no markdown formatting, no preamble:
{
  "faithfulness": <float>,
  "direction_correctness": <float>,
  "clarity": <float>,
  "consistency": <float>,
  "issues_found": "<brief one-sentence note on any problems, or 'none'>"
}"""


def judge_explanation(shap_data: dict, explanation_text: str, audience: str) -> dict:
    user_prompt = f"""AUDIENCE: {audience}

RAW SHAP DATA (ground truth):
{json.dumps(shap_data, indent=2)}

GENERATED EXPLANATION TO EVALUATE:
{explanation_text}"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


if __name__ == "__main__":
    with open("outputs/eval_set.json") as f:
        eval_set = json.load(f)

    all_scores = []

    for i, example in enumerate(eval_set):
        for audience in ["customer", "analyst"]:
            
            progress = ((i + 1) / len(eval_set)) * 100

            print(
                f"[{i+1}/{len(eval_set)} | {progress:.0f}%] "
                f"Generating + judging ({audience})..."
            )
            try:
                
                explanation = explain_with_llm(example, audience=audience)
                time.sleep(0.5)  # be gentle on rate limits
                scores = judge_explanation(example, explanation, audience)
                time.sleep(0.5)

                scores["row_index"] = example["row_index"]
                scores["audience"] = audience
                scores["predicted_probability"] = example["predicted_default_probability"]
                scores["generated_explanation"] = explanation
                all_scores.append(scores)
            except Exception as e:
                print(f"Error on row {example['row_index']} ({audience}): {e}")

    # ---------- Aggregate ----------
    def avg(key, subset=None):
        rows = subset if subset is not None else all_scores
        if not rows:
            return 0.0
        return sum(r[key] for r in rows) / len(rows)

    customer_scores = [s for s in all_scores if s["audience"] == "customer"]
    analyst_scores = [s for s in all_scores if s["audience"] == "analyst"]

    summary = {
        "n_examples": len(eval_set),
        "n_evaluations": len(all_scores),
        "overall": {
            "faithfulness": round(avg("faithfulness"), 3),
            "direction_correctness": round(avg("direction_correctness"), 3),
            "clarity": round(avg("clarity"), 3),
            "consistency": round(avg("consistency"), 3),
        },
        "customer_audience": {
            "faithfulness": round(avg("faithfulness", customer_scores), 3),
            "direction_correctness": round(avg("direction_correctness", customer_scores), 3),
            "clarity": round(avg("clarity", customer_scores), 3),
            "consistency": round(avg("consistency", customer_scores), 3),
        },
        "analyst_audience": {
            "faithfulness": round(avg("faithfulness", analyst_scores), 3),
            "direction_correctness": round(avg("direction_correctness", analyst_scores), 3),
            "clarity": round(avg("clarity", analyst_scores), 3),
            "consistency": round(avg("consistency", analyst_scores), 3),
        },
        "issues_flagged": [
            {"row_index": s["row_index"], "audience": s["audience"], "issue": s["issues_found"]}
            for s in all_scores if s["issues_found"].lower() != "none"
        ],
        "raw_scores": all_scores,
    }

    with open("outputs/explainability_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPLAINABILITY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Evaluated {summary['n_evaluations']} explanations "
          f"({summary['n_examples']} cases x 2 audiences)")
    
    print("\nOverall Scores:")
    for k, v in summary["overall"].items():
        print(f"  {k}: {v}")

    print("\nCustomer Scores:")
    for k, v in summary["customer_audience"].items():
        print(f"  {k}: {v}")

    print("\nAnalyst Scores:")
    for k, v in summary["analyst_audience"].items():
        print(f"  {k}: {v}")
        
    for k, v in summary["overall"].items():
        print(f"  {k}: {v}")
    print(f"\nIssues flagged: {len(summary['issues_flagged'])}")
    for issue in summary["issues_flagged"][:5]:
        print(f"  - row {issue['row_index']} ({issue['audience']}): {issue['issue']}")

    print("\nSaved: outputs/explainability_evaluation.json")