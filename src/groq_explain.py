
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

os.makedirs("explanations", exist_ok=True)

MODEL = "llama-3.3-70b-versatile"

FEATURE_UNITS = {
    "Age": "years",
    "Income": "annual income (currency units)",
    "LoanAmount": "loan amount requested (currency units)",
    "CreditScore": "credit score (300-850 typical range)",
    "MonthsEmployed": "months at current job",
    "NumCreditLines": "number of open credit lines",
    "InterestRate": "interest rate (%)",
    "LoanTerm": "loan term (months)",
    "DTIRatio": "debt-to-income ratio",
    "Education": "education level",
    "EmploymentType": "employment type",
    "MaritalStatus": "marital status",
    "HasMortgage": "has an existing mortgage",
    "HasDependents": "has dependents",
    "LoanPurpose": "stated purpose of the loan",
    "HasCoSigner": "has a co-signer",
}

SYSTEM_PROMPTS = {
    "customer": (
        "You are a helpful loan officer assistant explaining a lending decision "
        "to the loan applicant directly. Be warm, clear, and non-technical. "
        "Never mention 'SHAP', 'model', 'features', or any ML jargon. "
        "Speak in plain, respectful language. Keep it to 3-4 sentences. "
        "If the applicant was flagged as high risk, explain the top 2-3 real-world "
        "reasons in a way that helps them understand what would improve their chances "
        "Speak in plain, respectful language. Keep it to 3-4 sentences. "
        "If the applicant was flagged as high risk, explain the top 2-3 real-world reasons... "
        "If the applicant has a low predicted default probability... "
    ),
    "analyst": (
        "You are writing for a professional bank risk analyst.\n"
        "Do NOT write as if speaking to the applicant.\n"
        "Respond using exactly this format:\n\n"
        "Verdict:\n"
        "Risk Drivers:\n"
        "Recommendation:\n\n"
        "Use the applicant's actual feature values and indicate whether each factor increased or decreased the predicted risk for THIS applicant only. "
        "Do not infer general lending rules. "
        "Keep the response under 120 words."
    ),
}

def build_user_prompt(explanation: dict) -> str:
    proba = explanation["predicted_default_probability"]
    contributions = explanation["top_contributions"]

    lines = [
        
        f"Predicted default probability: {proba:.1%}",
        f"Base (average) default rate: {explanation['base_value']:.1%}",
        "",
        "IMPORTANT:",
        "- Explain only this applicant.",
        "- Do NOT assume that higher or lower values are always better or worse.",
        "- Do NOT invent lending rules.",
        "- Use the SHAP impacts only to explain this prediction.",
        "",
        "Factors for this applicant:",
    ]
    
    for c in contributions:
        unit = FEATURE_UNITS.get(c["feature"], "")
        direction = "increases risk" if c["shap_impact"] > 0 else "decreases risk"
        lines.append(
            f"- {c['feature']} = {c['value']} ({unit}): SHAP impact {c['shap_impact']:+.3f} "
            f"({direction})"
        )

    return "\n".join(lines)


def explain_with_llm(explanation: dict, audience: str = "customer") -> str:
    assert audience in SYSTEM_PROMPTS, "audience must be 'customer' or 'analyst'"
    
    try:
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[audience]},
                {"role": "user", "content": build_user_prompt(explanation)},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during Groq API call: {e}")
        return "Explanation could not be generated."

if __name__ == "__main__":
    with open("outputs/shap_examples.json") as f:
        examples = json.load(f)

    # Run both audiences on the first example (a defaulted case) as a demo
    example = examples[0]
    print("=" * 60)
    print("RAW SHAP DATA")
    print("=" * 60)
    print(json.dumps(example, indent=2))

    print("\n" + "=" * 60)
    print("CUSTOMER-FACING EXPLANATION")
    print("=" * 60)

    customer_text = explain_with_llm(
        example,
        audience="customer"
    )

    print(customer_text)

    print("\n" + "=" * 60)
    print("ANALYST-FACING EXPLANATION")
    print("=" * 60)

    analyst_text = explain_with_llm(
        example,
        audience="analyst"
    )

    print(analyst_text)

    with open("explanations/customer_explanation.txt", "w") as f:
        f.write(customer_text)

    with open("explanations/analyst_explanation.txt", "w") as f:
        f.write(analyst_text)

    print("\nSaved:")
    print("- explanations/customer_explanation.txt")
    print("- explanations/analyst_explanation.txt")