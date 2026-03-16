import joblib
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── LOAD SAVED MODEL AND ENCODERS ────────────

model       = joblib.load('gear_model.pkl')
le_brand    = joblib.load('encoder_brand.pkl')
le_geartype = joblib.load('encoder_geartype.pkl')

condition_map = {'Mint': 5, 'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}

# ── GET USER INPUT ────────────────────────────

print("=" * 50)
print("GEAR VALUATION TOOL")
print("=" * 50)

brand     = input("Brand (Gibson/Fender/PRS/Martin/Taylor/Ibanez/Marshall/Orange/Squier/Epiphone/Boss/Electro-Harmonix/Fender (Amp)/Seymour Duncan/DiMarzio): ")
gear_type = input("Gear type (Electric Guitar/Acoustic Guitar/Amplifier/Effects Pedal/Pickup): ")
condition = input("Condition (Mint/Excellent/Good/Fair/Poor): ")
age       = float(input("Age in years: "))
price     = float(input("Original price ($): "))

# ── ENCODE AND PREDICT ────────────────────────

features = pd.DataFrame([{
    'brand_encoded':     le_brand.transform([brand])[0],
    'gear_type_encoded': le_geartype.transform([gear_type])[0],
    'condition_encoded': condition_map[condition],
    'age_years':         age,
    'original_price':    price
}])

predicted = model.predict(features)[0]
retention = (predicted / price) * 100

print(f"\nPredicted resale price: ${predicted:,.2f}")
print(f"Estimated retention:    {retention:.1f}%")

# ── CLAUDE API — SMART VALUATION REPORT ──────

client = anthropic.Anthropic()

# EDA findings hardcoded from analysis.py results
brand_retention_data = {
    'PRS': 60.7, 'Gibson': 54.6, 'Martin': 51.9, 'Taylor': 49.3,
    'Fender': 46.2, 'Orange': 44.0, 'Fender (Amp)': 42.3,
    'Marshall': 39.2, 'Ibanez': 36.3, 'Seymour Duncan': 32.8,
    'Electro-Harmonix': 26.2, 'Boss': 24.3, 'Epiphone': 23.3,
    'Squier': 20.9, 'DiMarzio': 19.8
}

brand_avg = brand_retention_data.get(brand, 40.0)
diff      = retention - brand_avg
direction = "above" if diff > 0 else "below"

# ── PROMPT 1: SMART VALUATION WITH EDA REASONING ──

prompt_valuation = f"""You are an expert guitar gear appraiser with access to market data.

EDA Market Data:
- {brand}'s average resale retention across all listings: {brand_avg:.1f}%
- This specific listing's predicted retention: {retention:.1f}%
- Difference: {abs(diff):.1f}% {direction} the brand average

Gear details:
- Brand: {brand} | Type: {gear_type} | Condition: {condition}
- Age: {age} years | Original price: ${price:,.2f}
- Predicted resale price: ${predicted:,.2f}

Write a 3-paragraph valuation report:
Paragraph 1: Explain why this listing is {abs(diff):.1f}% {direction} {brand}'s average retention of {brand_avg:.1f}%. Be specific about which factors are driving this deviation.
Paragraph 2: Assess whether ${predicted:,.2f} is a fair asking price given the brand's market position.
Paragraph 3: Give 2-3 concrete tips to maximize sale price for this specific listing."""

print("\n" + "=" * 50)
print("AI VALUATION REPORT")
print("=" * 50)

response1 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt_valuation}]
)
print(response1.content[0].text)

# ── PROMPT 2: MARKET TREND ANALYSIS ───────────

prompt_trends = f"""You are a guitar gear market analyst. Here is brand retention data from a resale price dataset:

{chr(10).join([f"- {b}: {r:.1f}% retention" for b, r in sorted(brand_retention_data.items(), key=lambda x: -x[1])])}

Write a brief 2-paragraph market trend analysis:
Paragraph 1: What patterns do you see in which brands hold value best vs worst? What does this tell us about the guitar gear resale market?
Paragraph 2: For someone buying gear today with resale value in mind, what are your top 2-3 recommendations based on this data?"""

print("\n" + "=" * 50)
print("MARKET TREND ANALYSIS")
print("=" * 50)

response2 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=400,
    messages=[{"role": "user", "content": prompt_trends}]
)
print(response2.content[0].text)
print("\n✅ Done!")