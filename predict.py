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

# ── CLAUDE API — VALUATION REPORT ────────────

# load_dotenv() already loaded the key from .env
# anthropic.Anthropic() automatically reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

prompt = f"""You are an expert guitar gear appraiser. Based on the following data, write a 
short 3-paragraph valuation report for a seller. Be specific, practical, and helpful.

Gear details:
- Brand: {brand}
- Type: {gear_type}
- Condition: {condition}
- Age: {age} years
- Original price: ${price:,.2f}
- Predicted resale price: ${predicted:,.2f}
- Value retention: {retention:.1f}%

Paragraph 1: Assess the predicted resale value and what it means for this specific brand and gear type.
Paragraph 2: Explain what factors (condition, age, brand reputation) are most influencing this valuation.
Paragraph 3: Give 2-3 practical tips for maximizing sale price."""

print("\n" + "=" * 50)
print("AI VALUATION REPORT")
print("=" * 50)

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}]
)

print(message.content[0].text)
print("\n✅ Done!")