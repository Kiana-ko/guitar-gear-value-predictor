import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── LOAD ──────────────────────────────────────

df = pd.read_excel('data/guitar_gear_listings.xlsx')
print(f"Loaded {len(df)} rows")

# ── FEATURE ENGINEERING ───────────────────────

# condition has meaningful order so we map manually
condition_map = {'Mint': 5, 'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
df['condition_encoded'] = df['condition'].map(condition_map)

# brand and gear_type have no order so LabelEncoder assigns arbitrary integers
le_brand = LabelEncoder()
le_geartype = LabelEncoder()
df['brand_encoded']     = le_brand.fit_transform(df['brand'])
df['gear_type_encoded'] = le_geartype.fit_transform(df['gear_type'])

print("\nCondition encoding:")
print(df[['condition', 'condition_encoded']].drop_duplicates().sort_values('condition_encoded', ascending=False))

print("\nBrand encoding sample:")
print(df[['brand', 'brand_encoded']].drop_duplicates().sort_values('brand').head(8))

# ── DEFINE FEATURES AND TARGET ────────────────

# X = what the model learns FROM, y = what it predicts
X = df[['brand_encoded', 'gear_type_encoded', 'condition_encoded',
        'age_years', 'original_price']]
y = df['sold_price']

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# ── TRAIN / TEST SPLIT ────────────────────────

# 80% training, 20% testing — random_state makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# ── TRAIN ─────────────────────────────────────

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:25s}: {coef:+.2f}")
print(f"  {'intercept':25s}: {model.intercept_:+.2f}")

# ── EVALUATE ──────────────────────────────────

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\nMAE : ${mae:,.2f}  (average prediction error)")
print(f"R²  : {r2:.3f}  (1.0 = perfect, 0.0 = no better than guessing)")

# ── CHART: ACTUAL vs PREDICTED ────────────────

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=40, label='Predictions')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val],
        color='red', linestyle='--', linewidth=1.5, label='Perfect prediction')

ax.set_xlabel('Actual Sold Price ($)')
ax.set_ylabel('Predicted Sold Price ($)')
ax.set_title(f'Actual vs Predicted\nMAE: ${mae:,.0f}  |  R²: {r2:.3f}', fontweight='bold')
ax.legend()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig('chart5_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

# ── EXAMPLE PREDICTION ────────────────────────

example = pd.DataFrame([{
    'brand_encoded':     le_brand.transform(['Gibson'])[0],
    'gear_type_encoded': le_geartype.transform(['Electric Guitar'])[0],
    'condition_encoded': condition_map['Good'],
    'age_years':         5.0,
    'original_price':    2500.0
}])

predicted = model.predict(example)[0]
print(f"\nExample: 5yr old Gibson Electric Guitar, Good condition, originally $2,500")
print(f"Predicted resale price: ${predicted:,.2f}")

# ── SAVE MODEL ────────────────────────────────

joblib.dump(model,       'gear_model.pkl')
joblib.dump(le_brand,    'encoder_brand.pkl')
joblib.dump(le_geartype, 'encoder_geartype.pkl')

print("\nSaved: gear_model.pkl, encoder_brand.pkl, encoder_geartype.pkl")
print("✅ Done!")