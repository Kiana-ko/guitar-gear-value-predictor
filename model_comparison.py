import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── LOAD ──────────────────────────────────────

df = pd.read_excel('data/guitar_gear_listings.xlsx')

# ── FEATURE ENGINEERING ───────────────────────

condition_map = {'Mint': 5, 'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
df['condition_encoded'] = df['condition'].map(condition_map)

le_brand    = LabelEncoder()
le_geartype = LabelEncoder()
df['brand_encoded']     = le_brand.fit_transform(df['brand'])
df['gear_type_encoded'] = le_geartype.fit_transform(df['gear_type'])

# ── FEATURES AND TARGET ───────────────────────

X = df[['brand_encoded', 'gear_type_encoded', 'condition_encoded',
        'age_years', 'original_price']]
y = df['sold_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── TRAIN BOTH MODELS ─────────────────────────

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_r2   = r2_score(y_test, lr_pred)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)

print(f"Linear Regression — MAE: ${lr_mae:,.2f}  R²: {lr_r2:.3f}")
print(f"Random Forest     — MAE: ${rf_mae:,.2f}  R²: {rf_r2:.3f}")

# ── MODEL COMPARISON CHART ────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
models = ['Linear Regression', 'Random Forest']

axes[0].bar(models, [lr_mae, rf_mae],
            color=['steelblue', '#2ecc71'], edgecolor='white', width=0.5)
axes[0].set_title('MAE — lower is better', fontweight='bold')
axes[0].set_ylabel('Mean Absolute Error ($)')
for i, v in enumerate([lr_mae, rf_mae]):
    axes[0].text(i, v + 1, f'${v:,.0f}', ha='center', fontweight='bold')

axes[1].bar(models, [lr_r2, rf_r2],
            color=['steelblue', '#2ecc71'], edgecolor='white', width=0.5)
axes[1].set_title('R² — higher is better', fontweight='bold')
axes[1].set_ylabel('R² Score')
axes[1].set_ylim(0, 1)
for i, v in enumerate([lr_r2, rf_r2]):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.suptitle('Model Comparison: Linear Regression vs Random Forest',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('chart6_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── FEATURE IMPORTANCE CHART ──────────────────

feature_names = ['Brand', 'Gear Type', 'Condition', 'Age', 'Original Price']
importances   = rf.feature_importances_
sorted_idx    = importances.argsort()[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx] * 100,
    color='#e74c3c', edgecolor='white', width=0.5
)
for bar, val in zip(bars, importances[sorted_idx]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Feature Importance (%)')
ax.set_title('What Factors Drive Resale Price Most?', fontweight='bold')
plt.tight_layout()
plt.savefig('chart7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ── SAVE BEST MODEL ───────────────────────────

if rf_r2 > lr_r2:
    joblib.dump(rf, 'gear_model.pkl')
    print("\nBest model: Random Forest — saved as gear_model.pkl")
else:
    joblib.dump(lr, 'gear_model.pkl')
    print("\nBest model: Linear Regression — gear_model.pkl unchanged")