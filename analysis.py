import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_excel('data/guitar_gear_listings.xlsx')

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("COLUMN TYPES")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())

# For retention calculation responsible for normalizing sold price as % of original
# so comparisons across brands with different price points are fair
df['retention_pct'] = (df['sold_price'] / df['original_price']) * 100

# ── CHART 1: BRAND RETENTION ──────────────────

brand_retention = (
    df.groupby('brand')['retention_pct']
    .mean()
    .sort_values(ascending=False)
    .round(1)
)
print("\nBRAND RETENTION %")
print(brand_retention)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(brand_retention.index, brand_retention.values,
               color='steelblue', edgecolor='white')

for bar, val in zip(bars, brand_retention.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=9)

ax.set_xlabel('Average Retention %')
ax.set_title('Brand Value Retention — Which Brands Hold Their Value Best?',
             fontweight='bold')
ax.axvline(x=brand_retention.mean(), color='red', linestyle='--', alpha=0.6,
           label=f'Average: {brand_retention.mean():.1f}%')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('chart1_brand_retention.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CHART 2: CONDITION vs RETENTION ───────────

# reindex forces our custom order instead of alphabetical
condition_order = ['Mint', 'Excellent', 'Good', 'Fair', 'Poor']
condition_stats = (
    df.groupby('condition')['retention_pct']
    .agg(['mean', 'std'])
    .reindex(condition_order)
    .round(1)
)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
bars = ax.bar(condition_stats.index, condition_stats['mean'],
              color=colors, edgecolor='white', width=0.6,
              yerr=condition_stats['std'], capsize=5)

for bar, val in zip(bars, condition_stats['mean']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Average Retention %')
ax.set_title('How Condition Affects Resale Value', fontweight='bold')
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig('chart2_condition_retention.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CHART 3: AGE vs SOLD PRICE ────────────────

fig, ax = plt.subplots(figsize=(10, 6))
colors_map = {
    'Electric Guitar': '#3498db',
    'Acoustic Guitar': '#2ecc71',
    'Amplifier':       '#e74c3c',
    'Effects Pedal':   '#f39c12',
    'Pickup':          '#9b59b6'
}

for gear in df['gear_type'].unique():
    subset = df[df['gear_type'] == gear]
    ax.scatter(subset['age_years'], subset['sold_price'],
               alpha=0.5, label=gear,
               color=colors_map.get(gear, 'gray'), s=40)

ax.set_xlabel('Age (years)')
ax.set_ylabel('Sold Price ($)')
ax.set_title('Age vs Sold Price by Gear Type', fontweight='bold')

# For formatting responsible for displaying y-axis as currency ($1,500 not 1500)
# using a lambda via FuncFormatter
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.savefig('chart3_age_vs_price.png', dpi=150, bbox_inches='tight')
plt.show()

# ── CHART 4: GEAR TYPE RETENTION ─────────────

gear_retention = (
    df.groupby('gear_type')['retention_pct']
    .mean()
    .sort_values(ascending=False)
    .round(1)
)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(gear_retention.index, gear_retention.values,
              color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
              edgecolor='white', width=0.6)

for bar, val in zip(bars, gear_retention.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Average Retention %')
ax.set_title('Value Retention by Gear Category', fontweight='bold')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('chart4_gear_type_retention.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n EDA complete. 4 charts saved. Now run model.py")