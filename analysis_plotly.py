import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_excel('data/guitar_gear_listings.xlsx')
df['retention_pct'] = (df['sold_price'] / df['original_price']) * 100

# ── CHART 1: BRAND RETENTION ──────────────────

brand_retention = (
    df.groupby('brand')['retention_pct']
    .mean()
    .sort_values(ascending=True)
    .round(1)
    .reset_index()
)
brand_retention.columns = ['brand', 'retention_pct']

fig1 = px.bar(brand_retention, x='retention_pct', y='brand',
              orientation='h',
              title='Brand Value Retention — Which Brands Hold Their Value Best?',
              labels={'retention_pct': 'Average Retention %', 'brand': 'Brand'},
              color='retention_pct',
              color_continuous_scale='Blues',
              text='retention_pct')
fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig1.update_layout(coloraxis_showscale=False, height=500)
fig1.write_html('chart1_brand_retention.html')
fig1.show()

# ── CHART 2: CONDITION vs RETENTION ───────────

condition_order = ['Mint', 'Excellent', 'Good', 'Fair', 'Poor']
condition_stats = (
    df.groupby('condition')['retention_pct']
    .mean()
    .reindex(condition_order)
    .round(1)
    .reset_index()
)
condition_stats.columns = ['condition', 'retention_pct']

colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
fig2 = px.bar(condition_stats, x='condition', y='retention_pct',
              title='How Condition Affects Resale Value',
              labels={'retention_pct': 'Average Retention %', 'condition': 'Condition'},
              color='condition',
              color_discrete_sequence=colors,
              text='retention_pct')
fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig2.update_layout(showlegend=False, height=450)
fig2.write_html('chart2_condition_retention.html')
fig2.show()

# ── CHART 3: AGE vs SOLD PRICE SCATTER ────────

fig3 = px.scatter(df, x='age_years', y='sold_price',
                  color='gear_type',
                  title='Age vs Sold Price by Gear Type',
                  labels={'age_years': 'Age (years)',
                          'sold_price': 'Sold Price ($)',
                          'gear_type': 'Gear Type'},
                  hover_data=['brand', 'condition', 'original_price'],
                  opacity=0.6)
fig3.update_layout(height=500)
fig3.write_html('chart3_age_vs_price.html')
fig3.show()

# ── CHART 4: GEAR TYPE RETENTION ──────────────

gear_retention = (
    df.groupby('gear_type')['retention_pct']
    .mean()
    .sort_values(ascending=False)
    .round(1)
    .reset_index()
)
gear_retention.columns = ['gear_type', 'retention_pct']

fig4 = px.bar(gear_retention, x='gear_type', y='retention_pct',
              title='Value Retention by Gear Category',
              labels={'retention_pct': 'Average Retention %', 'gear_type': 'Gear Type'},
              color='gear_type',
              text='retention_pct')
fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig4.update_layout(showlegend=False, height=450)
fig4.write_html('chart4_gear_type_retention.html')
fig4.show()

print("✅ 4 interactive HTML charts saved and opened in browser.")