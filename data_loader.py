import pandas as pd

RANDOM_STATE = 42

# ---- Load Flipkart CSV ----
# Update this path to where your file actually is
df = pd.read_csv('/Users/apple/Downloads/flipkart_product.csv', encoding='latin-1')

print(f"Raw rows: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

# ---- Clean ----
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df = df.dropna(subset=['Rate', 'Summary'])
df = df[df['Rate'] != 3]

# ---- Sentiment label ----
df['sentiment'] = df['Rate'].apply(lambda x: 'positive' if x >= 4 else 'negative')

# ---- Combine Review + Summary as input text ----
df['input_text'] = df['Summary'].fillna('') + ' ' + df['Review'].fillna('')
df['input_text'] = df['input_text'].str.strip()

# ---- Category extraction ----
category_keywords = {
    'Electronics':  ['phone', 'mobile', 'laptop', 'tablet', 'camera',
                     'speaker', 'headphone', 'charger', 'cable', 'tv',
                     'television', 'monitor', 'keyboard', 'mouse', 'battery'],
    'Appliances':   ['cooler', 'fan', 'ac', 'refrigerator', 'washing',
                     'microwave', 'mixer', 'iron', 'heater', 'purifier'],
    'Fashion':      ['shirt', 'jeans', 'dress', 'saree', 'kurta', 'shoes',
                     'sandal', 'watch', 'bag', 'wallet', 'belt', 'trouser',
                     'legging', 'top', 'kurti', 'sari', 'chappal'],
    'Beauty':       ['cream', 'lotion', 'lipstick', 'shampoo', 'soap',
                     'perfume', 'moisturizer', 'serum', 'foundation', 'hair'],
    'Home':         ['bed', 'sofa', 'chair', 'table', 'curtain', 'pillow',
                     'mattress', 'lamp', 'shelf', 'storage', 'furniture'],
    'Kitchen':      ['pan', 'pot', 'cooker', 'bottle', 'container',
                     'knife', 'spoon', 'plate', 'glass', 'tiffin', 'flask'],
    'Sports':       ['yoga', 'gym', 'cycle', 'cricket', 'football',
                     'badminton', 'dumbbell', 'mat', 'racket', 'fitness'],
    'Toys':         ['toy', 'game', 'puzzle', 'doll', 'lego', 'kids',
                     'children', 'baby', 'infant', 'play'],
}

def extract_category(product_name):
    name_lower = str(product_name).lower()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    return 'Other'

df['category'] = df['ProductName'].apply(extract_category)

# ---- Balance per category ----
balanced_dfs = []
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    pos = cat_df[cat_df['sentiment'] == 'positive']
    neg = cat_df[cat_df['sentiment'] == 'negative']
    min_count = min(len(pos), len(neg))
    if min_count < 50:
        print(f"  Skipping {cat} — too few samples ({min_count})")
        continue
    balanced = pd.concat([
        pos.sample(n=min_count, random_state=RANDOM_STATE),
        neg.sample(n=min_count, random_state=RANDOM_STATE)
    ])
    balanced_dfs.append(balanced)
    print(f"  {cat}: {min_count} pos + {min_count} neg = {min_count*2:,} rows")

df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df_final = df_final[['input_text', 'sentiment', 'category', 'Rate']]
df_final.columns = ['input_text', 'sentiment', 'category', 'rating']

# ---- Save ----
df_final.to_csv('flipkart_balanced.csv', index=False)

print(f"\nTotal rows saved: {len(df_final):,}")
print("\nRows per category:")
print(df_final['category'].value_counts())
print("\nSentiment balance:")
print(df_final['sentiment'].value_counts())
print("\nSaved as: flipkart_balanced.csv")