import json
import pandas as pd

# https://www.kaggle.com/sureshmecad/mens-shoe-prices
df = pd.read_csv('MensShoePrices/archive/train.csv')

new_df = []
for i, row in df.iterrows():
    new_row = {
        'id': row.id,
        'dateupdated': row.dateupdated
    }
    new_json = {}
    for key, value in row.drop(new_row).items():
        if pd.isna(value):
            continue
        try:
            value = json.loads(value)
        except:
            pass
        new_json[key] = value
    new_row['graph'] = new_json
    new_df.append(new_row)
new_df = pd.DataFrame(new_df)

new_new_df = []
for name, group in new_df.groupby(['id', 'dateupdated']):
    new_new_df.append({
        'id': name[0],
        'dateupdated': name[1],
        'graph': json.dumps(group.json.tolist())
    })
new_new_df = pd.DataFrame(new_new_df)

new_new_df.to_csv('MensShoePrices.csv', index=False)
