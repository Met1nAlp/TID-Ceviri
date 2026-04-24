import json, sys
sys.path.insert(0, '.')

worst_ids = [126,172,65,6,122,221,95,206,47,53,192,62,175,86,111]
best_ids  = [223,219,214,204,201,196,191,187,180,179,174,170,168,167,157]

try:
    with open('class_mapping.json','r',encoding='utf-8') as f:
        m = json.load(f)
    print('class_mapping.json yuklendi')
except:
    import pandas as pd
    from src.training.config import DATA_DIR
    df = pd.read_csv(str(DATA_DIR / 'SignList_ClassId_TR_EN.csv'))
    m = {str(int(row['ClassId'])): row['TR'] for _,row in df.iterrows()}
    print('CSV yuklendi')

print('\n--- EN IYI SINIFLAR (yuzde 100 dogru) ---')
for i in best_ids:
    print(f'  ClassID {i:3d}  ->  {m.get(str(i), "?")}')

print('\n--- EN KOTÜ SINIFLAR ---')
for i in worst_ids:
    print(f'  ClassID {i:3d}  ->  {m.get(str(i), "?")}')
