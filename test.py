import pandas as pd

df = pd.read_excel('data/gis.xlsx')
df['FZI'] = pd.Series('', index=df.index)
df.to_excel('FZI_predictions.xlsx')

