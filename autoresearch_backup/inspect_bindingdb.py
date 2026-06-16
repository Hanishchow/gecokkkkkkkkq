import pandas as pd
import pyarrow.parquet as pq

base = r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup'

f = base + '/binddb_train-00000-of-00002.parquet'
df = pd.read_parquet(f, columns=['ligand', 'protein', 'ic50']).head(10)
print('Columns:', list(df.columns))
print('Sample ligand:', df['ligand'].iloc[0])
print('Sample protein:', df['protein'].iloc[0][:200])
print('Sample ic50:', df['ic50'].iloc[0])
print('ic50 dtype:', df['ic50'].dtype)
print()

# Show a few more
for i in range(min(3, len(df))):
    print(f'Row {i}: ligand={str(df.ligand.iloc[i])[:80]}, ic50={df.ic50.iloc[i]}')

# Count total rows and non-null
total = pq.ParquetFile(f).metadata.num_rows
print(f'\nFile 1 total rows: {total}')
df_all = pd.read_parquet(f, columns=['ic50'])
print(f'ic50 non-null: {df_all.ic50.notna().sum()}/{total}')

f2 = base + '/binddb_train-00001-of-00002.parquet'
total2 = pq.ParquetFile(f2).metadata.num_rows
print(f'\nFile 2 total rows: {total2}')
df_all2 = pd.read_parquet(f2, columns=['ic50'])
print(f'ic50 non-null: {df_all2.ic50.notna().sum()}/{total2}')
