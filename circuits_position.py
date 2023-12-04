import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

folder =r'C:\Users\thynnea\Downloads\formula1'
files = os.listdir(folder)
csv_file_name1 = [file for file in files if file.endswith('.csv')][0]
circuits = pd.read_csv(os.path.join(folder, csv_file_name1))

csv_file_name2 = [file for file in files if file.endswith('.csv')][10]
results = pd.read_csv(os.path.join(folder, csv_file_name2))
results.replace(r'\N', pd.NA, inplace = True)
results.dropna(subset = ['position'], inplace = True)
results['position'] = pd.to_numeric(results['position'], errors='coerce')

csv_file_name3 = [file for file in files if file.endswith('.csv')][9]
races = pd.read_csv(os.path.join(folder, csv_file_name3))
merged_df = pd.merge(races, results[['raceId', 'position', 'grid']], on = 'raceId')
final = merged_df.groupby('circuitId')[['grid', 'position']].corr().iloc[0::2,-1]
final = final.unstack()
plt.figure(figsize=(16, 12))
sns.heatmap(final, annot=True, cmap='viridis', linewidths=.3, fmt=".2f", cbar_kws={"shrink": 0.6}, annot_kws={"size": 6})
plt.title('Correlation between Starting and Finishing Positions at Each Circuit')
plt.xlabel('Starting Position', fontsize = 14)
plt.ylabel('Finishing Position', fontsize = 14)
plt.show()

