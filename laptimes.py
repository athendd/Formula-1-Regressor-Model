import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

folder =r'C:\Users\thynnea\Downloads\formula1'
files = os.listdir(folder)
csv_file_name1 = [file for file in files if file.endswith('.csv')][9]
races = pd.read_csv(os.path.join(folder, csv_file_name1))
csv_file_name2 = [file for file in files if file.endswith('.csv')][6]
races = races[races['year']>=2011]
races = races[races['year']<=2020]
lap_times = pd.read_csv(os.path.join(folder, csv_file_name2))
lap_times['milliseconds'] = lap_times['milliseconds'].astype(float)
lap_times = pd.merge(lap_times, races[['raceId', 'year']], on = 'raceId')
final = lap_times.groupby('year')['milliseconds'].mean()
plt.figure(figsize = (12,8))
final.plot(marker = 'o',linestyle = '-', color = 'red')
plt.title("Average Lap Times Over The Years", fontdict = {'fontsize':20, 'fontweight':'bold','color': 'black'})
plt.xlabel("Year", fontdict = {'fontsize':15,'color': 'black'})
plt.ylabel("Average Lap Time (milliseconds)", fontdict = {'fontsize':15,'color': 'black'})
plt.show()
