#dataset should have driverId, averaging finishing position, total points earned, teamId, team performance, circuit, starting position, number of pit stops
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler






folder =r'C:\Users\thynnea\Downloads\formula1'
files = os.listdir(folder)

csv_file_name1 = [file for file in files if file.endswith('.csv')][10]
results = pd.read_csv(os.path.join(folder, csv_file_name1))
csv_file_name3 = [file for file in files if file.endswith('.csv')][9]
races = pd.read_csv(os.path.join(folder, csv_file_name3))
csv_file_name5 = [file for file in files if file.endswith('.csv')][2]
constructor_results = pd.read_csv(os.path.join(folder, csv_file_name5))
csv_file_name6 = [file for file in files if file.endswith('.csv')][1]
constructors = pd.read_csv(os.path.join(folder, csv_file_name6))
csv_file_name7 = [file for file in files if file.endswith('.csv')][3]
constructor_standings = pd.read_csv(os.path.join(folder, csv_file_name7))


df = pd.merge(constructor_results, results, on=['raceId', 'constructorId'], suffixes=('_constructor', '_results'))
df = pd.merge(df, races, on='raceId')
df = pd.merge(df, constructors, on='constructorId')
df = pd.merge(df, constructor_standings, on=['raceId', 'constructorId'], suffixes=('_constructor', '_standings'))
# Feature engineering

df = df[(df['year'] >= 1961) & (df['year'] <= 1990)]
selected_columns = [
    'grid', 'laps',  
    'position_constructor',   
    'year', 'round', 'circuitId', 'points', 'position_standings', 'wins'  
]
print(df['position_constructor'])


df = df[selected_columns]
# Handle categorical variables if any
df = pd.get_dummies(df)
df = df.dropna()
X = df.drop('points', axis=1)
y = df['points']




y= np.log1p(y)


model = RandomForestRegressor(n_estimators=225, max_depth= 25, random_state=42)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and train the Random Forest Regressor model
# Make predictions on the test set
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
#one = model.feature_importances_
#two = X.columns
#importance_pairs = list(zip(one, two))
#sorted_important_pairs = sorted(importance_pairs, key = lambda x:x[0], reverse = True)
#for importance, variable in sorted_important_pairs:
    #print(f"{variable}: {importance}")


plt.figure(figsize=(12, 8))
plt.scatter(X_test['year'], np.expm1(y_test), label='Actual Points', alpha=0.7)
plt.scatter(X_test['year'], np.expm1(predictions), label='Predicted Points', alpha=0.7)
plt.title('Actual vs Predicted Points Over Time')
plt.xlabel('Year')
plt.ylabel('Points')
plt.legend()
plt.grid(True)
plt.show()

#big jump in the data 
#noticed that the target column was skewed so I decided to fix that 
