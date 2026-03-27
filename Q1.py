import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
medal_counts = pd.read_csv('/Users/huangruisi/Desktop/data/summerOly_medal_counts.csv', encoding='utf-8')
hosts = pd.read_csv('/Users/huangruisi/Desktop/data/summerOly_hosts.csv', encoding='utf-8')
programs = pd.read_csv('/Users/huangruisi/Desktop/data/summerOly_programs.csv', encoding='utf-8')

medal_counts = medal_counts[medal_counts['Year'] <= 2024]
X = medal_counts[['Year', 'Gold', 'Total']]
y_gold = medal_counts['Gold']
y_total = medal_counts['Total']

# Split the dataset into training and testing sets
X_train, X_test, y_gold_train, y_gold_test = train_test_split(X, y_gold, test_size=0.2, random_state=42)
_, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# Time series prediction models (can be improved)
gold_model = LinearRegression()
gold_model.fit(X_train, y_gold_train)

total_model = LinearRegression()
total_model.fit(X_train, y_total_train)

# Evaluation
gold_predictions = gold_model.predict(X_test)
total_predictions = total_model.predict(X_test)

gold_mse = mean_squared_error(y_gold_test, gold_predictions)
total_mse = mean_squared_error(y_total_test, total_predictions)

print(f'Gold Medal MSE: {gold_mse}')
print(f'Total Medal MSE: {total_mse}')

# Predict medal counts for 2028:
X_2028 = medal_counts[medal_counts['Year'] == 2024].copy()
X_2028['Year'] = 2028

X_2028['Predicted_Gold_2028'] = gold_model.predict(X_2028[['Year', 'Gold', 'Total']])
X_2028['Predicted_Total_2028'] = total_model.predict(X_2028[['Year', 'Gold', 'Total']])

# Save prediction results:
X_2028[['NOC', 'Predicted_Gold_2028', 'Predicted_Total_2028']].to_csv('/Users/huangruisi/Desktop/data/summerOly_medal_counts2.csv', mode='a', header=False, index=False)

# Identify countries that may perform better or worse:
improvements = X_2028[['NOC', 'Predicted_Gold_2028', 'Predicted_Total_2028']].copy()
improvements = improvements.merge(medal_counts[medal_counts['Year'] == 2024][['NOC', 'Gold', 'Total']], on='NOC')
improvements['Gold_Change'] = improvements['Predicted_Gold_2028'] - improvements['Gold']
improvements['Total_Change'] = improvements['Predicted_Total_2028'] - improvements['Total']

print(improvements.sort_values(by='Gold_Change', ascending=False).head(10))
print(improvements.sort_values(by='Total_Change', ascending=False).head(10))

# Predict the number of countries that will win their first gold medal:
first_medal_countries = improvements[(improvements['Gold'] == 0) & (improvements['Predicted_Gold_2028'] > 0)]
print(f'Countries predicted to win their first gold medal: {len(first_medal_countries)}')

# Consider the impact of the number and type of events on medal counts:
event_impact = programs.groupby('Sport').sum().reset_index()
event_impact = event_impact[['Sport', '2024']].rename(columns={'2024': 'Event_Count_2024'})

# Ensure 'NOC' column is present in event_impact before merging
event_impact = event_impact.merge(medal_counts[medal_counts['Year'] == 2024][['NOC', 'Total']], left_on='Sport', right_on='NOC', how='left')

# Print the top 10 sports with the most events in 2024 and their corresponding total medal counts
print(event_impact.sort_values(by='Event_Count_2024', ascending=False).head(10))
