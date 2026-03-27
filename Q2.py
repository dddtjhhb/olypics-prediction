import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv', encoding='utf-8')
athletes = pd.read_csv('data/summerOly_athletes.csv', encoding='utf-8')

# Identify instances of notable coaches changing countries
notable_coaches = {
    'Lang Ping': {'countries': ['USA', 'CHN'], 'start_years': [2005, 2013], 'end_years': [2008, 2021]},
    'Coach B': {'countries': ['USA'], 'start_years': [2010], 'end_years': [2020]},
    # Add more coaches as needed
}

# Analyze changes in medal counts before, during, and after a coach's tenure
def analyze_coach_impact(coach_info):
    results = []
    for country, start_year, end_year in zip(coach_info['countries'], coach_info['start_years'], coach_info['end_years']):
        before_tenure = medal_counts[(medal_counts['NOC'] == country) & (medal_counts['Year'] < start_year)]
        during_tenure = medal_counts[(medal_counts['NOC'] == country) & (medal_counts['Year'] >= start_year) & (medal_counts['Year'] <= end_year)]
        after_tenure = medal_counts[(medal_counts['NOC'] == country) & (medal_counts['Year'] > end_year)]
        
        before_medals = before_tenure['Total'].mean() if not before_tenure.empty else 0
        during_medals = during_tenure['Total'].mean() if not during_tenure.empty else 0
        after_medals = after_tenure['Total'].mean() if not after_tenure.empty else 0
        
        results.append((country, before_medals, during_medals, after_medals))
    return results

# Collect data for modeling
coach_impact_data = []
for coach, info in notable_coaches.items():
    impacts = analyze_coach_impact(info)
    for country, before, during, after in impacts:
        coach_impact_data.append({
            'coach': coach,
            'country': country,
            'before_medals': before,
            'during_medals': during,
            'after_medals': after
        })

coach_impact_df = pd.DataFrame(coach_impact_data)

# Create a model to quantify the impact of great coaches on medal counts
X = coach_impact_df[['before_medals']]
y = coach_impact_df['during_medals'] - coach_impact_df['before_medals']

model = LinearRegression()
model.fit(X, y)

# Estimate the contribution of the coaching effect to the total medal count
coach_impact_df['predicted_impact'] = model.predict(coach_impact_df[['before_medals']])
coach_impact_df['actual_impact'] = coach_impact_df['during_medals'] - coach_impact_df['before_medals']

print(coach_impact_df)

# Identify three countries and sports where investing in great coaches could have a significant impact
countries_sports = [
    {'country': 'USA', 'sport': 'Swimming'},
    {'country': 'CHN', 'sport': 'Gymnastics'},
    {'country': 'GBR', 'sport': 'Athletics'},
    # Add more countries and sports as needed
]

# Estimate the potential increase in medals for these countries and sports
def estimate_potential_increase(country, sport):
    avg_medals = athletes[(athletes['NOC'] == country) & (athletes['Sport'] == sport)]['Medal'].count() / len(athletes['Year'].unique())
    predicted_increase = model.predict([[avg_medals]])[0]
    return predicted_increase

potential_increases = []
for cs in countries_sports:
    increase = estimate_potential_increase(cs['country'], cs['sport'])
    potential_increases.append({
        'country': cs['country'],
        'sport': cs['sport'],
        'predicted_increase': increase
    })

potential_increases_df = pd.DataFrame(potential_increases)
print(potential_increases_df)
