import sys
import os
import random
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Read the file "penalties.csv"
df = pd.read_csv("penalties.csv")
# Print the top 5 rows
#print(df.head())
# Show the size of data
#print(df.shape)
# Look if there are some "holes" lol
#print(df.isnull().sum())
# Count each direction, turn the counts into percentages and
# pick out the baseline
#print(df["direction"].value_counts())
#print(df["direction"].value_counts(normalize=True).max())
# Separate answer from clues
X=(df[["keeper_tendency", "footedness", "last_direction", "match_situation"]])
y=(df["direction"])
#print(X.head())
#print(y.head())
# Change words to numbers lmao
X_numbers = pd.get_dummies(X)
#print(X_numbers.head())
# Cut the 200 rows into two piles. It shuffles them first, so the pile
# isn't just "the first 160 rows"
X_train, X_test, y_train, y_test = train_test_split(
    X_numbers, y, test_size=0.2, random_state=42
)
# Exhibit
#print(X_train.shape)
#print(X_test.shape)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
#print("Model trained!")
new_penalty = pd.DataFrame({"footedness" : ["left"], "keeper_tendency" : ["right"], "last_direction" : ["left"], "match_situation" : ["drawing"]})
#print(new_penalty)
new_penalty_numbers = pd.get_dummies(new_penalty)
#print(new_penalty_numbers.head())
new_penalty_numbers = new_penalty_numbers.reindex(columns=X_numbers.columns, fill_value=0)
#print(new_penalty_numbers.shape)
#print(new_penalty_numbers)
guess = model.predict(new_penalty_numbers)
#print(guess)
test_guesses = model.predict(X_test)
#print(accuracy_score(y_test, test_guesses))
accuracy = accuracy_score(y_test, test_guesses)
print("Baseline: 0.345, Model:", accuracy)
# My model scored 57.5% on 40 unseen penalties, compared with 34.5% for always guessing the most common direction, so it learned real patterns from my data.
# Ask the tester for the penalty taker's details
foot = input("Footedness (left/right): ").strip().lower()
last = input("Last penalty direction (left/center/right): ").strip().lower()
situation = input("Match situation (winning/drawing/losing): ").strip().lower()
keeper = input("Keeper tendency (left/center/right): ").strip().lower()

# Build the one-row table from the answers
new_penalty = pd.DataFrame({
    "footedness": [foot],
    "last_direction": [last],
    "match_situation": [situation],
    "keeper_tendency": [keeper],
})
new_penalty_numbers = pd.get_dummies(new_penalty)
new_penalty_numbers = new_penalty_numbers.reindex(columns=X_numbers.columns, fill_value=0)

# 6. Predict and show the answer
guess = model.predict(new_penalty_numbers)
print("The model thinks he will shoot", guess[0])
