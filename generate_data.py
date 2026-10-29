import os
import random
import pandas as pd
# Set the random seed so the dataset can be reproduced.
# Running the program again with the same code and seed produces
# the same sequence of random choices.
random.seed(42)
# Define the possible directions for a penalty kick
directions = ["left", "center", "right"]
# Create an empty list to store the data for every penalty.
rows = []
# Generate 200 simulated penalty situations.
for i in range(200):
    # Randomly choose the penalty taker's preferred foot,
    # the direction of their previous penalty, the match situation,
    # and the goalkeeper's tendency.
    last = random.choice(directions)
    foot = random.choice(["left", "right"])
    situation = random.choice(["winning", "drawing", "losing"])
    keeper = random.choice(directions)
    # Generate a random number to decide the initial penalty direction.
    # The probabilities below introduce patterns into the synthetic data.
    roll = random.random()
    if roll < 0.5:
        direction = foot
    elif roll < 0.8:
        direction = last
    else:
        direction = random.choice(directions)
    # Generate another random number to influence the direction
    # according to the match situation.
    pressure = random.random()
    # If the team is losing, the rules may change the initial prediction.
    if situation == "losing":
        if pressure < 0.4:
            direction = "center"
        elif pressure < 0.7:
            direction = keeper
    # If the match is tied, the previous direction or keeper tendency
    # may influence the final direction.
    elif situation == "drawing":
        if pressure < 0.5:
            direction = last
        elif pressure < 0.7:
            direction = keeper
    # If the team is winning, the taker's preferred foot or the
    # goalkeeper's tendency may influence the final direction.
    elif situation == "winning":
        if pressure < 0.5:
            direction = foot
        elif pressure < 0.6:
            direction = keeper
    # Store one complete simulated penalty as a list of feature values
    # followed by the final direction, which will be our target.
    rows.append([foot, last, situation, keeper, direction])

# Convert the collected rows into a pandas DataFrame.
# A DataFrame organises data into rows and named columns, like a table.
df = pd.DataFrame(rows, columns=["footedness", "last_direction",
                                 "match_situation", "keeper_tendency", "direction"])
# Export the dataset to a CSV file so it can be used later.
# index=False prevents pandas from adding an extra numbered column.
df.to_csv("penalties.csv", index=False)
# Display the folder where the CSV file was saved.
print("Saved in folder:", os.getcwd())
# Count how many penalties ended in each direction.
# This helps us inspect the overall distribution of the target variable.
print(df["direction"].value_counts())
# Count the penalty directions separately for each match situation.
# This helps us inspect whether the generated rules create patterns.
print(df.groupby("match_situation")["direction"].value_counts())
