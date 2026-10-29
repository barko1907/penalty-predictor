# Penalty Predictor ⚽🧤

Predicts where a penalty taker will shoot (left, center or right) using a Decision Tree. Built by a goalkeeper with Python, pandas and scikit-learn.

## How it works
- `generate_data.py` creates 200 synthetic penalties from football rules plus random noise.
- `penalty_predictor.py` trains a model on 80% of the data, tests it on the other 20%, then asks you for a new taker's details and predicts his shot.

**Inputs:** footedness, last penalty direction, match situation, keeper tendency

## Results
| Strategy | Accuracy |
|---|---|
| Always guess the most common direction | 34.5% |
| Decision Tree | **57.5%** |

## Run it
```bash
pip install pandas scikit-learn
python generate_data.py
python penalty_predictor.py
```
Type `center`, not `centre`.

## Limitations
The data is synthetic, so the model learns my rules, not real players. This is a learning project, not a scouting tool.

## Next
- Real labelled penalties
- Video analysis with OpenCV and MediaPipe
