import numpy as np
import pandas as pd
from data.wrmsse import WRMSSEEvaluator

# Fake data for 3 items
# Item 1: Normal
# Item 2: Sales are 0 continuously, creating missing price data (represented as NaN)
# Item 3: Has a NaN scale due to missing history

np.random.seed(42)

train_sales = np.array([
    [2, 3, 4, 2, 5] + [1] * 28,  # normal
    [0, 0, 0, 0, 0] + [0] * 28,  # zero sales
    [0, 1, np.nan, 2, 3] + [0] * 28,  # nan sales
])

train_prices = np.array([
    [1.5]*33,
    [np.nan]*33,  # Price is completely missing (NaN)
    [2.3]*33,
])

metadata = pd.DataFrame({
    'item_id': ['it1', 'it2', 'it3'],
    'store_id': ['st1', 'st1', 'st1'],
    'dept_id': ['de1', 'de1', 'de1'],
    'cat_id': ['ca1', 'ca1', 'ca1'],
    'state_id': ['sa1', 'sa1', 'sa1']
})

evaluator = WRMSSEEvaluator(train_sales, train_prices, metadata)

print("Scales:", evaluator.scales)
print("Weights:", evaluator.weights)

# Fake predictions and actuals
preds = np.random.rand(3, 28)
actuals = np.random.rand(3, 28)

wrmsse = evaluator.compute_wrmsse(preds, actuals)
print("WRMSSE:", wrmsse)
