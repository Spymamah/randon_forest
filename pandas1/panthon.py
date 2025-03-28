import pandas as pd

# Check version
print(pd.__version__)

# Create a sample DataFrame
data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)
