from sklearn.feature_extraction import FeatureHasher
import pandas as pd 

data = {'Color': ['Red', 'Green', 'Blue', 'Red', 'Yellow']}
df = pd.DataFrame(data)

hasher = FeatureHasher(n_features=3, input_type='string')

# ✅ Her rengi liste içinde veriyoruz
hashed_features = hasher.transform([[color] for color in df['Color']])

hashed_df = pd.DataFrame(
    hashed_features.toarray(), 
    columns=['Feature1', 'Feature2', 'Feature3']
)
print("Hashed Features Df:")
print(hashed_df)