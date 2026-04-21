from sklearn.datasets import load_wine
import pandas as pd
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print("Shape:", df.shape)
print("\nClasses: ", wine.target_names)
print("\nClass distribution:\n ", df['target'].value_counts())
print("\n Feature stats: \n ", df.describe())
