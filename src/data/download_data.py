import os
import requests
import pandas as pd


url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"


os.makedirs("data/raw", exist_ok=True)

response = requests.get(url)
response.raise_for_status()

with open("data/raw/reddit.csv", "wb") as file:
    file.write(response.content)

print("Dataset téléchargé avec succès !")


df = pd.read_csv("data/raw/reddit.csv")


print("\nDimensions du dataset :")
print(df.shape)

print("\nLes colonnes :")
print(df.columns.tolist())

print("\nLes premières lignes :")
print(df.head())

print("\nDistribution des sentiments :")
print(df["category"].value_counts().sort_index())
print(df.info())

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nCommentaires dupliqués :")
print(df.duplicated().sum())
