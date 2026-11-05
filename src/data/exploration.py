import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/processed/clean.csv")


print("Distribution des classes :")
print(df["label"].value_counts())


df["label"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribution des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de commentaires")
plt.xticks(
    ticks=[0, 1, 2],
    labels=["Négatif", "Neutre", "Positif"],
    rotation=0
)

plt.tight_layout()
plt.savefig("data/processed/distribution.png")
plt.show()


df["text_length"] = df["text"].apply(
    lambda x: len(x.split())
)

print("\nLongueur moyenne :")
print(df["text_length"].mean())

print("\nLongueur maximale :")
print(df["text_length"].max())


df["text_length"].hist(bins=30)

plt.title("Longueur des commentaires")
plt.xlabel("Nombre de mots")
plt.ylabel("Fréquence")

plt.tight_layout()
plt.savefig("data/processed/text_length.png")
plt.show()