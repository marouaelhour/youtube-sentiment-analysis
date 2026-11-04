
import re
import pandas as pd
import os

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\.\S+", " ", text)

    text = re.sub(r"[@#]\w+", " ", text)

    text = re.sub(r"[^a-z0-9\s'!?.,]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def run():

    df = pd.read_csv("data/raw/reddit.csv")

    print("Dimensions avant nettoyage :", df.shape)

    df = df.rename(columns={
        "clean_comment": "text",
        "category": "label"
    })

    df = df.dropna(subset=["text", "label"])

    df["label"] = pd.to_numeric(
        df["label"], errors="coerce"
    )

    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([-1, 0, 1])]
    df["label"] = df["label"].astype(int)

    df["text"] = df["text"].apply(clean_text)

    df = df[df["text"].str.len() > 2]

    df = df.drop_duplicates(subset=["text"])

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        "data/processed/clean.csv",
        index=False
    )

    print("\nNettoyage terminé !")

    print("Dimensions après nettoyage :", df.shape)

    print("\nDistribution des sentiments :")
    print(df["label"].value_counts().sort_index())

    print("\nFichier enregistré dans data/processed/clean.csv")


if __name__ == "__main__":
    run()
