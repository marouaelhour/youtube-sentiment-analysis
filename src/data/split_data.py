
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import os

def run():

    # 1. Charger le dataset nettoyé
    df = pd.read_csv("data/processed/clean.csv")

    print("Dimensions initiales :", df.shape)

    # 2. Afficher la distribution initiale
    print("\nDistribution avant équilibrage :")
    print(df["label"].value_counts().sort_index())

    # 3. Séparer les commentaires et les labels
    X = df[["text"]]
    y = df["label"]

    # 4. Équilibrer les classes
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X, y)

    print("\nDistribution après équilibrage :")
    print(y_balanced.value_counts().sort_index())

    # 5. Séparer les données en Train et Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.20,
        random_state=42,
        stratify=y_balanced
    )

    # 6. Créer les fichiers Train et Test
    train_data = pd.DataFrame({
        "text": X_train["text"],
        "label": y_train
    })

    test_data = pd.DataFrame({
        "text": X_test["text"],
        "label": y_test
    })

    os.makedirs("data/processed", exist_ok=True)

    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)

    # 7. Afficher les résultats
    print("\nDistribution dans Train :")
    print(y_train.value_counts().sort_index())

    print("\nDistribution dans Test :")
    print(y_test.value_counts().sort_index())

    print("\nDimensions finales :")
    print("Train :", train_data.shape)
    print("Test :", test_data.shape)

    print("\nSéparation terminée !")


if __name__ == "__main__":
    run()
