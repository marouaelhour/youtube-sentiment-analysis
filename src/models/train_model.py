
import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


def run():

    # Créer le dossier models
    os.makedirs("models", exist_ok=True)

    # 1. Charger les données
    train_data = pd.read_csv("data/processed/train.csv")
    test_data = pd.read_csv("data/processed/test.csv")

    # TF-IDF attend une Series contenant les textes
    X_train = train_data["text"].fillna("").astype(str)
    y_train = train_data["label"]

    X_test = test_data["text"].fillna("").astype(str)
    y_test = test_data["label"]

    print("Dimensions Train :", X_train.shape)
    print("Dimensions Test :", X_test.shape)

    print("\nDistribution des classes dans Train :")
    print(y_train.value_counts().sort_index())

    print("\nDistribution des classes dans Test :")
    print(y_test.value_counts().sort_index())

    # 2. Définir les algorithmes
    # TF-IDF transforme les textes en valeurs numériques.
    # RandomOverSampler équilibre uniquement les données Train.
    # Le sampler est exécuté à chaque fold de validation croisée.

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("sampler", RandomOverSampler(random_state=42)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("sampler", RandomOverSampler(random_state=42)),
            ("classifier", LinearSVC(random_state=42))
        ]),

        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("sampler", RandomOverSampler(random_state=42)),
            ("classifier", MultinomialNB())
        ])
    }

    # 3. Optimiser Logistic Regression avec GridSearchCV
    print("\nOptimisation de Logistic Regression...")

    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [1, 10, 20]
    }

    grid_search = GridSearchCV(
        estimator=models["Logistic Regression"],
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_lr = grid_search.best_estimator_

    print("\nMeilleurs paramètres :")
    print(grid_search.best_params_)

    # 4. Entraîner et comparer les algorithmes
    trained_models = {
        "Logistic Regression": best_lr
    }

    for name in ["Linear SVM", "Naive Bayes"]:

        print(f"\nEntraînement de {name}...")

        models[name].fit(X_train, y_train)
        trained_models[name] = models[name]

    results = []

    print("\n========== COMPARAISON DES MODÈLES ==========")

    # 5. Évaluer les trois algorithmes
    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        f1_macro = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=[-1, 0, 1],
            output_dict=True,
            zero_division=0
        )

        results.append({
            "Algorithme": name,
            "Accuracy": accuracy,
            "F1-score macro": f1_macro,
            "F1 négatif": report["-1"]["f1-score"],
            "F1 neutre": report["0"]["f1-score"],
            "F1 positif": report["1"]["f1-score"]
        })

        print(f"\n{name}")
        print("Accuracy :", round(accuracy, 4))
        print("F1-score macro :", round(f1_macro, 4))

        print(classification_report(
            y_test,
            y_pred,
            labels=[-1, 0, 1],
            target_names=["Négatif", "Neutre", "Positif"],
            zero_division=0
        ))

    # Tableau comparatif
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1-score macro",
        ascending=False
    )

    print("\nTableau comparatif :")
    print(results_df.to_string(index=False))

    # Enregistrer le tableau comparatif
    results_df.to_csv(
        "models/comparison_results.csv",
        index=False
    )

    # 6. Sélectionner le meilleur modèle
    best_name = results_df.iloc[0]["Algorithme"]
    best_model = trained_models[best_name]

    print("\nMeilleur modèle :", best_name)

    # 7. Évaluation finale du meilleur modèle
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n========== ÉVALUATION DU MEILLEUR MODÈLE ==========")

    print("\nAccuracy :")
    print(round(accuracy, 4))

    print("\nRapport de classification :")

    print(classification_report(
        y_test,
        y_pred,
        labels=[-1, 0, 1],
        target_names=["Négatif", "Neutre", "Positif"],
        zero_division=0
    ))

    # 8. Visualiser la matrice de confusion
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[-1, 0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Négatif", "Neutre", "Positif"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(f"Matrice de confusion - {best_name}")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()

    # 9. Mesurer le temps d'inférence pour 50 commentaires
    batch = X_test.iloc[:50]

    # Préparation du modèle avant la mesure
    best_model.predict(batch)

    repetitions = 100

    start_time = time.perf_counter()

    for _ in range(repetitions):
        best_model.predict(batch)

    end_time = time.perf_counter()

    inference_time = (
        (end_time - start_time) / repetitions
    ) * 1000

    print("\nTemps moyen d'inférence pour 50 commentaires :")
    print(round(inference_time, 2), "ms")

    # 10. Vérifier les critères du TP
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[-1, 0, 1],
        output_dict=True,
        zero_division=0
    )

    f1_negative = report_dict["-1"]["f1-score"]
    f1_neutral = report_dict["0"]["f1-score"]
    f1_positive = report_dict["1"]["f1-score"]

    print("\n========== VÉRIFICATION DES CRITÈRES ==========")

    accuracy_ok = accuracy >= 0.80

    f1_ok = (
        f1_negative > 0.75
        and f1_neutral > 0.75
        and f1_positive > 0.75
    )

    time_ok = inference_time < 100

    print(
        "Accuracy >= 80 % :",
        "VALIDÉ" if accuracy_ok else "NON VALIDÉ"
    )

    print(
        "F1-score de chaque classe > 0.75 :",
        "VALIDÉ" if f1_ok else "NON VALIDÉ"
    )

    print(
        "Inférence < 100 ms pour 50 commentaires :",
        "VALIDÉ" if time_ok else "NON VALIDÉ"
    )

    # 11. Sauvegarder le meilleur modèle
    joblib.dump(best_model, "models/best_model.joblib")

    print("\nMeilleur modèle sauvegardé dans models/best_model.joblib")
    print("Résultats enregistrés dans models/comparison_results.csv")
    print("Matrice de confusion enregistrée dans models/confusion_matrix.png")


if __name__ == "__main__":
    run()
