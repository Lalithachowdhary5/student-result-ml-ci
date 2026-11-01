import csv
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def generate_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_samples = 500

    study_hours = rng.integers(1, 11, size=n_samples)
    attendance = rng.integers(50, 100, size=n_samples)
    assignment_scores = rng.integers(40, 100, size=n_samples)
    practice_tests = rng.integers(0, 11, size=n_samples)

    score = (
        1.8 * study_hours
        + 0.5 * (attendance / 10)
        + 0.7 * (assignment_scores / 10)
        + 2.2 * practice_tests
    )
    labels = (score >= 14).astype(int)

    noisy_mask = rng.random(n_samples) < 0.08
    labels[noisy_mask] = 1 - labels[noisy_mask]

    features = np.column_stack([
        study_hours,
        attendance,
        assignment_scores,
        practice_tests,
    ])
    return features, labels


def write_dataset_csv(features: np.ndarray, labels: np.ndarray) -> None:
    with open("student_results.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["study_hours", "attendance", "assignment_score", "practice_tests", "result"])
        for row, label in zip(features, labels):
            writer.writerow([*row, label])


def main() -> None:
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    with open("student_result_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    write_dataset_csv(X, y)

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Model Accuracy   :", round(accuracy, 4))
    print("Training records :", len(X_train))
    print("Testing records  :", len(X_test))


if __name__ == "__main__":
    main()
