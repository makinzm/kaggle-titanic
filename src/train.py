"""My 1st Submission 0.74641."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import (
    SimpleImputer,
)
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.model_selection import (
    KFold,
)
from sklearn.preprocessing import (
    LabelEncoder,
)

import wandb
from wandb.integration.lightgbm import (
    log_summary,
    wandb_callback,
)

RANDOM_SEED = 42

# Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV data."""
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_df["Survived"] = np.nan
    return train_df, test_df

def _add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title from the Name column."""
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Rev": "Rev",
        "Dr": "Dr",
        "Col": "Col",
        "Major": "Major",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mme": "Mrs",
        "Don": "Mr",
        "Sir": "Mr",
        "Lady": "Mrs",
        "Capt": "Col",
        "the Countess": "Mrs",
        "Jonkheer": "Mr",
        "Dona": "Mrs",
    }
    df["Title"] = df["Name"].map(lambda x: x.split(",")[1].split(".")[0].strip())
    df["Title"] = df["Title"].map(title_mapping)
    return df

# Simplified feature engineering
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified feature engineering."""
    df = _add_title(df) #noqa: PD901
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])
    df["Title"] = LabelEncoder().fit_transform(df["Title"].fillna("Mr"))
    return df

# Prepare data for modeling
def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for modeling."""
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "CabinKnown", "Title"]
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    train_df[features] = imputer.fit_transform(train_df[features])
    test_df[features] = imputer.transform(test_df[features])

    X = train_df[features].to_numpy()
    y = train_df["Survived"].to_numpy()
    test_x = test_df[features].to_numpy()
    return X, y, test_x

# Cross-validation and model training
def train_and_evaluate(X: np.ndarray, y: np.ndarray, test_x: np.ndarray, cfg: dict) -> np.ndarray:
    """Train and evaluate."""
    num_cv = 10
    kf = KFold(n_splits=num_cv, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    test_predictions = np.zeros(test_x.shape[0])

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train LightGBM model
        lgbm = lgb.LGBMClassifier(**cfg)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                 callbacks=[
                    lgb.early_stopping(5, first_metric_only=True),
                    lgb.log_evaluation(period=10),
                    wandb_callback(),
                ])

        # Evaluate the model
        y_pred = lgbm.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        cv_scores.append(accuracy)
        print(f"{i+1}/{num_cv}::: CV Result({accuracy})")
        # Predict on the test set
        test_predictions += lgbm.predict_proba(test_x)[:, 1]
        log_summary(lgbm.booster_, save_model_checkpoint=True)
        wandb.log({
            "CV Accuracy Fold": accuracy,
            "Feature Importance": wandb.Table(data=[[f, imp] for f, imp in zip(range(len(lgbm.feature_importances_)), lgbm.feature_importances_, strict=False)],
                                                          columns=["Feature", "Importance"]),
        })

    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.4f}")
    return test_predictions / kf.get_n_splits()

# Main function to execute the workflow
def main() -> None:
    """Train and Inference."""
    cfg = {
        "random_state": RANDOM_SEED,
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbose": -1,
    }
    wandb.init(
        project = "kaggle-titanic-init",
        config = cfg,
    )

    train_df, test_df = load_data()
    X, y, test_x = prepare_data(train_df, test_df)
    test_predictions = train_and_evaluate(X, y, test_x, cfg)

    # Create submission file
    PassengerId = test_df["PassengerId"]  # Ensure PassengerId is correctly referenced
    threshold = 0.5
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": (test_predictions > threshold).astype(np.int32)})
    submission.to_csv("my_submission.csv", index=False)

    wandb.finish()

if __name__ == "__main__":
    main()
