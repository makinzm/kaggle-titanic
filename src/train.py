"""My 1st Submission 0.74641."""
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for modeling."""
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "CabinKnown", "Title"]
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    train_df[features] = imputer.fit_transform(train_df[features])
    test_df[features] = imputer.transform(test_df[features])

    train_feature = train_df[features]
    X = train_feature.to_numpy()
    y = train_df["Survived"].to_numpy()
    test_x = test_df[features].to_numpy()
    return train_feature, X, y, test_x

# Cross-validation and model training
def train_and_evaluate(X: np.ndarray, y: np.ndarray, test_x: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[lgb.LGBMClassifier]]:
    """Train and evaluate."""
    num_cv = 10
    kf = KFold(n_splits=num_cv, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    test_predictions = np.zeros(test_x.shape[0])
    cv_predictions = np.zeros(X.shape[0])
    cv_models = []

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
        cv_predictions[val_index] = y_pred
        cv_models.append(lgbm)

    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.4f}")
    return test_predictions / kf.get_n_splits(), cv_predictions, cv_models

def plot_wrong_cv(df: pd.DataFrame, cv_prediction: np.ndarray, target_col: str, order: int = 100) -> None:
    """正しい予測と誤った予測の分布を比較するヒストグラムとカウントプロットを描画します。

    Args:
        df: データフレーム
        cv_prediction: 予測値の配列
        target_col: ターゲットカラム名
        order: 考慮する誤った予測の数
    """
    wrong_indices = np.where(df[target_col] != cv_prediction)[0]
    wrong_indices = wrong_indices[:order]

    print(f"Number of wrong predictions: {len(wrong_indices)}")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = [col for col in df.select_dtypes(include=["object"]).columns if col != target_col]  # Remove target_col if not categorical

    # サブプロットの設定
    fig, axes = plt.subplots(nrows=len(df.columns) // 2, ncols=2, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]

        if col in numeric_cols:
            sns.kdeplot(df[col], label="All", ax=ax, fill=True)
            sns.kdeplot(df.loc[wrong_indices, col], label="Wrong", ax=ax, fill=True)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.legend()

        elif col in categorical_cols:
            all_density = df[col].value_counts(normalize=True) / len(df)
            wrong_density = df.loc[wrong_indices, col].value_counts(normalize=True) / len(wrong_indices)
            sns.barplot(x=all_density.index, y=all_density.values, ax=ax, color="skyblue", label="All")
            sns.barplot(x=wrong_density.index, y=wrong_density.values, ax=ax, color="orange", label="Wrong")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.legend()
        else:
            continue

        ax.set_title(col)

    plt.tight_layout()
    plt.savefig("wrong_result.png")

def plot_feature_importance(models: list[lgb.LGBMClassifier], feature_train_df: pd.DataFrame) -> None:
    """Plot feature importance.

    Reference:
    nyk510: [#2 初心者向け講座 モデルを改善する](https://www.guruguru.science/competitions/22/discussions/4f07af18-cd54-4bf4-bb9f-e3a90e6a9f69/)
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        one_fold = pd.DataFrame()
        one_fold["feature_importance"] = model.feature_importances_
        one_fold["column"] = feature_train_df.columns
        one_fold["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, one_fold],                                  axis=0, ignore_index=True)

    order = feature_importance_df.groupby("column")\
        .mean()[["feature_importance"]]\
        .sort_values("feature_importance", ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df,
                  x="feature_importance",
                  y="column",
                  hue="column",
                  legend=False,
                  order=order,
                  ax=ax,
                  palette="pastel",
                  fill = True,
                  orient="h")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()
    return fig, ax

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
    train_feature_df, X, y, test_x = prepare_data(train_df, test_df)
    test_predictions, cv_prediction, cv_models = train_and_evaluate(X, y, test_x, cfg)

    plot_wrong_cv(train_df, cv_prediction, target_col="Survived")
    fig, _ = plot_feature_importance(cv_models, train_feature_df)
    fig.savefig("feature_importance.png")

    # Create submission file
    PassengerId = test_df["PassengerId"]  # Ensure PassengerId is correctly referenced
    threshold = 0.5
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": (test_predictions > threshold).astype(np.int32)})
    submission.to_csv("my_submission.csv", index=False)

    wandb.finish()

if __name__ == "__main__":
    main()
