import argparse, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
from sklearn.preprocessing import PolynomialFeatures


# Simulator function
def sim_run_batch(p, input1, input2):
    input1 = np.asarray(input1, dtype=float)
    input2 = np.asarray(input2, dtype=float)
    raw = (p.alpha * input1) + (p.beta * input2) - (p.gamma * input1 * input2)
    raw = raw * p.efficiency
    return np.clip(raw, p.min_power, p.max_power)

# Build model based on kind
def build_model(kind):
    k = kind.lower()
    if k == "linearregression_poly2":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
    
    if k == "linearregression":
        return LinearRegression()

    if k == "randomforest":
        return RandomForestRegressor(n_estimators=500, max_depth=None,
                      min_samples_leaf=1, random_state=42)
    raise ValueError("Use LinearRegression or RandomForest")

# Root Mean Squared Error calc
def rmse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

# Mean Absolute Error calc
def mae(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean(np.abs(y - yhat)))

# Mean Absolute Percentage Error calc
def mape(y, yhat, eps=1e-9):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    d = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - yhat) / d)) * 100.0)

# Create parameter grid from sweeps
def grid_from_sweeps(sweeps):
    keys = sorted(sweeps.keys())
    vals = [list(v) for v in sweeps.values()]
    rows = list(itertools.product(*vals))
    return pd.DataFrame(rows, columns=keys)

# Generate synthetic dataset
def generate_synth(path, n, input1=(0.0, 1.0), input2=(5.0, 30.0),
                   noise=0.5, seed=42):
    rng = np.random.default_rng(seed)
    i1 = rng.uniform(input1[0], input1[1], size=n)
    i2 = rng.uniform(input2[0], input2[1], size=n)
    y = sim_run_batch(PARAMS, i1, i2) + rng.normal(0.0, noise, size=n)
    y = np.clip(y, PARAMS.min_power, PARAMS.max_power)
    pd.DataFrame({"input1": i1, "input2": i2, "output": y}).to_csv(
        path, index=False
    )
    return path

# Global parameters
class SimParams:
    def __init__(self,
                 alpha=8.0,
                 beta=0.7,
                 gamma=0.1,
                 efficiency=0.9,
                 min_power=0.0,
                 max_power=50.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.efficiency = efficiency
        self.min_power = min_power
        self.max_power = max_power

PARAMS = SimParams()

def run_training_and_report(data_path, model_kind, report_path):
    # Load dataset
    df = pd.read_csv(data_path)
    if not {"input1", "input2", "output"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: input1, input2, output")

    X = df[["input1", "input2"]].values
    y = df["output"].values

    # train/test split (modified with .1, .2, .3, amd .4 test sizes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )

    # train
    model = build_model(model_kind)
    model.fit(X_train, y_train)

    # predict
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    # train metrics
    train_rmse = rmse(y_train, yhat_train)
    train_mae = mae(y_train, yhat_train)
    train_mape = mape(y_train, yhat_train)
    train_r2 = r2_score(y_train, yhat_train)

    # test metrics
    test_rmse = rmse(y_test, yhat_test)
    test_mae = mae(y_test, yhat_test)
    test_mape = mape(y_test, yhat_test)
    test_r2 = r2_score(y_test, yhat_test)

    # Print 
    print("\n===== MODEL PERFORMANCE =====")
    print(f"Model: {model_kind}")
    print(f"-- TRAIN --")
    print(f"RMSE : {train_rmse:.4f}")
    print(f"MAE  : {train_mae:.4f}")
    print(f"MAPE : {train_mape:.2f}%")
    print(f"R²   : {train_r2:.4f}")

    print(f"\n-- TEST --")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"MAE  : {test_mae:.4f}")
    print(f"MAPE : {test_mape:.2f}%")
    print(f"R²   : {test_r2:.4f}")

    # save detailed report 
    # Name report as baseName_modelKind.csv using report_path
    base, ext = os.path.splitext(report_path)
    if ext == "":
        ext = ".csv"
    report_file = f"{base}_{model_kind}{ext}"

    df["prediction"] = model.predict(X)  # predictions on all data
    df["abs_error"] = np.abs(df["output"] - df["prediction"])
    df.to_csv(report_file, index=False)
    print(f"\nSaved detailed report: {report_file}")

    # parameter sweep eval
    sweep_rmse = sweep_mae = sweep_mape = None
    try:
        i1_min, i1_max = df["input1"].min(), df["input1"].max()
        i2_min, i2_max = df["input2"].min(), df["input2"].max()

        sweeps = {
            "input1": np.linspace(i1_min, i1_max, 25),
            "input2": np.linspace(i2_min, i2_max, 25)
        }
        sweep = grid_from_sweeps(sweeps)

        X_sw = sweep[["input1", "input2"]].values
        y_sw_model = model.predict(X_sw)
        y_sw_sim = sim_run_batch(
            PARAMS,
            sweep["input1"].values,
            sweep["input2"].values
        )

        sweep_rmse = rmse(y_sw_sim, y_sw_model)
        sweep_mae = mae(y_sw_sim, y_sw_model)
        sweep_mape = mape(y_sw_sim, y_sw_model)

        print("\n===== PARAMETER SWEEP (Model vs Simulator) =====")
        print(f"Sweep RMSE : {sweep_rmse:.4f}")
        print(f"Sweep MAE  : {sweep_mae:.4f}")
        print(f"Sweep MAPE : {sweep_mape:.2f}%")

    except Exception as e:
        print("Sweep evaluation skipped:", e)

    # APPEND ROW TO model_summary.csv
    summary_row = pd.DataFrame([{
        "model_name": f"{model_kind}_ts30", # modified to reflect test cases
        "train_RMSE": train_rmse,
        "train_MAE": train_mae,
        "train_MAPE": train_mape,
        "train_R2": train_r2,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "test_MAPE": test_mape,
        "test_R2": test_r2,
        "sweep_RMSE": sweep_rmse,
        "sweep_MAE": sweep_mae,
        "sweep_MAPE": sweep_mape
    }])

    summary_path = "model_summary.csv"
    write_header = not os.path.exists(summary_path)
    summary_row.to_csv(summary_path, mode="a",
                       header=write_header, index=False)

    print(f"\nUpdated {summary_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Digital Twin Surrogate Lab")
    parser.add_argument(
        "--data",
        type=str,
        default="sample_data.csv",
        help="Path to training CSV (must include input1, input2, output)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LinearRegression",
        choices=["LinearRegression", "RandomForest", "LinearRegression_Poly2"],
        help="Surrogate model type"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="twin_report.csv",
        help="Base path to save output report (model name will be appended)"
    )

    args = parser.parse_args(argv)
    run_training_and_report(args.data, args.model, args.report)


if __name__ == "__main__":
    main()