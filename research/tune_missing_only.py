"""Search only Appendix-2 parameters absent from the existing all-model results.

The values already stored in result_*.json -> all -> best_params are never
sampled again.  They are loaded as fixed values for every Optuna trial.

Example
-------
python tune_missing_only.py --data D:\\data\\final_data_100k_64.parquet --trials 30

This is a validation-only tuner: it does not fit final models or overwrite any
result_*.json file.  Its output is missing_only_params.json.
"""
import argparse
import ast
import json
import random
import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from project_config import resolve_data_dir

SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Copied from the existing result_*.json reports' `all.best_params`.  This
# makes the file runnable by itself when those reports are not present on the
# execution machine.  Local report files, when available, still take priority.
EMBEDDED_ALL_BEST_PARAMS = {
    "RF": {"pos_weight": 1.0059696244768448, "n_estimators": 200, "max_depth": 16},
    "MLP": {"pos_weight": 14.126101375282198, "hidden_dim": 64, "lr": 0.008597863855323283, "dropout": 0.3962968545354246},
    "CNN": {"pos_weight": 2.3109194364678345, "num_filters": 128, "lr": 0.00853034202936806, "dropout": 0.16970609854442822},
    "RNN": {"pos_weight": 2.2272196620276747, "hidden_dim": 96, "num_layers": 2, "lr": 0.007491421003810144, "dropout": 0.11609455438801682},
    "LSTM": {"pos_weight": 1.5642024418779221, "hidden_dim": 64, "num_layers": 2, "lr": 0.0060389264225586565, "dropout": 0.14786809026946124},
    "CNN-LSTM": {"pos_weight": 2.7295633269204744, "filters": 32, "lstm_units": 96, "lr": 0.008661173591489274, "dropout": 0.2009426731154088},
    "RNN-LSTM": {"pos_weight": 3.3166707993289486, "rnn_units": 128, "lstm_units": 32, "lr": 0.003182923396179607, "dropout": 0.22877387637809685},
}


def resolve_data_path(value):
    """Accept a parquet file or a data directory on Windows/WSL alike."""
    text = str(value)
    # Jupyter in the supplied environment runs on Linux but accesses Windows
    # drives through /mnt/<drive>.  Convert a familiar D:\\... input for it.
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match and Path("/").exists() and not Path(text).exists():
        text = f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/') }"
    path = Path(text)
    return path / "final_data_64.parquet" if path.is_dir() else path


def read_json_objects(path):
    """Also tolerate old result files that accidentally have appended JSON."""
    text = path.read_text(encoding="utf-8-sig").lstrip()
    decoder, objects = json.JSONDecoder(), []
    while text:
        value, end = decoder.raw_decode(text)
        objects.append(value)
        text = text[end:].lstrip()
    return objects


def load_fixed_params(directory):
    aliases = {"CNNLSTM": "CNN-LSTM", "RNNLSTM": "RNN-LSTM", "xgb": "XGBoost", "tabnet": "TabNet"}
    fixed = {}
    # Result reports are often kept in a predict/results subfolder.  Search
    # below the supplied project directory rather than requiring co-location.
    for path in directory.rglob("result_*.json"):
        key = path.stem.removeprefix("result_")
        name = aliases.get(key, key.upper())
        # The last object containing `all` is the latest appended result.
        try:
            reports = read_json_objects(path)
        except (OSError, json.JSONDecodeError):
            # A partial/empty old report must not prevent use of the embedded
            # all.best_params fallback.
            continue
        for report in reports:
            if "all" in report and "best_params" in report["all"]:
                fixed[name] = report["all"]["best_params"].copy()
    return fixed


def feature_columns(mlp_path, columns):
    tree = ast.parse(mlp_path.read_text(encoding="utf-8-sig"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"cols_v0", "cols_v1", "cols_v2", "cols_v3", "cols_v4"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    wanted = sum((values[k] for k in ("cols_v0", "cols_v1", "cols_v2", "cols_v3", "cols_v4")), [])
    return [col for col in wanted if col in columns]


class MLP(nn.Module):
    # hidden_dim/dropout are fixed existing parameters; n_layers is newly searched.
    def __init__(self, width, p):
        super().__init__()
        layers, inp = [], width
        for _ in range(p["n_layers"]):
            layers += [nn.Linear(inp, p["hidden_dim"]), nn.ReLU(), nn.Dropout(p["dropout"])]
            inp = p["hidden_dim"]
        layers.append(nn.Linear(inp, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN(nn.Module):
    # kernel_size, pool_size and num_conv_layers are really applied per trial.
    def __init__(self, width, p):
        super().__init__()
        blocks, channels = [], 1
        for _ in range(p["num_conv_layers"]):
            blocks += [
                nn.Conv1d(channels, p["num_filters"], p["kernel_size"], padding=p["kernel_size"] // 2),
                nn.ReLU(), nn.MaxPool1d(p["pool_size"], ceil_mode=True),
            ]
            channels = p["num_filters"]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(p["dropout"]), nn.Linear(channels, 1))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1))).squeeze(1)


class Recurrent(nn.Module):
    def __init__(self, width, p, cell):
        super().__init__()
        cls = nn.RNN if cell == "RNN" else nn.LSTM
        self.rnn = cls(width, p["hidden_dim"], num_layers=p["num_layers"], batch_first=True,
                       dropout=p["dropout"] if p["num_layers"] > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(p["hidden_dim"], 1))

    def forward(self, x):
        return self.head(self.rnn(x.unsqueeze(1))[0][:, -1]).squeeze(1)


class CNNLSTM(nn.Module):
    def __init__(self, width, p):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, p["filters"], p["kernel_size"], padding=p["kernel_size"] // 2),
                                  nn.ReLU(), nn.MaxPool1d(p["pool_size"], ceil_mode=True))
        self.lstm = nn.LSTM(p["filters"], p["lstm_units"], batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(p["lstm_units"], 1))

    def forward(self, x):
        seq = self.conv(x.unsqueeze(1)).transpose(1, 2)
        return self.head(self.lstm(seq)[0][:, -1]).squeeze(1)


class RNNLSTM(nn.Module):
    def __init__(self, width, p):
        super().__init__()
        self.rnn = nn.RNN(width, p["rnn_units"], batch_first=True)
        self.lstm = nn.LSTM(p["rnn_units"], p["lstm_units"], batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(p["lstm_units"], 1))

    def forward(self, x):
        return self.head(self.lstm(self.rnn(x.unsqueeze(1))[0])[0][:, -1]).squeeze(1)


def neural_pr_auc(make_model, p, x_train, y_train, x_valid, y_valid, epochs):
    model = make_model(p).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["l2"])
    pos_weight = torch.tensor([p["pos_weight"]], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.float32)),
                        batch_size=p["batch_size"], shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(x_valid).to(DEVICE))).cpu().numpy()
    return float(average_precision_score(y_valid, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(resolve_data_dir()), help="Parquet file or data directory containing final_data_64.parquet")
    parser.add_argument("--results-dir", default=None, help="Folder containing result_*.json (searched recursively)")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument("--models", default="RF,MLP,CNN,RNN,LSTM,CNN-LSTM,RNN-LSTM")
    parser.add_argument("--out", default="missing_only_params.json")
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    here = Path(__file__).resolve().parent
    fixed = {name: params.copy() for name, params in EMBEDDED_ALL_BEST_PARAMS.items()}
    fixed.update(load_fixed_params(Path(args.results_dir) if args.results_dir else here))
    selected = {item.strip() for item in args.models.split(",")}
    missing_results = {"source": "result_*.json:all.best_params", "trials": args.trials, "epochs": args.epochs, "models": {}}

    data_path = resolve_data_path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_parquet(data_path)
    features = feature_columns(here / "MLP.py", df.columns)
    if not features:
        raise RuntimeError("Could not derive the all-feature column list from MLP.py")
    if len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=SEED)
    x = df[features].fillna(0).to_numpy(np.float32)
    y = df["is_purchased"].to_numpy(np.int32)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=.17647, stratify=y, random_state=SEED)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_valid = scaler.transform(x_valid).astype(np.float32)

    def run(name, objective):
        if name not in selected:
            return
        if name not in fixed:
            raise FileNotFoundError(f"result for {name} with all.best_params was not found")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=args.trials)
        missing_results["models"][name] = {
            "fixed_all_best_params": fixed[name],
            "missing_only_best_params": study.best_params,
            "validation_pr_auc": study.best_value,
        }
        print(name, missing_results["models"][name])

    def neural(name, build, suggest):
        def objective(trial):
            p = dict(fixed[name])
            p.update(suggest(trial))
            return neural_pr_auc(lambda q: build(x_train.shape[1], q), p, x_train, y_train, x_valid, y_valid, args.epochs)
        run(name, objective)

    def rf_objective(trial):
        p = fixed["RF"]
        weights = {0: 1.0, 1: p["pos_weight"]}
        model = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5), class_weight=weights,
            n_jobs=-1, random_state=SEED)
        return float(average_precision_score(y_valid, model.fit(x_train, y_train).predict_proba(x_valid)[:, 1]))
    run("RF", rf_objective)

    neural("MLP", MLP, lambda t: {"n_layers": t.suggest_int("n_layers", 1, 3), "l2": t.suggest_float("l2", 1e-6, 1e-2, log=True), "batch_size": t.suggest_categorical("batch_size", [16, 32, 64, 128])})
    neural("CNN", CNN, lambda t: {"kernel_size": t.suggest_int("kernel_size", 2, 5), "pool_size": t.suggest_int("pool_size", 2, 3), "num_conv_layers": t.suggest_int("num_conv_layers", 1, 2), "l2": t.suggest_float("l2", 1e-6, 1e-2, log=True), "batch_size": t.suggest_categorical("batch_size", [32, 64, 128])})
    neural("RNN", lambda w, p: Recurrent(w, p, "RNN"), lambda t: {"l2": t.suggest_float("l2", 1e-6, 1e-2, log=True), "batch_size": t.suggest_categorical("batch_size", [32, 64, 128])})
    neural("LSTM", lambda w, p: Recurrent(w, p, "LSTM"), lambda t: {"l2": t.suggest_float("l2", 1e-5, 1e-2, log=True), "batch_size": t.suggest_categorical("batch_size", [32, 64, 128])})
    neural("CNN-LSTM", CNNLSTM, lambda t: {"kernel_size": t.suggest_int("kernel_size", 2, 5), "pool_size": t.suggest_int("pool_size", 2, 4), "l2": t.suggest_float("l2", 1e-5, 1e-2, log=True), "batch_size": t.suggest_categorical("batch_size", [32, 64, 128])})
    neural("RNN-LSTM", RNNLSTM, lambda t: {"l2": t.suggest_float("l2", 1e-5, 1e-2, log=True), "batch_size": t.suggest_categorical("batch_size", [32, 64, 128])})
    Path(args.out).write_text(json.dumps(missing_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
