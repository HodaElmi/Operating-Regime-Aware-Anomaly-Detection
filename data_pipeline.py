import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import plot_labels_counts


IGNORE_LABEL = -1


def load_cbm_dataset(path):
    data = np.loadtxt(path)
    X = data[:, :-2]
    y = data[:, -2:]
    return X, y

CMAPSS_UNIT_COL = 0
CMAPSS_CYCLE_COL = 1


def load_cmapss_dataset(train_path="./CMAPSSData/train_03.npy", test_path="./CMAPSSData/test_03.npy"):
    train_raw = np.load(train_path)
    test_raw = np.load(test_path)

    y_train = train_raw[:, -1]
    y_test = test_raw[:, -1]

    unit_train = train_raw[:, CMAPSS_UNIT_COL].astype(int)
    unit_test = test_raw[:, CMAPSS_UNIT_COL].astype(int)

    n_cols = train_raw.shape[1] - 1
    keep_cols = [c for c in range(n_cols) if c not in (CMAPSS_UNIT_COL, CMAPSS_CYCLE_COL)]

    X_train = train_raw[:, keep_cols]
    X_test = test_raw[:, keep_cols]

    plot_labels_counts(y_train)
    return X_train, y_train, unit_train, X_test, y_test, unit_test


def label_coefficients(y):

    y_binned = np.zeros_like(y, dtype=int)

    for col in range(y.shape[1]):
        col_values = y[:, col]
        low, high = col_values.min(), col_values.max()
        bins = np.linspace(low, high, 4)
        labels = np.digitize(col_values, bins[1:-1]) - 1
        labels = np.clip(labels, -1, 1)
        y_binned[:, col] = labels

    y_shifted = y_binned + 1
    y_combined = np.minimum(y_shifted[:, 0], y_shifted[:, 1])

    plot_labels_counts(y_combined)
    return y_combined


def cbm_binary_labels(y_class, healthy_label=2):

    return (y_class.astype(int) != healthy_label).astype(int)


def cmapss_sparse_binary_labels(y, normal_rul=100, anomaly_rul=30):

    y = np.asarray(y)
    unique = np.unique(y)

    if set(unique.astype(int).tolist()).issubset({0, 1}) and len(unique) <= 2:
        return y.astype(int)

    if set(unique.astype(int).tolist()).issubset({0, 1, 2}) and len(unique) <= 3:
        labels = np.full(len(y), IGNORE_LABEL, dtype=int)
        labels[y.astype(int) == 2] = 0
        labels[y.astype(int) == 0] = 1
        return labels

    labels = np.full(len(y), IGNORE_LABEL, dtype=int)
    labels[y > normal_rul] = 0
    labels[y <= anomaly_rul] = 1
    return labels


def print_split_stats(name, y):
    labels, counts = np.unique(y, return_counts=True)
    stats = {int(k): int(v) for k, v in zip(labels, counts)}
    print(f"[{name}] labels -1=ignore, 0=normal, 1=anomaly: {stats}")


def chronological_split(X, y, split_train=0.7, split_val=0.15):
    n = len(X)
    n_train = int(n * split_train)
    n_val = int(n * split_val)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_train_val_tail(X, y, val_split=0.2):

    n_val = int(len(X) * val_split)
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def split_train_val_by_unit(X, y, unit_ids, val_split=0.2, seed=42):

    rng = np.random.RandomState(seed)
    units = np.unique(unit_ids)
    rng.shuffle(units)

    n_val_units = max(1, int(round(len(units) * val_split)))
    val_units = set(units[:n_val_units].tolist())

    val_mask = np.isin(unit_ids, list(val_units))
    train_mask = ~val_mask

    return (
        X[train_mask], y[train_mask], unit_ids[train_mask],
        X[val_mask], y[val_mask], unit_ids[val_mask],
    )


def create_forecast_windows(X, y, regimes, window_size=32, stride=1, drop_ignore=True):

    X_windows, targets, labels, reg_ids = [], [], [], []
    T = len(X)

    for start in range(0, T - window_size, stride):
        end = start + window_size
        label = int(y[end])
        if drop_ignore and label == IGNORE_LABEL:
            continue

        X_windows.append(X[start:end])
        targets.append(X[end])
        labels.append(label)
        reg_ids.append(regimes[end])

    return (
        np.asarray(X_windows),
        np.asarray(targets),
        np.asarray(labels),
        np.asarray(reg_ids),
    )


def create_forecast_windows_grouped(X, y, regimes, unit_ids, window_size=32, stride=1, drop_ignore=True):

    order = np.argsort(unit_ids, kind="stable")
    unit_ids = unit_ids[order]
    X = X[order]
    y = y[order]
    regimes = regimes[order]

    X_chunks, t_chunks, y_chunks, r_chunks = [], [], [], []

    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        Xw, tw, yw, rw = create_forecast_windows(
            X[mask], y[mask], regimes[mask],
            window_size=window_size, stride=stride, drop_ignore=drop_ignore,
        )
        if len(Xw) == 0:
            continue
        X_chunks.append(Xw)
        t_chunks.append(tw)
        y_chunks.append(yw)
        r_chunks.append(rw)

    if not X_chunks:
        return (
            np.empty((0, window_size, X.shape[1])),
            np.empty((0, X.shape[1])),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    return (
        np.concatenate(X_chunks),
        np.concatenate(t_chunks),
        np.concatenate(y_chunks),
        np.concatenate(r_chunks),
    )


class ForecastDataset(Dataset):
    def __init__(self, X, target, label, regime):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.long)
        self.regime = torch.tensor(regime, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.target[idx], self.label[idx], self.regime[idx]


def _fit_regime_model(X_for_fit, n_regimes=3, regime_feature_idx=None, random_state=42):
    if regime_feature_idx is None:
        regime_feature_idx = list(range(min(3, X_for_fit.shape[1])))

    if len(X_for_fit) < n_regimes:
        raise ValueError("Not enough samples to fit the regime model.")

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_for_fit[:, regime_feature_idx])

    kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    kmeans.fit(Z_train)

    return scaler, kmeans, regime_feature_idx


def _fit_regime_model_healthy_only(X_train, y_train, n_regimes=3, regime_feature_idx=None):
    normal_idx = y_train == 0
    X_fit = X_train[normal_idx]
    if len(X_fit) < max(32, n_regimes):
        print(f"[warning] only {len(X_fit)} healthy points for regime fitting. Falling back to all non-ignored training points.")
        X_fit = X_train[y_train != IGNORE_LABEL]
    return _fit_regime_model(X_fit, n_regimes=n_regimes, regime_feature_idx=regime_feature_idx)


def _predict_regimes(X, regime_scaler, regime_model, regime_feature_idx):
    Z = regime_scaler.transform(X[:, regime_feature_idx])
    return regime_model.predict(Z)


def _make_loader(X, target, label, regime, batch_size, shuffle):
    ds = ForecastDataset(X, target, label, regime)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_cbm_dataloaders(
    path,
    window_size=32,
    stride=1,
    batch_size=64,
    split_train=0.7,
    split_val=0.15,
    split_test=0.15,
    n_regimes=3,
    regime_feature_idx=None,
):
    X, y_raw = load_cbm_dataset(path)
    y_class = label_coefficients(y_raw)
    y_anom = cbm_binary_labels(y_class, healthy_label=2)

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X, y_anom, split_train=split_train, split_val=split_val
    )
    print_split_stats("CBM train", y_train)
    print_split_stats("CBM val", y_val)
    print_split_stats("CBM test", y_test)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    regime_scaler, regime_model, regime_feature_idx = _fit_regime_model_healthy_only(
        X_train, y_train, n_regimes=n_regimes, regime_feature_idx=regime_feature_idx
    )

    r_train = _predict_regimes(X_train, regime_scaler, regime_model, regime_feature_idx)
    r_val = _predict_regimes(X_val, regime_scaler, regime_model, regime_feature_idx)
    r_test = _predict_regimes(X_test, regime_scaler, regime_model, regime_feature_idx)

    Xw_train, tw_train, yw_train, rw_train = create_forecast_windows(
        X_train, y_train, r_train, window_size, stride
    )
    Xw_val, tw_val, yw_val, rw_val = create_forecast_windows(
        X_val, y_val, r_val, window_size, stride
    )
    Xw_test, tw_test, yw_test, rw_test = create_forecast_windows(
        X_test, y_test, r_test, window_size, stride
    )

    train_loader = _make_loader(Xw_train, tw_train, yw_train, rw_train, batch_size, True)
    val_loader = _make_loader(Xw_val, tw_val, yw_val, rw_val, batch_size, False)
    test_loader = _make_loader(Xw_test, tw_test, yw_test, rw_test, batch_size, False)

    meta = {
        "x_scaler": x_scaler,
        "regime_scaler": regime_scaler,
        "regime_model": regime_model,
        "regime_feature_idx": regime_feature_idx,
    }
    return train_loader, val_loader, test_loader, meta


def build_cmapss_dataloaders(
    train_path="./CMAPSSData/train_03.npy",
    test_path="./CMAPSSData/test_03.npy",
    window_size=32,
    stride=1,
    batch_size=64,
    val_split=0.2,
    n_regimes=3,
    regime_feature_idx=None,
    normal_rul=100,
    anomaly_rul=30,
):
    (
        X_train_raw, y_train_raw, unit_train_raw,
        X_test_raw, y_test_raw, unit_test,
    ) = load_cmapss_dataset(train_path, test_path)

    y_train_full = cmapss_sparse_binary_labels(
        y_train_raw, normal_rul=normal_rul, anomaly_rul=anomaly_rul
    )
    y_test = cmapss_sparse_binary_labels(
        y_test_raw, normal_rul=normal_rul, anomaly_rul=anomaly_rul
    )

    X_train_raw, y_train, unit_train, X_val_raw, y_val, unit_val = split_train_val_by_unit(
        X_train_raw, y_train_full, unit_train_raw, val_split=val_split
    )
    print_split_stats("CMAPSS train", y_train)
    print_split_stats("CMAPSS val", y_val)
    print_split_stats("CMAPSS test", y_test)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)

    regime_scaler, regime_model, regime_feature_idx = _fit_regime_model_healthy_only(
        X_train, y_train, n_regimes=n_regimes, regime_feature_idx=regime_feature_idx
    )

    r_train = _predict_regimes(X_train, regime_scaler, regime_model, regime_feature_idx)
    r_val = _predict_regimes(X_val, regime_scaler, regime_model, regime_feature_idx)
    r_test = _predict_regimes(X_test, regime_scaler, regime_model, regime_feature_idx)

    Xw_train, tw_train, yw_train, rw_train = create_forecast_windows_grouped(
        X_train, y_train, r_train, unit_train, window_size, stride
    )
    Xw_val, tw_val, yw_val, rw_val = create_forecast_windows_grouped(
        X_val, y_val, r_val, unit_val, window_size, stride
    )
    Xw_test, tw_test, yw_test, rw_test = create_forecast_windows_grouped(
        X_test, y_test, r_test, unit_test, window_size, stride
    )

    train_loader = _make_loader(Xw_train, tw_train, yw_train, rw_train, batch_size, True)
    val_loader = _make_loader(Xw_val, tw_val, yw_val, rw_val, batch_size, False)
    test_loader = _make_loader(Xw_test, tw_test, yw_test, rw_test, batch_size, False)

    meta = {
        "x_scaler": x_scaler,
        "regime_scaler": regime_scaler,
        "regime_model": regime_model,
        "regime_feature_idx": regime_feature_idx,
        "normal_rul": normal_rul,
        "anomaly_rul": anomaly_rul,
    }
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    # NOTE: data.ipynb saves files as train_01.npy/test_01.npy (FD001) and
    # train_03.npy/test_03.npy (FD003) -- there is no train_1.npy/test_1.npy
    # produced anywhere in the notebook. Make sure config.yaml's
    # cmapss_train_path / cmapss_test_path (and the paths below) point at a
    # file that actually exists on disk, e.g. train_01.npy / test_01.npy.
    train_loader, val_loader, test_loader, _ = build_cmapss_dataloaders(
        train_path="./CMAPSSData/train_01.npy",
        test_path="./CMAPSSData/test_01.npy",
        window_size=32,
        stride=1,
        batch_size=64,
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))