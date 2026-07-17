import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


IGNORE_LABEL = -1


def load_cbm_dataset(path):
    data = np.loadtxt(path)
    X = data[:, :-2]
    y = data[:, -2:]
    return X, y


CMAPSS_UNIT_COL = 0
CMAPSS_CYCLE_COL = 1
CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _cmapss_paths(data_dir, fd):
    fd = fd.upper()
    if fd not in CMAPSS_SUBSETS:
        raise ValueError(f"Unknown C-MAPSS subset '{fd}', expected one of {CMAPSS_SUBSETS}")
    return (
        os.path.join(data_dir, f"train_{fd}.txt"),
        os.path.join(data_dir, f"test_{fd}.txt"),
        os.path.join(data_dir, f"RUL_{fd}.txt"),
    )


def _cmapss_train_rul(raw):
    unit = raw[:, CMAPSS_UNIT_COL].astype(int)
    cycle = raw[:, CMAPSS_CYCLE_COL]
    rul = np.zeros(len(raw))
    for u in np.unique(unit):
        mask = unit == u
        rul[mask] = cycle[mask].max() - cycle[mask]
    return rul


def _cmapss_test_rul(raw, rul_at_end):

    unit = raw[:, CMAPSS_UNIT_COL].astype(int)
    cycle = raw[:, CMAPSS_CYCLE_COL]
    rul = np.zeros(len(raw))
    for u, final_rul in zip(np.unique(unit), rul_at_end):
        mask = unit == u
        rul[mask] = final_rul + (cycle[mask].max() - cycle[mask])
    return rul


def load_cmapss_dataset(fd = "FD003", data_dir = "./CMAPSSData"):

    train_path, test_path, rul_path = _cmapss_paths(data_dir, fd)

    train_raw = np.loadtxt(train_path)
    test_raw = np.loadtxt(test_path)
    rul_at_end = np.loadtxt(rul_path)

    y_train = _cmapss_train_rul(train_raw)
    y_test = _cmapss_test_rul(test_raw, rul_at_end)

    unit_train = train_raw[:, CMAPSS_UNIT_COL].astype(int)
    unit_test = test_raw[:, CMAPSS_UNIT_COL].astype(int)

    keep_cols = [c for c in range(train_raw.shape[1]) if c not in (CMAPSS_UNIT_COL, CMAPSS_CYCLE_COL)]
    X_train = train_raw[:, keep_cols]
    X_test = test_raw[:, keep_cols]

    return X_train, y_train, unit_train, X_test, y_test, unit_test


def cmapss_sparse_binary_labels(y, normal_rul = 100, anomaly_rul = 30):
    """normal: RUL > normal_rul | anomaly: RUL <= anomaly_rul | ignore: in between."""
    y = np.asarray(y)
    labels = np.full(len(y), IGNORE_LABEL, dtype=int)
    labels[y > normal_rul] = 0
    labels[y <= anomaly_rul] = 1
    return labels


def print_split_stats(name, y):
    labels, counts = np.unique(y, return_counts = True)
    stats = {int(k): int(v) for k, v in zip(labels, counts)}
    print(f"[{name}] labels -1=ignore, 0=normal, 1=anomaly: {stats}")


def chronological_split(X, y, split_train = 0.7, split_val = 0.15):
    n = len(X)
    n_train = int(n * split_train)
    n_val = int(n * split_val)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_train_val_tail(X, y, val_split = 0.2):

    n_val = int(len(X) * val_split)
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def split_train_val_by_unit(X, y, unit_ids, val_split = 0.2, seed = 42):

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


def create_forecast_windows(X, y, regimes, window_size = 32, stride = 1, drop_ignore = True):

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

    X_chunks, t_chunks, y_chunks, r_chunks, u_chunks = [], [], [], [], []

    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        Xw, tw, yw, rw = create_forecast_windows(
            X[mask], y[mask], regimes[mask],
            window_size=window_size, stride=stride, drop_ignore = drop_ignore,
        )
        if len(Xw) == 0:
            continue
        X_chunks.append(Xw)
        t_chunks.append(tw)
        y_chunks.append(yw)
        r_chunks.append(rw)
        u_chunks.append(np.full(len(Xw), uid))

    if not X_chunks:
        return (
            np.empty((0, window_size, X.shape[1])),
            np.empty((0, X.shape[1])),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    return (
        np.concatenate(X_chunks),
        np.concatenate(t_chunks),
        np.concatenate(y_chunks),
        np.concatenate(r_chunks),
        np.concatenate(u_chunks),
    )



def cbm_physical_binary_labels(y_coeff, healthy_threshold=0.975):
    y_coeff = np.asarray(y_coeff, dtype=float)
    healthy = (y_coeff[:, 0] >= healthy_threshold) & (y_coeff[:, 1] >= healthy_threshold)
    return (~healthy).astype(int)


def create_pointwise_samples(X, y, regimes):
    """Create length-1 pseudo-windows for non-temporal/tabular CBM rows."""
    X_seq = X[:, None, :]
    target = X.copy()
    return X_seq, target, y.astype(int), regimes.astype(int)


class ForecastDataset(Dataset):
    def __init__(self, X, target, label, regime, unit=None):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.target = torch.tensor(target, dtype = torch.float32)
        self.label = torch.tensor(label, dtype = torch.long)
        self.regime = torch.tensor(regime, dtype = torch.long)

        if unit is None:
            unit = np.arange(len(X))
        self.unit = torch.tensor(unit, dtype = torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.target[idx], self.label[idx], self.regime[idx]


def _fit_regime_model(X_for_fit, n_regimes = 3, regime_feature_idx = None, random_state = 42,
                       algorithm = "kmeans", hdbscan_min_cluster_size = None, knn_n_neighbors = 5):
    if regime_feature_idx is None:
        regime_feature_idx = list(range(min(3, X_for_fit.shape[1])))

    if len(X_for_fit) < n_regimes:
        raise ValueError("Not enough samples to fit the regime model.")

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_for_fit[:, regime_feature_idx])

    if algorithm == "kmeans":
        model = KMeans(n_clusters = n_regimes, random_state=random_state, n_init = 10)
        model.fit(Z_train)
        knn = None
    elif algorithm == "hdbscan":
        if hdbscan_min_cluster_size is None:
            raise ValueError("hdbscan_min_cluster_size is required when algorithm='hdbscan'.")
        model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        train_labels = model.fit_predict(Z_train)
        non_noise = train_labels != -1
        knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
        knn.fit(Z_train[non_noise], train_labels[non_noise])
    else:
        raise ValueError(f"Unknown regime clustering algorithm '{algorithm}'.")

    return scaler, model, regime_feature_idx, knn


def _fit_regime_model_healthy_only(X_train, y_train, n_regimes=3, regime_feature_idx=None,
                                    algorithm="kmeans", hdbscan_min_cluster_size=None, knn_n_neighbors=5):
    normal_idx = y_train == 0
    X_fit = X_train[normal_idx]
    if len(X_fit) < max(32, n_regimes):
        print(f"[warning] only {len(X_fit)} healthy points for regime fitting. Falling back to all non-ignored training points.")
        X_fit = X_train[y_train != IGNORE_LABEL]
    return _fit_regime_model(
        X_fit, n_regimes=n_regimes, regime_feature_idx=regime_feature_idx,
        algorithm=algorithm, hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        knn_n_neighbors=knn_n_neighbors,
    )


def _predict_regimes(X, regime_scaler, regime_model, regime_feature_idx, regime_knn=None):
    Z = regime_scaler.transform(X[:, regime_feature_idx])
    if regime_knn is not None:
        return regime_knn.predict(Z)
    return regime_model.predict(Z)


def _make_loader(X, target, label, regime, batch_size, shuffle, unit=None):
    ds = ForecastDataset(X, target, label, regime, unit=unit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_cbm_dataloaders(
    path,
    window_size = 32,
    stride = 1,
    batch_size = 64,
    split_train = 0.7,
    split_val = 0.15,
    split_test = 0.15,
    n_regimes = 3,
    regime_feature_idx = None,
    healthy_threshold = 0.975,
    algorithm = "kmeans",
    hdbscan_min_cluster_size = None,
):

    if abs((split_train + split_val + split_test) - 1.0) > 1e-6:
        raise ValueError("CBM split_train + split_val + split_test must sum to 1.0")

    X, y_raw = load_cbm_dataset(path)
    y_anom = cbm_physical_binary_labels(y_raw, healthy_threshold = healthy_threshold)

    idx = np.arange(len(X))
    train_idx, temp_idx = train_test_split(
        idx,
        train_size = split_train,
        random_state = 42,
        shuffle = True,
        stratify = y_anom,
    )
    relative_test = split_test / (split_val + split_test)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size = relative_test,
        random_state = 42,
        shuffle = True,
        stratify = y_anom[temp_idx],
    )

    X_train, y_train = X[train_idx], y_anom[train_idx]
    X_val, y_val = X[val_idx], y_anom[val_idx]
    X_test, y_test = X[test_idx], y_anom[test_idx]

    print(f"[CBM] point-wise mode enabled; ignoring window_size={window_size} and stride={stride}.")
    print(f"[CBM] healthy_threshold={healthy_threshold}; normal means both decay coefficients >= threshold.")
    print_split_stats("CBM train", y_train)
    print_split_stats("CBM val", y_val)
    print_split_stats("CBM test", y_test)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    regime_scaler, regime_model, regime_feature_idx, regime_knn = _fit_regime_model_healthy_only(
        X_train, y_train, n_regimes=n_regimes, regime_feature_idx=regime_feature_idx,
        algorithm=algorithm, hdbscan_min_cluster_size=hdbscan_min_cluster_size,
    )

    r_train = _predict_regimes(X_train, regime_scaler, regime_model, regime_feature_idx, regime_knn)
    r_val = _predict_regimes(X_val, regime_scaler, regime_model, regime_feature_idx, regime_knn)
    r_test = _predict_regimes(X_test, regime_scaler, regime_model, regime_feature_idx, regime_knn)

    Xw_train, tw_train, yw_train, rw_train = create_pointwise_samples(X_train, y_train, r_train)
    Xw_val, tw_val, yw_val, rw_val = create_pointwise_samples(X_val, y_val, r_val)
    Xw_test, tw_test, yw_test, rw_test = create_pointwise_samples(X_test, y_test, r_test)

    train_loader = _make_loader(Xw_train, tw_train, yw_train, rw_train, batch_size, True)
    val_loader = _make_loader(Xw_val, tw_val, yw_val, rw_val, batch_size, False)
    test_loader = _make_loader(Xw_test, tw_test, yw_test, rw_test, batch_size, False)

    meta = {
        "x_scaler": x_scaler,
        "regime_scaler": regime_scaler,
        "regime_model": regime_model,
        "regime_feature_idx": regime_feature_idx,
        "regime_knn": regime_knn,
        "healthy_threshold": healthy_threshold,
        "pointwise_cbm": True,
    }
    return train_loader, val_loader, test_loader, meta

def build_cmapss_dataloaders(
    fd = "FD003",
    data_dir = "./CMAPSSData",
    window_size = 32,
    stride = 1,
    batch_size = 64,
    val_split = 0.2,
    n_regimes = 3,
    regime_feature_idx = None,
    normal_rul = 100,
    anomaly_rul = 30,
    algorithm = "kmeans",
    hdbscan_min_cluster_size = None,
):
    (
        X_train_raw, y_train_raw, unit_train_raw,
        X_test_raw, y_test_raw, unit_test,
    ) = load_cmapss_dataset(fd = fd, data_dir = data_dir)

    y_train_full = cmapss_sparse_binary_labels(
        y_train_raw, normal_rul = normal_rul, anomaly_rul = anomaly_rul
    )
    y_test = cmapss_sparse_binary_labels(
        y_test_raw, normal_rul = normal_rul, anomaly_rul = anomaly_rul
    )

    X_train_raw, y_train, unit_train, X_val_raw, y_val, unit_val = split_train_val_by_unit(
        X_train_raw, y_train_full, unit_train_raw, val_split = val_split
    )
    print_split_stats("CMAPSS train", y_train)
    print_split_stats("CMAPSS val", y_val)
    print_split_stats("CMAPSS test", y_test)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)

    regime_scaler, regime_model, regime_feature_idx, regime_knn = _fit_regime_model_healthy_only(
        X_train, y_train, n_regimes=n_regimes, regime_feature_idx=regime_feature_idx,
        algorithm=algorithm, hdbscan_min_cluster_size=hdbscan_min_cluster_size,
    )

    r_train = _predict_regimes(X_train, regime_scaler, regime_model, regime_feature_idx, regime_knn)
    r_val = _predict_regimes(X_val, regime_scaler, regime_model, regime_feature_idx, regime_knn)
    r_test = _predict_regimes(X_test, regime_scaler, regime_model, regime_feature_idx, regime_knn)

    Xw_train, tw_train, yw_train, rw_train, uw_train = create_forecast_windows_grouped(
        X_train, y_train, r_train, unit_train, window_size, stride
    )
    Xw_val, tw_val, yw_val, rw_val, uw_val = create_forecast_windows_grouped(
        X_val, y_val, r_val, unit_val, window_size, stride
    )
    Xw_test, tw_test, yw_test, rw_test, uw_test = create_forecast_windows_grouped(
        X_test, y_test, r_test, unit_test, window_size, stride
    )

    train_loader = _make_loader(Xw_train, tw_train, yw_train, rw_train, batch_size, True, unit=uw_train)
    val_loader = _make_loader(Xw_val, tw_val, yw_val, rw_val, batch_size, False, unit=uw_val)
    test_loader = _make_loader(Xw_test, tw_test, yw_test, rw_test, batch_size, False, unit=uw_test)

    meta = {
        "x_scaler": x_scaler,
        "regime_scaler": regime_scaler,
        "regime_model": regime_model,
        "regime_feature_idx": regime_feature_idx,
        "regime_knn": regime_knn,
        "normal_rul": normal_rul,
        "anomaly_rul": anomaly_rul,
        "fd": fd.upper(),
        "train_rul": y_train_raw,
        "test_rul": y_test_raw,
    }
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    train_loader, val_loader, test_loader, _ = build_cmapss_dataloaders(
        fd = "FD001",
        window_size = 32,
        stride = 1,
        batch_size = 64,
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
