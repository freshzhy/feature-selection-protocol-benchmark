"""
data_loader.py
统一数据加载模块。对外接口：
    load_dataset(name) -> (X, y, meta)
    list_datasets()    -> 打印所有可用数据集名称
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config

# ─────────────────────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """将所有 object/string/category 列 LabelEncode 为整数。"""
    le = LabelEncoder()
    # 兼容 pandas 2.x+：同时选取 object、string、category 类型
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def _compute_top_k(n_features: int) -> int:
    """动态计算 Top-K：min(TOP_K_MAX, max(TOP_K_MIN, n_features // 2))。"""
    k = max(config.TOP_K_MIN, n_features // 2)
    return min(config.TOP_K_MAX, k)


def _make_meta(name: str, X: pd.DataFrame, y: pd.Series, source: str) -> dict:
    """构建 meta 字典，供下游实验脚本使用。"""
    n_classes = y.nunique()
    return {
        "name":      name,
        "source":    source,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes":  n_classes,
        "top_k":      _compute_top_k(X.shape[1]),
        "imbalance_ratio": round(y.value_counts().iloc[0] / len(y), 4),
    }


def _find_under(root: Path, filename: str) -> Path | None:
    """在目录（含子目录）中按文件名查找第一个匹配。"""
    hits = list(root.rglob(filename))
    return hits[0] if hits else None


# ─────────────────────────────────────────────────────────────────────────────
# OpenML
# ─────────────────────────────────────────────────────────────────────────────

def _load_openml(entry: dict):
    """加载 OpenML CSV 文件（dataset/openml/*.csv），标签列名为 target。"""
    path = config.OPENML_DIR / entry["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"[OpenML] 文件不存在: {path}\n"
            f"请确认已将 OpenML 数据集下载到 {config.OPENML_DIR}"
        )

    df = pd.read_csv(path)

    # 标签列名统一为 "target"（OpenML 下载时已设定）
    if "target" in df.columns:
        target_col = "target"
    elif "class" in df.columns:
        target_col = "class"
    else:
        target_col = df.columns[-1]

    df = _encode_categoricals(df)
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = pd.Series(
        LabelEncoder().fit_transform(df[target_col].astype(str)),
        name="target"
    )
    return X.reset_index(drop=True), y, _make_meta(entry["name"], X, y, "openml")


# ─────────────────────────────────────────────────────────────────────────────
# UCI — 优先读解压文件夹，回退到 zip
# ─────────────────────────────────────────────────────────────────────────────

def _load_uci(entry: dict):
    name     = entry["name"]
    data_dir = config.DATA_ROOT / entry.get("data_dir", "")
    zip_path = config.DATA_ROOT / entry["zip_file"]

    use_dir = data_dir.is_dir()

    if not use_dir and not zip_path.exists():
        raise FileNotFoundError(
            f"[UCI] 找不到解压目录 {data_dir} 和 zip {zip_path}"
        )

    # ── Madelon：特征文件 + 标签文件分开 ──
    if name == "madelon":
        if use_dir:
            data_file  = _find_under(data_dir, "madelon_train.data")
            label_file = _find_under(data_dir, "madelon_train.labels")
            if not data_file or not label_file:
                raise ValueError(f"[UCI/madelon] 目录内未找到 train .data/.labels: {data_dir}")
            X = pd.read_csv(data_file, sep=entry.get("sep", " "), header=None)
            y_raw = pd.read_csv(label_file, sep=entry.get("sep", " "), header=None).squeeze()
        else:
            with zipfile.ZipFile(zip_path) as zf:
                data_files  = [f for f in zf.namelist() if f.endswith(".data") and "train" in f]
                label_files = [f for f in zf.namelist() if f.endswith(".labels") and "train" in f]
                X = pd.read_csv(io.BytesIO(zf.read(data_files[0])), sep=" ", header=None)
                y_raw = pd.read_csv(io.BytesIO(zf.read(label_files[0])), sep=" ", header=None).squeeze()

        # 去掉空格分隔产生的尾部空列
        X = X.loc[:, X.notna().any()]
        y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), name="target")

    # ── 通用：单文件 CSV，标签在最后一列 ──
    else:
        csv_name = entry["csv_file"]
        if use_dir:
            match = _find_under(data_dir, csv_name)
            if match is None:
                raise ValueError(f"[UCI/{name}] 目录内未找到 {csv_name}: {data_dir}")
            raw = pd.read_csv(
                match,
                sep=entry.get("sep", ","),
                header=None if not entry.get("has_header", False) else "infer",
            )
        else:
            with zipfile.ZipFile(zip_path) as zf:
                matches = [f for f in zf.namelist() if f.endswith(csv_name)]
                raw = pd.read_csv(
                    io.BytesIO(zf.read(matches[0])),
                    sep=entry.get("sep", ","),
                    header=None if not entry.get("has_header", False) else "infer",
                )
        X = raw.iloc[:, :-1].copy()
        y_raw = raw.iloc[:, -1]
        X = _encode_categoricals(X)
        y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), name="target")

    X = X.reset_index(drop=True)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    X = X.apply(pd.to_numeric, errors="coerce").replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    y = y.reset_index(drop=True)

    return X, y, _make_meta(name, X, y, "uci")


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle — 优先读解压文件夹，回退到 zip
# ─────────────────────────────────────────────────────────────────────────────

def _load_kaggle(entry: dict):
    data_dir = config.DATA_ROOT / entry["data_dir"]
    zip_path = config.DATA_ROOT / entry["zip_file"]
    csv_file = entry["csv_file"]

    if data_dir.is_dir():
        match = _find_under(data_dir, csv_file)
        if match is None:
            raise ValueError(f"[Kaggle/{entry['name']}] 目录内未找到 {csv_file}: {data_dir}")
        df = pd.read_csv(match)
    elif zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            matches = [f for f in zf.namelist() if f.endswith(csv_file)]
            if not matches:
                raise ValueError(f"[Kaggle/{entry['name']}] zip 内未找到 {csv_file}")
            df = pd.read_csv(io.BytesIO(zf.read(matches[0])))
    else:
        raise FileNotFoundError(
            f"[Kaggle] 找不到解压目录 {data_dir} 和 zip {zip_path}"
        )

    target_col = entry["target_col"]
    if target_col not in df.columns:
        raise ValueError(
            f"[Kaggle/{entry['name']}] 标签列 '{target_col}' 不存在。"
            f"实际列名: {list(df.columns)}"
        )

    # 删除 ID 类列（列名含 id 且值唯一）
    id_cols = [c for c in df.columns
               if "id" in c.lower() and df[c].nunique() == len(df)]
    df = df.drop(columns=id_cols)

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col]

    X = _encode_categoricals(X)
    X = X.apply(pd.to_numeric, errors="coerce").replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), name="target")

    return X.reset_index(drop=True), y.reset_index(drop=True), \
           _make_meta(entry["name"], X, y, "kaggle")


# ─────────────────────────────────────────────────────────────────────────────
# NIDS — 读主 CSV（跳过特征说明表），分层采样 100K 行
# ─────────────────────────────────────────────────────────────────────────────

def _load_nids(entry: dict):
    data_dir = config.NIDS_DIR / entry["data_dir"]
    if not data_dir.exists():
        raise FileNotFoundError(f"[NIDS] 目录不存在: {data_dir}")

    # 只读主数据 CSV，跳过 NetFlow_v2_Features.csv 等非数据文件
    csv_files = sorted(
        f for f in data_dir.glob("*.csv")
        if f.name not in config.NIDS_SKIP_FILES
    )
    if not csv_files:
        raise FileNotFoundError(f"[NIDS] {data_dir} 下没有数据 CSV 文件")

    chunks = []
    for f in csv_files:
        try:
            chunks.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  [警告] 读取 {f.name} 失败: {e}")

    df = pd.concat(chunks, ignore_index=True)

    target_col = entry["target_col"]
    if target_col not in df.columns:
        raise ValueError(
            f"[NIDS/{entry['name']}] 标签列 '{target_col}' 不存在。\n"
            f"实际列名: {list(df.columns)}"
        )

    # 删去元数据列（IP地址、端口、Attack 等）—— Label 不在其中！
    drop_cols = [c for c in config.NIDS_DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=[target_col])

    # 分层采样
    sample_n = entry["sample_n"]
    if len(df) > sample_n:
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(
            df,
            test_size=sample_n / len(df),
            random_state=config.RANDOM_SEED,
            stratify=df[target_col],
        )

    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col]

    X = _encode_categoricals(X)
    X = X.apply(pd.to_numeric, errors="coerce").replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), name="target")

    return X.reset_index(drop=True), y.reset_index(drop=True), \
           _make_meta(entry["name"], X, y, "nids")


# ─────────────────────────────────────────────────────────────────────────────
# 路由表
# ─────────────────────────────────────────────────────────────────────────────

def _build_registry() -> dict:
    registry = {}
    for entry in config.OPENML_DATASETS:
        registry[entry["name"]] = (_load_openml, entry)
    for entry in config.UCI_DATASETS:
        registry[entry["name"]] = (_load_uci, entry)
    for entry in config.KAGGLE_DATASETS:
        registry[entry["name"]] = (_load_kaggle, entry)
    for entry in config.NIDS_DATASETS:
        registry[entry["name"]] = (_load_nids, entry)
    return registry

_REGISTRY = _build_registry()


# ─────────────────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str):
    """
    加载指定数据集，返回 (X, y, meta)。

    Parameters
    ----------
    name : str — 数据集名称，用 list_datasets() 查看。

    Returns
    -------
    X    : pd.DataFrame — 特征矩阵（已编码、无缺失值）
    y    : pd.Series    — 标签（0-based 整数编码）
    meta : dict         — 元信息
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"未知数据集: '{name}'。可用: {list(_REGISTRY.keys())}"
        )
    loader_fn, entry = _REGISTRY[name]
    X, y, meta = loader_fn(entry)
    print(f"[load_dataset] {name}: {meta['n_samples']} 样本 × "
          f"{meta['n_features']} 特征，{meta['n_classes']} 类，"
          f"top_k={meta['top_k']}")
    return X, y, meta


def list_datasets() -> list:
    """打印并返回所有可用数据集名称。"""
    names = list(_REGISTRY.keys())
    print("=" * 50)
    print(f"共 {len(names)} 个数据集：")
    sources = {"openml": [], "uci": [], "kaggle": [], "nids": []}
    for entry_list, src in [
        (config.OPENML_DATASETS,  "openml"),
        (config.UCI_DATASETS,     "uci"),
        (config.KAGGLE_DATASETS,  "kaggle"),
        (config.NIDS_DATASETS,    "nids"),
    ]:
        for e in entry_list:
            sources[src].append(e["name"])
    for src, ns in sources.items():
        print(f"  [{src.upper()}] {ns}")
    print("=" * 50)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    list_datasets()

    all_names = list(_REGISTRY.keys())
    errors = []
    for name in all_names:
        try:
            X, y, meta = load_dataset(name)
            assert len(X) == len(y), "X 和 y 行数不一致！"
            assert X.isnull().sum().sum() == 0, "X 中仍有缺失值！"
        except Exception as e:
            errors.append((name, str(e)))
            print(f"  [ERROR] {name}: {e}")

    print("\n" + "=" * 50)
    if errors:
        print(f"加载失败 {len(errors)} 个数据集：")
        for name, err in errors:
            print(f"  {name}: {err}")
    else:
        print("全部数据集加载成功 ✓")
