"""
config.py
全局配置：路径、数据集元信息、实验超参数
其他所有模块均从这里 import，不在代码中写死路径或参数。
"""

from pathlib import Path

# ─────────────────────────────────────────────
# 1. 项目根目录（config.py 所在位置）
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
# 2. 数据集根目录
# ─────────────────────────────────────────────
DATA_ROOT      = ROOT / "dataset"
OPENML_DIR     = DATA_ROOT / "openml"          # 14 个 OpenML CSV
NIDS_DIR       = DATA_ROOT                     # NF-* 子目录在 dataset/ 下
RESULTS_DIR    = ROOT / "results"              # 实验结果输出
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 3. OpenML 数据集元信息
#    标签列名均为 "target"（OpenML 下载时统一命名）
# ─────────────────────────────────────────────
OPENML_DATASETS = [
    {"name": "diabetes",                    "file": "diabetes.csv"},
    {"name": "credit-g",                    "file": "credit-g.csv"},
    {"name": "breast-w",                    "file": "breast-w.csv"},
    {"name": "credit-approval",             "file": "credit-approval.csv"},
    {"name": "kr-vs-kp",                    "file": "kr-vs-kp.csv"},
    {"name": "banknote-authentication",     "file": "banknote-authentication.csv"},
    {"name": "blood-transfusion",           "file": "blood-transfusion-service-center.csv"},
    {"name": "phoneme",                     "file": "phoneme.csv"},
    {"name": "qsar-biodeg",                 "file": "qsar-biodeg.csv"},
    {"name": "steel-plates-fault",          "file": "steel-plates-fault.csv"},
    {"name": "car",                         "file": "car.csv"},
    {"name": "adult",                       "file": "adult.csv"},
    {"name": "kc1",                         "file": "kc1.csv"},
    {"name": "kc2",                         "file": "kc2.csv"},
]

# ─────────────────────────────────────────────
# 4. UCI 数据集元信息
#    data_dir: 解压后文件夹名（相对于 DATA_ROOT）
#    csv_file: 主数据文件名
# ─────────────────────────────────────────────
UCI_DATASETS = [
    {
        "name":       "madelon",
        "data_dir":   "madelon",                # dataset/madelon/
        "zip_file":   "madelon.zip",             # 回退用
        "sep":        " ",
    },
    {
        "name":       "ionosphere",
        "data_dir":   "ionosphere",
        "zip_file":   "ionosphere.zip",
        "csv_file":   "ionosphere.data",
        "target_col": -1,
        "sep":        ",",
        "has_header": False,
    },
    {
        "name":       "sonar",
        "data_dir":   "connectionist+bench+sonar+mines+vs+rocks",
        "zip_file":   "connectionist+bench+sonar+mines+vs+rocks.zip",
        "csv_file":   "sonar.all-data",
        "target_col": -1,
        "sep":        ",",
        "has_header": False,
    },
    {
        "name":       "spambase",
        "data_dir":   "spambase",
        "zip_file":   "spambase.zip",
        "csv_file":   "spambase.data",
        "target_col": -1,
        "sep":        ",",
        "has_header": False,
    },
]

# ─────────────────────────────────────────────
# 5. Kaggle 数据集元信息
#    data_dir: 解压后文件夹名（含空格，与原 zip 同名去掉 .zip）
# ─────────────────────────────────────────────
KAGGLE_DATASETS = [
    {
        "name":       "creditcard",
        "data_dir":   "Credit Card Fraud Detection",
        "zip_file":   "Credit Card Fraud Detection.zip",
        "csv_file":   "creditcard.csv",
        "target_col": "Class",
    },
    {
        "name":       "telco-churn",
        "data_dir":   "Telco Customer Churn",
        "zip_file":   "Telco Customer Churn.zip",
        "csv_file":   "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "target_col": "Churn",
    },
    {
        "name":       "hr-attrition",
        "data_dir":   "HR Analytics - Employee Attrition",
        "zip_file":   "HR Analytics - Employee Attrition.zip",
        "csv_file":   "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "target_col": "Attrition",
    },
]

# ─────────────────────────────────────────────
# 6. NIDS 数据集元信息（用于实验4跨数据集迁移）
# ─────────────────────────────────────────────
NIDS_DATASETS = [
    {
        "name":         "UNSW-NB15",
        "data_dir":     "NF-UNSW-NB15-v2/data",
        "target_col":   "Label",
        "sample_n":     100_000,
    },
    {
        "name":         "CIC-IDS2018",
        "data_dir":     "NF-CSE-CIC-IDS2018-v2/data",
        "target_col":   "Label",
        "sample_n":     100_000,
    },
    {
        "name":         "ToN-IoT",
        "data_dir":     "NF-ToN-IoT-v2/data",
        "target_col":   "Label",
        "sample_n":     100_000,
    },
    {
        "name":         "BoT-IoT",
        "data_dir":     "NF-BoT-IoT-v2/data",
        "target_col":   "Label",
        "sample_n":     100_000,
    },
]

# NIDS: 需要删除的元数据列（IP地址、端口、攻击类型标签）
# ★ Label 不在这里！Label 是标签列，由 target_col 指定，不能删。
NIDS_DROP_COLS = ["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                  "L4_SRC_PORT", "L4_DST_PORT"]

# NIDS data/ 目录下需要跳过的非数据文件（特征说明表，不是流量记录）
NIDS_SKIP_FILES = {"NetFlow_v2_Features.csv"}

# ─────────────────────────────────────────────
# 7. 特征选择实验通用参数
# ─────────────────────────────────────────────
TOP_K_MIN = 3
TOP_K_MAX = 20
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 8. 下游分类器（XGBoost）固定超参数
# ─────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators  = 200,
    max_depth     = 6,
    learning_rate = 0.1,
    random_state  = RANDOM_SEED,
    n_jobs        = -1,
    eval_metric   = "mlogloss",
    verbosity     = 0,
)

# ─────────────────────────────────────────────
# 9. 实验2：Bootstrap 稳定性
# ─────────────────────────────────────────────
BOOTSTRAP_N       = 50
BOOTSTRAP_RATIO   = 0.8

STABILITY_DATASETS = [
    "diabetes", "credit-g", "qsar-biodeg", "kc1",
    "madelon",  "ionosphere", "spambase", "sonar",
    "steel-plates-fault", "adult",
]

# ─────────────────────────────────────────────
# 10. 实验3：噪声注入
# ─────────────────────────────────────────────
NOISE_RATIOS = [0.1, 0.2, 0.5]

# ─────────────────────────────────────────────
# 11. 交叉验证
# ─────────────────────────────────────────────
CV_FOLDS = 5
