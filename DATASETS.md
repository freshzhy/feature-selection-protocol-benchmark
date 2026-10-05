# Datasets: provenance and how to obtain them

No dataset files are included. Place the files under `dataset/` with the layout that
`config.py` expects. SHA-256 prefixes (first 16 hex digits) identify the exact files used.

## OpenML (14), fetched with `download_openml.py` into `dataset/openml/`

| Name | OpenML ID | File | SHA-256 |
|---|---|---|---|
| adult | 1590 | adult.csv | 32b09d94a593bf92 |
| banknote-authentication | 1462 | banknote-authentication.csv | 87982e488585ebba |
| blood-transfusion | 1464 | blood-transfusion-service-center.csv | 4f6891b04dc618b2 |
| breast-w | 15 | breast-w.csv | c8fe479e5cf30a67 |
| car | 40975 | car.csv | 64bd1117a99053e6 |
| credit-approval | 29 | credit-approval.csv | 6d0cbea3cc4a36fb |
| credit-g | 31 | credit-g.csv | a847ce53bb4d8147 |
| diabetes | 37 | diabetes.csv | 75e0f254b2ddd212 |
| kc1 | 1063 | kc1.csv | 85c924b80c7d0bf4 |
| kc2 | 1067 | kc2.csv | f1776a3654b6308f |
| kr-vs-kp | 3 | kr-vs-kp.csv | 8ffb28f847baf18b |
| phoneme | 1489 | phoneme.csv | 650f902de961d61b |
| qsar-biodeg | 1494 | qsar-biodeg.csv | 57ef0901ee7e6461 |
| steel-plates-fault | 40982 | steel-plates-fault.csv | f579ad0c6b5386c4 |

## UCI Machine Learning Repository (4)

| Name | Directory under dataset/ | File(s) | SHA-256 |
|---|---|---|---|
| madelon | madelon/ | madelon_train.data, madelon_train.labels (training split, 2,000 rows) | 0b6e37711efdc7ab, 10bffcc3b017467c |
| ionosphere | ionosphere/ | ionosphere.data | 46d52186b84e20be |
| sonar | connectionist+bench+sonar+mines+vs+rocks/ | sonar.all-data | e90434cdbf00fcf9 |
| spambase | spambase/ | spambase.data | b1ef93de71f97714 |

Madelon's relevant columns are not published (the column order is randomised). The
manuscript uses the 20-feature putative relevant set identified by
`madelon_putative_set.py` (f28, f48, f64, f105, f128, f153, f241, f281, f318, f336,
f338, f378, f433, f442, f451, f453, f455, f472, f475, f493). An earlier version of
this study wrongly assumed f0–f19.

## Kaggle (3)

| Name | Directory under dataset/ | CSV | Label | SHA-256 |
|---|---|---|---|---|
| creditcard | Credit Card Fraud Detection/ | creditcard.csv | Class | 76274b691b16a6c4 |
| telco-churn | Telco Customer Churn/ | WA_Fn-UseC_-Telco-Customer-Churn.csv | Churn | 88be4b93fbe0cc83 |
| hr-attrition | HR Analytics - Employee Attrition/ | WA_Fn-UseC_-HR-Employee-Attrition.csv | Attrition | a5c31e38bd7fafc9 |

## NIDS, NetFlow v2 (4), cross-dataset transfer only

NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2 and NF-BoT-IoT-v2 from the
University of Queensland NIDS datasets page (Sarhan, Layeghy and Portmann, Mobile
Networks and Applications 27:357–370, 2022, doi:10.1007/s11036-021-01843-0), each in
`dataset/<name>/data/`. `data_loader.py` drops `Attack`, the source/destination
addresses and ports (39 numeric features remain), draws a stratified sample of
100,000 flows (seed 42), sets infinite values to missing and missing values to zero.

## Preprocessing common to all general datasets

Categorical features are integer-coded per original variable (no one-hot expansion);
missing categories form their own level; rows with missing numeric values are dropped
(Breast-W: 16 rows, Credit-Approval: 24 rows). No preprocessing step uses the target,
and every scaler and selector is fitted on the data it receives (the outer training
fold under the nested protocol).
