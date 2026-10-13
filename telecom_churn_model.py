"""
Week 3 - Predictive Modeling and Analysis for Telecom Services
Customer churn prediction on the cleaned dataset produced in Week 2.
Run:  python telecom_churn_model.py
"""
import os, json, subprocess, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, precision_recall_curve, average_precision_score)
warnings.filterwarnings("ignore")
R = {}

# ------------------------------------------------------------ 1. LOAD DATA
if not os.path.exists("telecom_clean.csv"):
    subprocess.run([sys.executable, "telecom_preprocessing.py"], check=True)   # Week 2 output
df = pd.read_csv("telecom_clean.csv")
R["shape"] = list(df.shape); R["churn_rate"] = round(float(df["churn"].mean()), 4)

numeric = ["tenure_months", "monthly_charges", "total_charges", "monthly_data_gb", "call_minutes",
           "sms_count", "support_calls", "network_latency_ms", "avg_monthly_spend"]
binary  = ["senior_citizen", "is_heavy_user"]
categ   = ["gender", "contract_type", "internet_service", "payment_method", "tenure_group"]

X = df[numeric + binary + categ].copy(); y = df["churn"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
R["split"] = [len(Xtr), len(Xte)]

def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), categ)])

# ------------------------------------------------------------ 2. CANDIDATE MODELS
def pipe(model): return Pipeline([("pre", make_pre()), ("model", model)])
models = {
    "Majority baseline": pipe(DummyClassifier(strategy="most_frequent")),
    "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, class_weight="balanced", random_state=42)),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", n_jobs=-1, random_state=42)),
    "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=42)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scoring = {"auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision", "acc": "accuracy"}
R["cv"] = {}
for n, m in models.items():
    s = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
    R["cv"][n] = {k: [round(float(s[f"test_{k}"].mean()), 3), round(float(s[f"test_{k}"].std()), 3)] for k in scoring}

# ------------------------------------------------------------ 3. HYPERPARAMETER TUNING
tune = {
 "Logistic Regression": (models["Logistic Regression"], {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
 "Random Forest": (models["Random Forest"], {"model__max_depth": [4, 6, 8, 12, None], "model__min_samples_leaf": [5, 10, 20, 40],
                                              "model__max_features": ["sqrt", 0.5]}),
 "Gradient Boosting": (models["Gradient Boosting"], {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1],
                                                     "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0]}),
}
tuned, R["tuning"] = {}, {}
for n, (m, grid) in tune.items():
    rs = RandomizedSearchCV(m, grid, n_iter=min(12, int(np.prod([len(v) for v in grid.values()]))), scoring="roc_auc",
                            cv=cv, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    tuned[n] = rs.best_estimator_
    R["tuning"][n] = {"best_params": {k.replace("model__", ""): (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in rs.best_params_.items()},
                      "cv_auc": round(float(rs.best_score_), 3)}
tuned["Decision Tree"] = models["Decision Tree"]

# ------------------------------------------------------------ 4. SELECTION BY CV (not by test set)
cand = {n: R["tuning"][n]["cv_auc"] for n in R["tuning"]}
best_name = max(cand, key=cand.get); R["selected"] = best_name
best = tuned[best_name].fit(Xtr, ytr)

# ------------------------------------------------------------ 5. TEST EVALUATION OF ALL MODELS
def metrics(yt, p, thr=.5):
    yh = (p >= thr).astype(int)
    return {"accuracy": round(accuracy_score(yt, yh), 3), "precision": round(precision_score(yt, yh), 3),
            "recall": round(recall_score(yt, yh), 3), "f1": round(f1_score(yt, yh), 3), "auc": round(roc_auc_score(yt, p), 3)}
R["test"] = {}; probs = {}
for n in ["Majority baseline", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
    m = (models[n].fit(Xtr, ytr) if n == "Majority baseline" else tuned[n].fit(Xtr, ytr))
    p = m.predict_proba(Xte)[:, 1]; probs[n] = p
    thr = .5
    R["test"][n] = metrics(yte, p, thr) if n != "Majority baseline" else {"accuracy": round(accuracy_score(yte, m.predict(Xte)), 3), "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5}

# ROC figure
fig, ax = plt.subplots(figsize=(5.6, 4.4))
for n in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
    fpr, tpr, _ = roc_curve(yte, probs[n]); ax.plot(fpr, tpr, label=f"{n} (AUC {R['test'][n]['auc']})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the test set"); ax.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.savefig("fig3_roc.png", dpi=170); plt.close()

# ------------------------------------------------------------ 6. THRESHOLD CHOICE (business cost, out-of-fold on TRAIN)
C_MISS, C_OFFER, SAVE = 500.0, 50.0, 0.4   # illustrative assumptions: value of a lost customer, cost of an offer, offer success rate
oof = cross_val_predict(tuned[best_name], Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
def net_benefit(yt, p, thr):
    flag = p >= thr; tp = ((flag) & (yt == 1)).sum(); fp = ((flag) & (yt == 0)).sum()
    return float(tp * SAVE * C_MISS - (tp + fp) * C_OFFER)
grid = np.linspace(.05, .9, 86)
nb_oof = [net_benefit(ytr.values, oof, t) for t in grid]
thr_best = float(grid[int(np.argmax(nb_oof))]); R["threshold"] = round(thr_best, 2)
R["assumptions"] = {"lost_customer_value": C_MISS, "offer_cost": C_OFFER, "offer_success_rate": SAVE}
pte = best.predict_proba(Xte)[:, 1]
R["selected_test_default"] = metrics(yte, pte, .5); R["selected_test_tuned"] = metrics(yte, pte, thr_best)
nb_te = [net_benefit(yte.values, pte, t) for t in grid]
R["net_benefit"] = {"no_campaign": 0.0, "contact_all": round(net_benefit(yte.values, np.ones(len(yte)), .5), 0),
                    "model_default_threshold": round(net_benefit(yte.values, pte, .5), 0),
                    "model_tuned_threshold": round(net_benefit(yte.values, pte, thr_best), 0)}
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(grid, np.array(nb_oof) / len(ytr), label="Training (cross-validated)"); ax.plot(grid, np.array(nb_te) / len(yte), label="Test set", alpha=.8)
ax.axvline(thr_best, color="red", ls="--", lw=1, label=f"Chosen threshold = {thr_best:.2f}")
ax.set_xlabel("Probability threshold"); ax.set_ylabel("Net benefit per customer (currency units)"); ax.set_title("Retention campaign value vs. threshold")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("fig3_threshold.png", dpi=170); plt.close()

# ------------------------------------------------------------ 7. CONFUSION MATRICES
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for a, thr, t in [(axes[0], .5, "Threshold 0.50"), (axes[1], thr_best, f"Threshold {thr_best:.2f}")]:
    cm = confusion_matrix(yte, (pte >= thr).astype(int)); a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["Stay", "Churn"]); a.set_yticklabels(["Stay", "Churn"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(t)
    R[f"cm_{t}"] = cm.tolist()
plt.tight_layout(); plt.savefig("fig3_confusion.png", dpi=170); plt.close()

# ------------------------------------------------------------ 8. INTERPRETATION
pre_fit = best.named_steps["pre"]; names = [n.split("__")[-1] for n in pre_fit.get_feature_names_out()]
lr = tuned["Logistic Regression"].fit(Xtr, ytr)
coef = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
R["lr_coef"] = coef.round(3).to_dict()
top = pd.concat([coef.head(7), coef.tail(7)])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.barh(top.index, top.values, color=["#C0504D" if v > 0 else "#4F81BD" for v in top.values])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Logistic regression coefficient (standardised features)")
ax.set_title("Strongest drivers of churn (red = raises risk)"); plt.tight_layout(); plt.savefig("fig3_coef.png", dpi=170); plt.close()

pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values()
R["perm_importance"] = imp.round(4).sort_values(ascending=False).head(8).to_dict()
fig, ax = plt.subplots(figsize=(6, 3.8)); imp.tail(9).plot.barh(ax=ax, color="#2E75B6")
ax.set_xlabel("Drop in test AUC when feature is shuffled"); ax.set_title(f"Permutation importance ({best_name})")
plt.tight_layout(); plt.savefig("fig3_perm.png", dpi=170); plt.close()

# decile lift
d = pd.DataFrame({"p": pte, "y": yte.values}); d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = d.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"))
lift["churn_rate_pct"] = (100 * lift["churners"] / lift["customers"]).round(1)
lift["lift"] = (lift["churn_rate_pct"] / (100 * yte.mean())).round(2)
R["lift"] = lift.reset_index().astype({"decile": int}).to_dict("records")
R["top3_decile_capture_pct"] = round(100 * lift["churners"].iloc[:3].sum() / lift["churners"].sum(), 1)

# segment risk
seg = df.assign(p=best.predict_proba(X)[:, 1])
R["seg_contract"] = (seg.groupby("contract_type")["p"].mean() * 100).round(1).to_dict()
R["seg_support"] = (seg.assign(sc=seg["support_calls"].clip(upper=4)).groupby("sc")["churn"].mean() * 100).round(1).to_dict()
R["seg_pay"] = (seg.groupby("payment_method")["churn"].mean() * 100).round(1).to_dict()
R["seg_tenure"] = (seg.groupby("tenure_group")["churn"].mean() * 100).round(1).to_dict()

json.dump(R, open("results3.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
