# %% [markdown]
# ## A3.4 Metrics and the cost of errors
#
# - On a 69% malignant dataset, I think plain accuracy says something nobody would defend.
# - I show it in two steps: (a) score predictors that know nothing, (b) put an explicit price on each error and re-rank them.
# - Everything uses the **lesion-level test split** with the 3-class `diagnosis_1`, from `splits.split_lesions`.

# %%
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

CLASSES = config.PRIMARY_LABELS  # ["Benign", "Indeterminate", "Malignant"]

lesions = data.build_lesion_table()
train_ids, val_ids, test_ids = splits.split_lesions(lesions, seed=config.SEED)
# A metric computed on a leaked split measures nothing, so the partition is checked
# before any number below is trusted.
assert splits.verify_splits(lesions, train_ids, val_ids, test_ids)["partition_ok"]

train = lesions[lesions["lesion_id"].isin(train_ids)]
test = lesions[lesions["lesion_id"].isin(test_ids)]
y_train = train[config.PRIMARY_LABEL_COL].to_numpy()
y_test = test[config.PRIMARY_LABEL_COL].to_numpy()

# The dummies look at no features whatsoever; sklearn still requires an X of the right
# length, so a column of zeros is the honest placeholder for "no input is used".
X_train = np.zeros((len(y_train), 1))
X_test = np.zeros((len(y_test), 1))

test_counts = pd.Series(y_test).value_counts().reindex(CLASSES)
print(f"train lesions {len(y_train):,} | test lesions {len(y_test):,}")
print(pd.DataFrame({"n": test_counts, "share": test_counts / len(y_test)}).round(3))

# %% [markdown]
# ### (a) Three predictors that know nothing
#
# - `DummyClassifier` fitted on train, evaluated on test.
# - `random_state = config.SEED`, so the random ones give the same result every time.

# %%
DUMMIES = {
    "always-Malignant": dict(strategy="constant", constant="Malignant"),
    "stratified-random": dict(strategy="stratified"),
    "uniform-random": dict(strategy="uniform"),
}


def fit_dummy(**kwargs) -> np.ndarray:
    """Fit a dummy on the train labels and return its predictions for the test set."""
    model = DummyClassifier(random_state=config.SEED, **kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def score_row(name: str, y_pred: np.ndarray) -> dict:
    # zero_division=0: a constant predictor never predicts two of the three classes,
    # so their precision is 0/0. Scoring that as 0 is the point, not a nuisance.
    return {
        "predictor": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, labels=CLASSES, average="macro",
                             zero_division=0),
    }


predictions = {name: fit_dummy(**kw) for name, kw in DUMMIES.items()}
scores = pd.DataFrame([score_row(n, p) for n, p in predictions.items()])
print(scores.set_index("predictor").round(3))

# %%
cm_stratified = pd.DataFrame(
    confusion_matrix(y_test, predictions["stratified-random"], labels=CLASSES),
    index=pd.Index(CLASSES, name="true"),
    columns=pd.Index(CLASSES, name="predicted"),
)
print(cm_stratified)

fig, ax = plt.subplots(figsize=(5.4, 4.6))
ConfusionMatrixDisplay(cm_stratified.to_numpy(), display_labels=CLASSES).plot(
    ax=ax, cmap="Oranges", colorbar=False, values_format="d"
)
# The project style turns gridlines on globally; on a heatmap they only add noise.
ax.grid(False)
ax.set_title("stratified-random dummy, test lesions")
plt.show()

# %% [markdown]
# - **Always-Malignant gets 0.694 accuracy.** That looks like a working model, but it's one hard-coded answer.
# - Its balanced accuracy is 1/3 = 0.333 (free for 3 classes) and macro-F1 is 0.273.
# - The accuracy only reflects that 69.4% of test lesions are malignant.
# - The stratified-random confusion matrix: 0.557 accuracy, with **164** malignant lesions predicted as Benign.
# - That's why I report **balanced accuracy and macro-F1**, not accuracy alone.

# %% [markdown]
# ### (b) An explicit cost matrix
#
# Rows = truth, columns = prediction:
#
# - **true Malignant, predicted anything else = 50**: missed cancer, it doesn't get removed
# - **true Benign, predicted anything else = 1**: one unnecessary biopsy
# - **true Indeterminate, predicted anything else = 5**: worse than a false alarm, better than a missed cancer
# - **correct = 0**
#
# One simplification I keep from the brief: cost depends on the *true* class, so Malignant predicted as Indeterminate also costs 50, even though in real life "indeterminate" would still lead to follow-up.

# %%
MISS_MALIGNANT = 50.0        # truth Malignant called anything else
FALSE_ALARM = 1.0            # truth Benign called anything else
INDETERMINATE_ERROR = 5.0    # truth Indeterminate called anything else

_ERROR_COST = {"Malignant": MISS_MALIGNANT, "Benign": FALSE_ALARM,
               "Indeterminate": INDETERMINATE_ERROR}

cost_matrix = pd.DataFrame(
    [[0.0 if t == p else _ERROR_COST[t] for p in CLASSES] for t in CLASSES],
    index=pd.Index(CLASSES, name="true"),
    columns=pd.Index(CLASSES, name="predicted"),
)
print(cost_matrix)


def expected_cost(y_pred: np.ndarray) -> float:
    """Mean cost per lesion: the confusion matrix weighted cell-by-cell by the costs."""
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    return float((cm * cost_matrix.to_numpy()).sum() / len(y_test))


print(f"\nasymmetry: a missed malignancy costs "
      f"{MISS_MALIGNANT / FALSE_ALARM:.0f}x an unnecessary biopsy")

# %% [markdown]
# - **The 50:1 ratio is a clinical decision, not something the data tells me.**
# - It means "50 unnecessary biopsies are worth catching one more melanoma".
# - I think a dermatologist should set this number, since thresholds, class weights and model choice all follow from it.

# %%
# The fourth predictor the brief asks for, plus the third constant, so that "which
# constant is optimal" is answered over all three and not just two of them.
predictions["always-Benign"] = fit_dummy(strategy="constant", constant="Benign")
predictions["always-Indeterminate"] = fit_dummy(strategy="constant",
                                                constant="Indeterminate")

summary = pd.DataFrame([score_row(n, p) for n, p in predictions.items()])
summary = summary.set_index("predictor")
summary["cost_per_lesion"] = [expected_cost(predictions[n]) for n in summary.index]
# method="min" so the three predictors tied at balanced accuracy 1/3 share a rank
# instead of being separated by an averaging artefact.
summary["rank_accuracy"] = summary["accuracy"].rank(ascending=False, method="min").astype(int)
summary["rank_cost"] = summary["cost_per_lesion"].rank(ascending=True, method="min").astype(int)
summary["rank_balanced_acc"] = summary["balanced_accuracy"].rank(
    ascending=False, method="min").astype(int)
print(summary.round(3))

# %% [markdown]
# ### (c) Does the accuracy ranking agree with the cost ranking?

# %%
best_constant = summary.loc[
    [n for n in summary.index if n.startswith("always-")], "cost_per_lesion"
].idxmin()
acc_order = summary.sort_values(by="accuracy", ascending=False).index.tolist()
cost_order = summary.sort_values(by="cost_per_lesion").index.tolist()
bal_order = summary.sort_values(by="balanced_accuracy", ascending=False).index.tolist()

print("cheapest constant     :", best_constant,
      f"({summary.loc[best_constant, 'cost_per_lesion']:.2f} per lesion)")
print("by accuracy       :", " > ".join(acc_order))
print("by cost (cheapest):", " < ".join(cost_order))
print("by balanced acc   :", " > ".join(bal_order))
print("accuracy ranking == cost ranking        :", acc_order == cost_order)
print("balanced-accuracy ranking == cost ranking:", bal_order == cost_order)

ratio = (summary.loc["always-Benign", "cost_per_lesion"]
         / summary.loc["always-Malignant", "cost_per_lesion"])
print(f"\nalways-Benign costs {ratio:.0f}x always-Malignant, "
      f"while scoring the identical balanced accuracy "
      f"({summary.loc['always-Benign', 'balanced_accuracy']:.3f}).")

# %%
# The rankings above happen to agree on accuracy. That agreement is a property of these
# five predictors, not a law, and it is worth showing that it breaks — so here is a
# predictor constructed FROM the true test labels: correct on everything except that it
# lets roughly a quarter of the malignancies through as "Benign". It is not a model
# anyone could fit (it peeks at the answer); it is an existence proof that a predictor
# can beat another on accuracy and be far worse on cost.
rng = np.random.default_rng(config.SEED)
is_malignant = y_test == "Malignant"
missed = is_malignant & (rng.random(len(y_test)) < 0.25)
y_leaky = np.where(missed, "Benign", y_test)

rows = pd.DataFrame([
    {"predictor": "always-Malignant",
     "accuracy": summary.loc["always-Malignant", "accuracy"],
     "cost_per_lesion": summary.loc["always-Malignant", "cost_per_lesion"]},
    {"predictor": "misses 25% of malignancies",
     "accuracy": accuracy_score(y_test, y_leaky),
     "cost_per_lesion": expected_cost(y_leaky)},
]).set_index("predictor")
print(rows.round(3))
print(f"\n{int(missed.sum())} missed malignancies buy "
      f"{rows.loc['misses 25% of malignancies', 'accuracy'] - rows.loc['always-Malignant', 'accuracy']:+.3f} "
      f"accuracy and cost "
      f"{rows.loc['misses 25% of malignancies', 'cost_per_lesion'] / rows.loc['always-Malignant', 'cost_per_lesion']:.0f}x more.")

# %% [markdown]
# **My answer**
#
# - **Best constant under my costs: always-Malignant, 0.39 per lesion.** It never misses a cancer, so it only pays for false alarms: 224 benign x 1 + 17 indeterminate x 5 = 309 over 787 lesions.
# - **Always-Benign: 34.80 per lesion (89x worse)**, because it misses all 546 malignant lesions. Always-Indeterminate is 34.97.
# - **Accuracy vs cost:** on these predictors they happen to agree, but from what I understood that's only because 69.4% are malignant. My constructed example beats always-Malignant on accuracy (0.813 vs 0.694) but costs 24x more, since it sends 147 cancers home.
# - **The real disagreement is with balanced accuracy / macro-F1:**
#   - uniform-random (0.369) and stratified-random (0.363) rank *above* always-Malignant (0.333) on balanced accuracy, and macro-F1 ties uniform-random with it (0.273 each)
#   - yet always-Malignant is 29x and 60x cheaper than those two
#   - always-Benign and always-Malignant have the *same* balanced accuracy (0.333) but differ 89x in cost
# - **My conclusion:** the metric I optimise is a clinical value judgement, so I choose it on purpose. Like in A1.4: malignant sensitivity at a fixed specificity, plus balanced accuracy, macro-F1 and the confusion matrix, with the cost matrix as a tie-breaker.
# - **Caveat:** "always-Malignant is cheapest" only holds because the dataset is 69.4% malignant. In a normal clinic where most lesions are benign, removing everything would be far too expensive.

