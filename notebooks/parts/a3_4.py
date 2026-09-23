# %% [markdown]
# ## A3.4 Metrics and the cost of errors
#
# A metric is not a neutral measuring device. It is a statement about which mistakes
# matter, and on a 69% malignant dataset the default metric — accuracy — makes a
# statement nobody would defend out loud. This section proves that on the project's own
# test split: first by scoring predictors that contain no information at all, then by
# writing down an explicit cost matrix and re-ranking the same predictors under it.
#
# Everything runs on the **lesion-level test split** with the 3-class `diagnosis_1`
# target. The split comes from `splits.split_lesions`, so it is the same partition the
# rest of the project uses, and the two images of a lesion can never straddle it.

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
# `DummyClassifier` is fitted on the training labels and evaluated on the test labels.
# `random_state` is fixed at `config.SEED` so the two random strategies give the same
# table on every rerun.

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
# **This is the whole reason the exercise exists.** Always-Malignant scores an accuracy
# of 0.694 — a number that looks like a working classifier and would pass unexamined in
# a slide deck. It is a single hard-coded string. Its balanced accuracy is exactly
# 1/3 = 0.333, the value a three-class predictor gets for free, and its macro-F1 is
# 0.273 because two of its three per-class F1 scores are 0. The accuracy is high only
# because 69.4% of the test lesions are malignant; the metric is reporting the class
# prior, not the model.
#
# The stratified-random confusion matrix shows the same thing from the other side: it
# spreads its guesses over all three classes in the training proportions, gets 0.557
# accuracy, and still has 164 malignant lesions sitting in the "predicted Benign" cell.
# That is why this project reports **balanced accuracy and macro-F1**, never accuracy
# alone — both of those metrics refuse to reward a predictor for knowing the prior.

# %% [markdown]
# ### (b) An explicit cost matrix
#
# Balanced accuracy and macro-F1 fix the prior problem but still treat all three
# classes as equally valuable, which no dermatologist does. So the errors get an
# explicit price, in the brief's units. Rows are the truth, columns are the prediction:
#
# * **row Malignant, any other column = 50.** A malignancy that the system did not call
#   malignant. The lesion is not excised; the cost is a delayed cancer diagnosis.
# * **row Benign, any other column = 1.** A benign lesion routed to work-up: one
#   unnecessary biopsy, some patient anxiety, a modest bill.
# * **row Indeterminate, any other column = 5.** An indeterminate lesion mis-triaged in
#   either direction — worse than a false alarm, better than a missed cancer.
# * **diagonal = 0.** Correct predictions are free.
#
# One simplification has to be stated rather than hidden: the cost is charged by the
# **true** class, so truth-Malignant / predicted-Indeterminate also costs 50, even
# though in a real clinic an "indeterminate" call still triggers follow-up and would be
# cheaper than a confident "benign". This matrix is therefore slightly pessimistic about
# that one cell; the brief's scheme is kept as given so the numbers stay checkable.

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
# **The asymmetry is a clinical judgement, not a mathematical one.** Nothing in the data
# implies 50:1. That ratio says a clinician would accept fifty unnecessary biopsies to
# catch one extra melanoma, and it is the single number a domain expert should be asked
# to set — because every threshold, every class weight and every "best" model downstream
# is a consequence of it.

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
# **Answer.** Under this cost matrix the optimal constant is **always-Malignant, at 0.39
# per lesion**. It never misses a malignancy, so it pays only the cheap false-alarm
# price: 224 benign lesions x 1 plus 17 indeterminate x 5 = 309 over 787 lesions.
# Always-Benign is catastrophic at **34.80 per lesion**, 89x worse, because it hands
# back all 546 malignancies at 50 each — and it is exactly the kind of predictor that
# looks reasonable on paper, with a confusion matrix full of correct benign calls.
# Always-Indeterminate is worse still at 34.97.
#
# Do the rankings agree? **On accuracy, on these five predictors, yes — and that
# agreement is a coincidence of this dataset, not a reassurance.** With 69.4% malignant
# prevalence, the constant that maximises accuracy is also the one that avoids the
# expensive error, so the two orderings coincide. The moment a predictor trades
# malignant recall for overall correctness the agreement breaks: the constructed
# predictor above beats always-Malignant on accuracy, 0.813 against 0.694, and costs
# 24x more per lesion - 147 malignancies sent home.
#
# The disagreement that is *already present* in the table is with the class-balanced
# metrics this project actually uses. Balanced accuracy ranks uniform-random (0.369) and
# stratified-random (0.363) **above** always-Malignant (0.333), and macro-F1 ranks
# stratified-random above it and uniform-random level with it (0.273 each) — yet
# always-Malignant is 29x and 60x cheaper than those two. Worse, always-Benign and always-Malignant have the *identical*
# balanced accuracy of 0.333 while differing by a factor of 89 in cost. So balanced
# accuracy fixes the prior problem of part (a) and is still blind to the asymmetry of
# part (b). No single scalar covers both.
#
# The conclusion the project needs: **the metric you optimise encodes a clinical value
# judgement, so it has to be chosen deliberately and stated, not inherited from a
# library default.** Consistent with the framing in A1.4, this project optimises malignant
# sensitivity at a fixed specificity, reports balanced accuracy and macro-F1 beside it
# together with the full confusion matrix, and uses the cost matrix as the tie-breaker
# when two models are close.
#
# The honest caveat: "always-Malignant is cost-optimal" is an artefact of a 69.4%
# malignant dataset combined with this 50:1 ratio. MILK10k is a curated biopsy-referral
# archive, not a population. In a primary-care clinic where a large majority of lesions
# are benign, the same matrix would make blanket excision ruinously expensive and would
# favour a completely different operating point. That is precisely why the deployment
# population has to be stated before any metric is quoted.
