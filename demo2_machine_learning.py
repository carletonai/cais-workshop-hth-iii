"""
DEMO 2 - Machine learning: let the computer learn the rules from data

Builds on demo 1: same patients, same question, and at the end we race
demo 1's hand-written rules against two classic ML models:
    * a decision tree       learns its own IF-ELSE rules from the data
    * logistic regression   predict -> measure the loss -> nudge the weights -> repeat
                            (it's a single artificial neuron: the bridge to demo 3)

Run:   python demo2_machine_learning.py              (press Enter to step through)
       python demo2_machine_learning.py --no-pause
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from demo1_rule_based import predict_with_rules
from heart_data import (
    LABEL_COLUMN,
    LEAKY_COLUMN,
    describe,
    feature_columns,
    load_patients,
    pause,
    print_step,
    print_title,
    score,
    split_patients,
    split_train_validation,
)

# Decision tree hyperparameters
TREE_DEPTHS = list(range(1, 11))   # how many questions deep the tree is allowed to go
TUNING_REPEATS = 30                # train/validation reshuffles to average over (test stays locked away)

# Logistic regression hyperparameters
LEARNING_RATE = 0.1                # how big each nudge to the weights is
TRAINING_STEPS = 400
L2_STRENGTH = 0.01                 # regularization: a penalty on big weights, to fight overfitting

RACE_REPEATS = 50                  # reruns of the final race on different shuffles of the data

CAIS_RED = "#E2383F"
CHARCOAL = "#2B2F36"


# ---------------------------------------------------------------- decision tree

def tree_as_python(tree_model, feature_names):
    """Turn a trained decision tree back into if/else code, just like demo 1's rules."""
    tree = tree_model.tree_
    code_lines = ["def learned_rules(patient):"]

    def outcomes_below(node):
        if tree.children_left[node] == -1:  # leaf
            return {int(np.argmax(tree.value[node][0]))}
        return outcomes_below(tree.children_left[node]) | outcomes_below(tree.children_right[node])

    def write(node, indent_level):
        indent = "    " * indent_level
        outcomes = outcomes_below(node)
        if len(outcomes) == 1:  # every branch below gives the same answer, so just return it
            outcome = outcomes.pop()
            code_lines.append(f"{indent}return {outcome}   # {'HIGH RISK' if outcome else 'low risk'}")
            return
        feature = feature_names[tree.feature[node]]
        code_lines.append(f'{indent}if patient["{feature}"] <= {tree.threshold[node]:g}:')
        write(tree.children_left[node], indent_level + 1)
        code_lines.append(f"{indent}else:")
        write(tree.children_right[node], indent_level + 1)

    write(0, 1)
    return "\n".join(code_lines)


def make_tree(depth):
    # class_weight="balanced": deaths are rarer (~1 in 3), so each one counts for more.
    # Without it, the tree drifts toward demo 1's lazy "always say survive".
    return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0)


# ---------------------------------------------------------- logistic regression

def sigmoid(logits):
    return 1 / (1 + np.exp(-logits))


def balanced_sample_weights(labels):
    """Same idea as class_weight="balanced": make each death count for more in the loss."""
    num_patients = len(labels)
    num_deaths = labels.sum()
    death_weight = num_patients / (2 * num_deaths)
    survivor_weight = num_patients / (2 * (num_patients - num_deaths))
    return np.where(labels == 1, death_weight, survivor_weight)


def train_logistic_regression(features_train, labels_train, verbose=False):
    num_patients, num_features = features_train.shape
    weights = np.zeros(num_features)
    bias = 0.0
    sample_weights = balanced_sample_weights(labels_train)
    loss_history = []

    for step in range(TRAINING_STEPS):
        # 1. PREDICT: a probability of death for every training patient
        probability_of_death = sigmoid(features_train @ weights + bias)

        # 2. LOSS: how wrong were we? (binary cross-entropy, plus the L2 penalty)
        clipped = np.clip(probability_of_death, 1e-12, 1 - 1e-12)
        per_patient_loss = -(labels_train * np.log(clipped) + (1 - labels_train) * np.log(1 - clipped))
        loss = np.mean(sample_weights * per_patient_loss) + L2_STRENGTH * np.sum(weights ** 2)
        loss_history.append(loss)

        # 3. MINIMIZE: gradient descent, nudge every weight a little bit downhill
        prediction_error = sample_weights * (probability_of_death - labels_train)
        weights_gradient = features_train.T @ prediction_error / num_patients + 2 * L2_STRENGTH * weights
        bias_gradient = prediction_error.mean()
        weights -= LEARNING_RATE * weights_gradient
        bias -= LEARNING_RATE * bias_gradient

        if verbose and (step in (0, 10, 25, 50, 100, 200) or step == TRAINING_STEPS - 1):
            print(f"    step {step:4d}    loss {loss:.4f}")

    return weights, bias, loss_history


def fit_logistic_regression(train, features, verbose=False):
    # Scale every feature to mean 0, std 1 using ONLY the training patients' stats
    # (platelets are in the 100,000s while ejection fraction is ~40; unscaled, gradient descent crawls).
    feature_means = train[features].mean()
    feature_stds = train[features].std()

    def scaled(split):
        return ((split[features] - feature_means) / feature_stds).to_numpy()

    weights, bias, loss_history = train_logistic_regression(
        scaled(train), train[LABEL_COLUMN].to_numpy(), verbose=verbose)

    def predict(split):
        return (sigmoid(scaled(split) @ weights + bias) >= 0.5).astype(int)

    return predict, weights, loss_history


# -------------------------------------------------------------------- the race

def contenders(test, features, tree_model, predict_logistic_regression):
    return [
        ("'always say survive'", np.zeros(len(test), dtype=int)),
        ("hand-written rules (demo 1)", [predict_with_rules(patient)[0] for _, patient in test.iterrows()]),
        (f"decision tree (depth {tree_model.max_depth})", tree_model.predict(test[features])),
        ("logistic regression", predict_logistic_regression(test)),
    ]


def race_on_many_splits(patients, features, tree_depth):
    """Retrain and re-test every contender on RACE_REPEATS different shuffles; return the averages."""
    all_results = []
    for seed in range(RACE_REPEATS):
        train, _, test = split_patients(patients, seed=seed)
        tree_model = make_tree(tree_depth).fit(train[features], train[LABEL_COLUMN])
        predict_logistic_regression, _, _ = fit_logistic_regression(train, features)
        all_results.append([
            (name, score(predictions, test[LABEL_COLUMN]))
            for name, predictions in contenders(test, features, tree_model, predict_logistic_regression)
        ])
    averages = []
    for contender_index, (name, _) in enumerate(all_results[0]):
        results = [split_results[contender_index][1] for split_results in all_results]
        mean_accuracy = np.mean([result["accuracy"] for result in results])
        mean_deaths_caught = np.mean([result["deaths_caught"] / result["deaths_total"] for result in results])
        averages.append((name, mean_accuracy, mean_deaths_caught))
    return averages


# ------------------------------------------------------------------------ plots

def show_depth_plot(mean_train_accuracy, mean_validation_accuracy, best_depth):
    figure, axes = plt.subplots(figsize=(8, 4.8))
    axes.plot(TREE_DEPTHS, mean_train_accuracy, "o-", color=CHARCOAL, label="train (patients it learned from)")
    axes.plot(TREE_DEPTHS, mean_validation_accuracy, "o-", color=CAIS_RED, linewidth=2.5,
              label="validation (patients it has never seen)")
    best_index = TREE_DEPTHS.index(best_depth)
    axes.scatter([best_depth], [mean_validation_accuracy[best_index]], s=220, facecolors="none",
                 edgecolors=CAIS_RED, linewidths=2, zorder=5)
    axes.annotate(f"best: depth {best_depth}", (best_depth, mean_validation_accuracy[best_index]),
                  textcoords="offset points", xytext=(0, -26), ha="center", color=CAIS_RED)
    axes.text(TREE_DEPTHS[0], 0.97, "underfitting\n(too simple)", va="top", color="grey")
    axes.text(TREE_DEPTHS[-1], 0.6, "overfitting\n(memorizing)", va="bottom", ha="right", color="grey")
    axes.set_xticks(TREE_DEPTHS)
    axes.set_ylim(0.55, 1.0)
    axes.set_xlabel("max depth of the tree  (hyperparameter)")
    axes.set_ylabel("accuracy")
    axes.set_title("Decision tree: deeper isn't always better")
    axes.legend(loc="center right")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    print("\n  (close the plot window to continue)")
    plt.show()


def show_loss_plot(loss_history):
    figure, axes = plt.subplots(figsize=(8, 4.8))
    axes.plot(loss_history, color=CAIS_RED, linewidth=2.5)
    axes.set_xlabel("training step")
    axes.set_ylabel("loss  (how wrong the predictions are)")
    axes.set_title("Logistic regression: gradient descent pushes the loss down")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    print("\n  (close the plot window to continue)")
    plt.show()


# ------------------------------------------------------------------------- demo

def main():
    patients = load_patients()
    features = feature_columns(patients)
    train, validation, test = split_patients(patients)

    print_title("DEMO 2 - Machine learning: let the computer learn the rules")

    # ----------------------------------------------------------------------
    print_step("Step 1 - Split the patients (shuffled, same death rate in each split)")
    for name, split, role in [
        ("train", train, "the model learns from these"),
        ("validation", validation, "we tune settings (hyperparameters) on these"),
        ("test", test, "the final exam: locked away until the very end"),
    ]:
        print(f"    {name:<11} {len(split):>3} patients  ({split[LABEL_COLUMN].mean():.0%} died)   <- {role}")
    print(f"\n  {len(features)} features per patient. We drop `{LEAKY_COLUMN}` (days of follow-up): patients who")
    print("  died have short follow-ups BECAUSE they died, so it leaks the answer.")
    pause()

    # ----------------------------------------------------------------------
    print_step("Step 2 - A decision tree writes its own IF-ELSE rules")
    small_tree = make_tree(depth=2).fit(train[features], train[LABEL_COLUMN])
    print(f"  A depth-2 tree trained on the {len(train)} training patients, printed as Python:\n")
    for line in tree_as_python(small_tree, features).splitlines():
        print("    " + line)
    print("\n  Demo 1's doctor wrote:  ejection_fraction < 30  or  serum_creatinine > 1.5")
    print("  The tree was never told any medicine. Compare the features and thresholds it picked.")
    pause()

    # ----------------------------------------------------------------------
    print_step("Step 3 - Tuning: how deep should the tree be?")
    print(f"  One {len(validation)}-patient validation set is noisy (one patient = {1 / len(validation):.1%}),")
    print(f"  so we reshuffle train/validation {TUNING_REPEATS} times and average. The test set stays locked.\n")
    development = pd.concat([train, validation])
    train_accuracy = np.zeros((TUNING_REPEATS, len(TREE_DEPTHS)))
    validation_accuracy = np.zeros((TUNING_REPEATS, len(TREE_DEPTHS)))
    for repeat in range(TUNING_REPEATS):
        repeat_train, repeat_validation = split_train_validation(development, seed=repeat)
        for depth_index, depth in enumerate(TREE_DEPTHS):
            tree_model = make_tree(depth).fit(repeat_train[features], repeat_train[LABEL_COLUMN])
            train_accuracy[repeat, depth_index] = score(
                tree_model.predict(repeat_train[features]), repeat_train[LABEL_COLUMN])["accuracy"]
            validation_accuracy[repeat, depth_index] = score(
                tree_model.predict(repeat_validation[features]), repeat_validation[LABEL_COLUMN])["accuracy"]
    mean_train_accuracy = train_accuracy.mean(axis=0)
    mean_validation_accuracy = validation_accuracy.mean(axis=0)
    best_depth = TREE_DEPTHS[int(np.argmax(mean_validation_accuracy))]

    print("    depth    train acc    validation acc")
    for depth, train_acc, validation_acc in zip(TREE_DEPTHS, mean_train_accuracy, mean_validation_accuracy):
        note = ""
        if depth == best_depth:
            note = "<- best on validation"
        elif depth == TREE_DEPTHS[0]:
            note = "<- underfitting: too simple"
        elif depth == TREE_DEPTHS[-1]:
            note = "<- overfitting: memorizes training patients"
        print(f"    {depth:>5}      {train_acc:6.1%}        {validation_acc:6.1%}     {note}")
    print("\n  Train accuracy keeps climbing, validation doesn't. Past the best depth, the tree is")
    print("  learning quirks of specific patients, not patterns that carry over to new ones.")
    show_depth_plot(mean_train_accuracy, mean_validation_accuracy, best_depth)
    final_tree = make_tree(best_depth).fit(train[features], train[LABEL_COLUMN])
    pause()

    # ----------------------------------------------------------------------
    print_step("Step 4 - Logistic regression: predict -> loss -> minimize -> repeat")
    print("  Features scaled using training stats only. Weights start at zero.")
    print(f"  Learning rate {LEARNING_RATE}, {TRAINING_STEPS} steps:\n")
    predict_logistic_regression, weights, loss_history = fit_logistic_regression(train, features, verbose=True)

    print("\n  What it learned (features are scaled, so bigger |weight| = more influence):\n")
    for feature_index in np.argsort(-np.abs(weights)):
        weight = weights[feature_index]
        direction = "higher -> more risk" if weight > 0 else "higher -> less risk"
        bar = "#" * int(round(abs(weight) * 12))
        print(f"    {features[feature_index]:<26} {weight:+.2f}  {bar:<14} {direction}")
    print("\n  weighted sum of inputs -> sigmoid activation -> probability of death")
    print("  That's exactly ONE artificial neuron. Stack lots of them in layers -> a neural network.")
    show_loss_plot(loss_history)
    pause()

    # ----------------------------------------------------------------------
    print_step(f"Step 5 - Final exam: the {len(test)} test patients, used exactly once")
    for name, predictions in contenders(test, features, final_tree, predict_logistic_regression):
        print(f"    {name:<32} {describe(score(predictions, test[LABEL_COLUMN]))}")
    print(f"\n  Each test patient is worth {1 / len(test):.1%}, so one exam is partly luck.")
    pause()

    print(f"\n  Is it a fluke? Same race, retrained on {RACE_REPEATS} different shuffles of the data (averages):\n")
    averages = race_on_many_splits(patients, features, best_depth)
    for name, mean_accuracy, mean_deaths_caught in averages:
        print(f"    {name:<32} accuracy {mean_accuracy:6.1%}   deaths caught {mean_deaths_caught:6.1%}")

    rules_accuracy = averages[1][1]
    best_model_accuracy = max(averages[2][1], averages[3][1])
    if rules_accuracy > best_model_accuracy:
        print("\n  Plot twist: the doctor's rules win. With ~180 training patients, decades of medical")
        print("  knowledge beat learning from scratch. ML pulls ahead with lots of data, or when nobody")
        print("  can write the rules at all (images, speech, text).")
    else:
        print("\n  Learning from data beats the hand-written rules, even on this small dataset.")

    top_tree_features = [features[index] for index in np.argsort(-final_tree.feature_importances_)[:2]]
    top_weight_features = [features[index] for index in np.argsort(-np.abs(weights))[:2]]
    print("\n  What ML did do, with zero medical knowledge:")
    print(f"    tree's most important features:          {', '.join(top_tree_features)}")
    print(f"    logistic regression's biggest weights:   {', '.join(top_weight_features)}")
    print("    doctor's rules (demo 1) led with:        ejection_fraction, serum_creatinine")
    print("  And unlike the rules, the models can keep improving as more patients come in.")
    print("\n  -> Next: neural networks. Many neurons, many layers, same predict/loss/minimize loop.\n")


if __name__ == "__main__":
    main()
