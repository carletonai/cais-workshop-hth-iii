"""
DEMO 1 - Rule-based AI: a hand-written IF-ELSE "expert system"

Before machine learning, a lot of AI was experts writing rules by hand.
Here we play cardiologist: a few IF-ELSE rules that flag heart-failure
patients at high risk of dying during follow-up. No training, no learning,
just if-statements a human wrote.

Run:   python demo1_rule_based.py              (press Enter to step through)
       python demo1_rule_based.py --no-pause
"""
import numpy as np

from heart_data import (
    LABEL_COLUMN,
    describe,
    load_patients,
    pause,
    print_confusion_matrix,
    print_step,
    print_title,
    score,
)


def predict_with_rules(patient):
    """Return (prediction, reason). prediction: 1 = HIGH RISK of dying, 0 = low risk."""

    # Rule 1: a healthy heart pumps out ~50-70% of its blood with each beat.
    #         Below 30%, the heart muscle is badly weakened.
    if patient["ejection_fraction"] < 30:
        return 1, f"ejection fraction {patient['ejection_fraction']:.0f}% is below 30%"

    # Rule 2: the kidneys clear creatinine out of the blood (normal ~0.6-1.3 mg/dL).
    #         High creatinine means the kidneys are struggling on top of the heart.
    if patient["serum_creatinine"] > 1.5:
        return 1, f"serum creatinine {patient['serum_creatinine']:.1f} mg/dL is above 1.5"

    # Rule 3: rules can combine conditions with AND / OR.
    #         Low blood sodium (normal 135-145 mEq/L) is a warning sign in heart
    #         failure, especially in older patients.
    if patient["age"] >= 75 and patient["serum_sodium"] < 135:
        return 1, f"age {patient['age']:.0f} with low sodium ({patient['serum_sodium']:.0f} mEq/L)"

    return 0, "no rule fired"


def main():
    patients = load_patients()
    labels = patients[LABEL_COLUMN]

    print_title("DEMO 1 - Rule-based AI: IF-ELSE rules written by a human")
    print(f"  {len(patients)} heart-failure patients. {labels.sum()} of them died during follow-up "
          f"({labels.mean():.0%}).")
    print("  Goal: flag the patients at HIGH RISK of dying.\n")
    print("  Our hand-written rules (see predict_with_rules):")
    print("    1. ejection fraction < 30%                  ->  HIGH RISK")
    print("    2. serum creatinine > 1.5 mg/dL             ->  HIGH RISK")
    print("    3. age >= 75 AND serum sodium < 135 mEq/L   ->  HIGH RISK")
    print("    otherwise                                   ->  low risk")
    pause()

    # Run the rules on every patient
    results = [predict_with_rules(patient) for _, patient in patients.iterrows()]
    predictions = np.array([prediction for prediction, _ in results])
    reasons = [reason for _, reason in results]

    print_step("Try the rules on a few patients")
    example_rows = [
        np.flatnonzero((predictions == 1) & (labels == 1))[0],  # a death the rules caught
        np.flatnonzero((predictions == 0) & (labels == 0))[0],  # a survivor the rules cleared
        np.flatnonzero((predictions == 0) & (labels == 1))[0],  # a death the rules missed
    ]
    for row in example_rows:
        patient = patients.iloc[row]
        verdict = "HIGH RISK" if predictions[row] == 1 else "low risk"
        outcome = "died" if labels.iloc[row] == 1 else "survived"
        print(f"\n  Patient #{row}: age {patient['age']:.0f}, ejection fraction {patient['ejection_fraction']:.0f}%, "
              f"creatinine {patient['serum_creatinine']:.1f}, sodium {patient['serum_sodium']:.0f}")
        print(f"    rules say:  {verdict}  ({reasons[row]})")
        print(f"    actually:   {outcome}")
    print("\n  Nice property: a rule-based system can always tell you WHY it decided.")
    pause()

    print_step(f"How good are the rules? (all {len(patients)} patients)")
    print_confusion_matrix(predictions, labels)
    print()
    print(f"    hand-written rules          {describe(score(predictions, labels))}")
    print(f"    'always say survive'        {describe(score(np.zeros(len(labels), dtype=int), labels))}")
    print("\n  Doing NOTHING scores ~68% accuracy, because most patients survive.")
    print("  Accuracy alone can lie: also check how many deaths each approach actually catches.")
    pause()

    print_step("The catch")
    print("  + Simple, fast, no data needed, and it explains itself.")
    print("  - Every threshold (30, 1.5, 75, 135) is a human's guess. Better ones? Trial and error.")
    print("  - It never improves: a thousand new patients later, same rules, same mistakes.")
    print("  - Someone has to know the rules. For a photo of a cat, what would you even write?")
    print("\n  -> Demo 2: let the computer learn the rules from the data instead.\n")


if __name__ == "__main__":
    # This guard lets demo 2 import predict_with_rules without re-running this whole demo.
    main()
