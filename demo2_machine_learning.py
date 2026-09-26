from sklearn.tree import DecisionTreeClassifier


def main():
    print("Demo 2: Machine learning AI")
    print("Toy example: sorting fruit by weight.")

    weights = [[120], [130], [140], [121], [135], [90]]
    labels = ["apple", "apple", "apple", "orange", "orange", "orange"]

    print("\nTraining examples:")
    for weight, label in zip(weights, labels):
        print(f"{weight[0]}g -> {label}")

    model = DecisionTreeClassifier(max_depth=1, random_state=0)
    model.fit(weights, labels)
    print(f"\nThe model learned a cutoff of {model.tree_.threshold[0]:.0f}g. Guessing new unknown weights: \n")

    new_weights = [[100], [108], [156]]
    predictions = model.predict(new_weights)
    for weight, prediction in zip(new_weights, predictions):
        print(f"{weight[0]}g -> {prediction}")


if __name__ == "__main__":
    main()
