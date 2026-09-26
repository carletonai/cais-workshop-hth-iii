def predict_fruit(weight):
    if weight < 150:
        return "apple"
    return "orange"


def main():
    print("Demo 1: Rule-based AI")
    print("Toy example: sorting fruit by weight.")
    print("A human wrote the rule: below 150g = apple, otherwise = orange.\n")

    for weight in [135, 155, 185]:
        print(f"{weight}g -> {predict_fruit(weight)}")

    print("\nThere is no training. Changing the rule means editing the code.")


if __name__ == "__main__":
    main()
