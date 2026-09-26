import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class HeartFailureNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.layers(x)


def load_data(seed):
    path = Path(__file__).with_name("heart_failure_clinical_records_dataset.csv")
    data = np.loadtxt(path, skiprows=1, delimiter=",")
    x = data[:, :11].copy()
    y = data[:, -1].astype(np.int64)
    x[:, [2, 6, 7]] = np.log1p(x[:, [2, 6, 7]])

    train_rows, test_rows = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
    )
    train_rows, val_rows = train_test_split(
        train_rows, test_size=0.25, stratify=y[train_rows], random_state=seed
    )
    scaler = StandardScaler().fit(x[train_rows])
    datasets = []
    for rows in (train_rows, val_rows, test_rows):
        features = torch.tensor(scaler.transform(x[rows]), dtype=torch.float32)
        labels = torch.tensor(y[rows], dtype=torch.long)
        datasets.append(TensorDataset(features, labels))
    return datasets


def train_model(train, validation, seed, baseline=False):
    torch.manual_seed(seed)
    input_size = train.tensors[0].shape[1]
    if baseline:
        model = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    else:
        model = HeartFailureNetwork(input_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        counts = torch.bincount(train.tensors[1], minlength=2)
        weights = len(train) / (2 * counts.float())
        criterion = nn.CrossEntropyLoss(weight=weights)

    loader = DataLoader(
        train, batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, (50 if baseline else 400) + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(validation.tensors[0]), validation.tensors[1]).item()
        if val_loss < best_loss - 0.0001:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if not baseline and stale_epochs >= 40:
            break

    if not baseline:
        model.load_state_dict(best_state)
    return model, 50 if baseline else best_epoch


def evaluate(name, labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    balanced = balanced_accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, zero_division=0)
    print(f"{name:<22} {accuracy:>9.1%} {balanced:>12.1%} {recall:>12.1%}")
    return confusion_matrix(labels, predictions, labels=[0, 1])


def predict(model, dataset):
    model.eval()
    with torch.no_grad():
        return model(dataset.tensors[0]).argmax(dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train, validation, test = load_data(args.seed)

    print("Predicting DEATH_EVENT from 11 features; follow-up duration excluded.")
    for name, dataset in (("Train", train), ("Validation", validation), ("Test", test)):
        counts = torch.bincount(dataset.tensors[1], minlength=2).tolist()
        print(f"{name}: {len(dataset)} patients, {counts[0]} survivors, {counts[1]} deaths")

    model, epoch = train_model(train, validation, args.seed)
    print(f"\nMLP: 11 -> 32 -> 16 -> 2; best validation loss at epoch {epoch}")
    models = []
    if args.compare:
        baseline, _ = train_model(train, validation, args.seed, baseline=True)
        models.append(("Previous MLP + SGD", baseline))
    models.append(("Updated MLP + AdamW", model))
    majority = int(torch.bincount(train.tensors[1]).argmax())

    for name, dataset in (("Validation", validation), ("Test", test)):
        labels = dataset.tensors[1].numpy()
        print(f"\n{name} results")
        print(f"{'Model':<22} {'Accuracy':>9} {'Balanced acc':>12} {'Death recall':>12}")
        evaluate("Majority baseline", labels, np.full(len(labels), majority))
        for model_name, candidate in models:
            matrix = evaluate(model_name, labels, predict(candidate, dataset))
        print("Updated MLP confusion matrix (rows=actual, columns=predicted; survivor, death):")
        print(matrix)
    print("\nComparison uses the same split and preprocessing for both MLPs.")
    print("The test set is used only for reporting, not training or early stopping.")


if __name__ == "__main__":
    main()
