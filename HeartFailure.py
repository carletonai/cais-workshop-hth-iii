import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

# ============================================================
# Dataset
# ============================================================

class HeartFailureDataset(torch.utils.data.Dataset):

    def __init__(self, filepath, rows, fit_stats=None):
        self.data = numpy.loadtxt(filepath, skiprows=1, delimiter=',')
        self.data = self.data[rows, :]
        self.x_data = self.data[:, :-1]

        # Use provided stats (from training set) or compute them
        if fit_stats is None:
            self.x_min = numpy.min(self.x_data, axis=0)
            self.x_max = numpy.max(self.x_data, axis=0)
        else:
            self.x_min, self.x_max = fit_stats

        # feature normalization using consistent stats
        denom = self.x_max - self.x_min
        denom[denom == 0] = 1
        self.x_data = (self.x_data - self.x_min) / denom
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)

        self.y_data = torch.tensor(self.data[:, -1], dtype=torch.long)

    def get_fit_stats(self):
        """Return (x_min, x_max) so val/test can use the same normalization."""
        return (self.x_min, self.x_max)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index, :]
        y = self.y_data[index]
        return x, y

# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):
    """A residual block with skip connection for tabular data."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(x + self.block(x)))

# ============================================================
# Network — wider + residual connections + GELU
# ============================================================

class HeartFailureNetwork(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_classes=2, num_res_blocks=3, dropout=0.2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Output head with bottleneck
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.head(x)
        return x

# ============================================================
# Mixup augmentation (critical for small datasets)
# ============================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup: creates virtual training examples by interpolating pairs."""
    if alpha > 0:
        lam = numpy.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup-augmented data."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# Setup
# ============================================================

dataset_filepath = r"/home/jeremy/cais_workshop/heart_failure_clinical_records_dataset.csv"

# Shuffle data with a fixed seed for reproducibility, then split
# This ensures balanced class distribution across splits
numpy.random.seed(42)
all_data = numpy.loadtxt(dataset_filepath, skiprows=1, delimiter=',')
n_samples = len(all_data)
shuffled_indices = numpy.random.permutation(n_samples)

# Use shuffled indices for the splits
train_rows = shuffled_indices[:200]
val_rows = shuffled_indices[200:250]
test_rows = shuffled_indices[250:]

# create datasets — fit normalization on train, reuse for val/test
train_dataset = HeartFailureDataset(dataset_filepath, train_rows)
fit_stats = train_dataset.get_fit_stats()
val_dataset = HeartFailureDataset(dataset_filepath, val_rows, fit_stats=fit_stats)
test_dataset = HeartFailureDataset(dataset_filepath, test_rows, fit_stats=fit_stats)

# Print class distributions
print(f"Train classes: {numpy.bincount(train_dataset.y_data.numpy())}")
print(f"Val classes:   {numpy.bincount(val_dataset.y_data.numpy())}")
print(f"Test classes:  {numpy.bincount(test_dataset.y_data.numpy())}")

batch_size = 16  # smaller batch = more updates per epoch on small data

# load models w data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_rows), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_rows), shuffle=False)

heart_failure_network = HeartFailureNetwork(
    input_dim=12,
    hidden_dim=128,
    num_classes=2,
    num_res_blocks=3,
    dropout=0.2,
)
print(heart_failure_network)
total_params = sum(p.numel() for p in heart_failure_network.parameters())
print(f"Total parameters: {total_params:,}")

# Compute class weights to handle imbalanced classes
train_labels = train_dataset.y_data.numpy()
class_counts = numpy.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights}")

# Label smoothing helps prevent overconfidence
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# AdamW optimizer (decoupled weight decay, better than Adam for regularization)
optimizer = torch.optim.AdamW(heart_failure_network.parameters(), lr=0.003, weight_decay=1e-3)

# Cosine annealing with warm restarts — cyclically reduces and resets LR
num_epochs = 300
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

# ============================================================
# Training loop with mixup, early stopping, and model saving
# ============================================================

best_val_loss = float('inf')
best_val_acc = 0.0
best_model_state = None
patience = 80  # early stopping patience (generous for cosine restarts)
patience_counter = 0
use_mixup = True

for epoch in range(num_epochs):

    # ---- TRAINING ----
    heart_failure_network.train()
    running_loss, running_correct, running_total = 0.0, 0, 0

    for batch_x, batch_y in train_loader:

        if use_mixup:
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4)
            batch_y_pred = heart_failure_network(mixed_x)
            loss = mixup_criterion(criterion, batch_y_pred, y_a, y_b, lam)

            # For accuracy tracking, use the dominant label
            running_correct += (lam * (batch_y_pred.argmax(dim=1) == y_a).sum().item() +
                                (1 - lam) * (batch_y_pred.argmax(dim=1) == y_b).sum().item())
        else:
            batch_y_pred = heart_failure_network(batch_x)
            loss = criterion(batch_y_pred, batch_y)
            running_correct += (batch_y_pred.argmax(dim=1) == batch_y).sum().item()

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(heart_failure_network.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(epoch + running_total / len(train_dataset))

        running_loss += loss.item() * batch_x.size(0)
        running_total += batch_x.size(0)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total

    # ---- VALIDATION ----
    heart_failure_network.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_y_pred = heart_failure_network(batch_x)
            loss = criterion(batch_y_pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            val_correct += (batch_y_pred.argmax(dim=1) == batch_y).sum().item()
            val_total += batch_x.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    # Track best model (by val accuracy primarily, then loss as tiebreaker)
    if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in heart_failure_network.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    # Print every 20 epochs
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

# ============================================================
# Restore best model and evaluate on test set
# ============================================================

print(f"\n{'='*60}")
print(f"Best Val Loss: {best_val_loss:.4f}  Best Val Acc: {best_val_acc:.4f}")

if best_model_state is not None:
    heart_failure_network.load_state_dict(best_model_state)
    print("Restored best model weights.")

heart_failure_network.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_y_pred = heart_failure_network(batch_x)
        loss = criterion(batch_y_pred, batch_y)

        preds = batch_y_pred.argmax(dim=1)
        test_acc = (preds == batch_y).sum().item() / len(batch_y)
        print(f"Test Loss: {loss.item():.4f}  Test Acc: {test_acc:.4f}")

        # Per-class accuracy breakdown
        for c in range(2):
            mask = (batch_y == c)
            if mask.sum() > 0:
                class_acc = (preds[mask] == batch_y[mask]).sum().item() / mask.sum().item()
                print(f"  Class {c} accuracy: {class_acc:.4f} ({mask.sum().item()} samples)")
