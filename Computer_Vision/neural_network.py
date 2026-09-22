import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Choose device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Select dataset: 'cifar10' or 'mnist'
dataset_name = 'cifar10'

if dataset_name.lower() == 'cifar10':
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    num_classes = 10
    input_dim = 32*32*3
elif dataset_name.lower() == 'mnist':
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    num_classes = 10
    input_dim = 28*28
else:
    raise ValueError('Unknown dataset')

# Split train into train/val
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset,
    [train_size, val_size],
    generator=generator
)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyper‑parameters
hidden_dim = 100  # you can tune this
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 10

# Initialize model, loss function and optimizer
model = TwoLayerNet(input_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Initialize history lists before the training loop
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# Training loop skeleton
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / total
    train_acc = 100. * correct / total

    # Evaluate on validation set
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += (loss.item() * images.size(0))
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted.eq(labels).sum().item())

    # Calculate final validation metrics for the epoch
    val_loss = val_running_loss / val_total
    val_acc = 100. * val_correct / val_total

    # Append metrics to history lists once per epoch
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    # Print epoch summary once per epoch
    print(
      f"Epoch {epoch+1}/{num_epochs}, "
      f"Train loss: {train_loss:.4f}, "
      f"Val loss: {val_loss:.4f}, "
      f"Train acc: {train_acc:.2f}%, "
       f"Val acc: {val_acc:.2f}%"
   )

   import matplotlib.pyplot as plt

best_val_acc = max(
    val_acc_history
)

best_epoch = (
    np.argmax(val_acc_history) + 1
)

print()
print(
    f"Best validation accuracy: "
    f"{best_val_acc:.2f}%"
)

print(
    f"Best epoch: {best_epoch}"
)

epochs = range(
    1,
    num_epochs + 1
)

plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    train_acc_history,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_acc_history,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Two-Layer Neural Network Accuracy"
)

plt.legend()
plt.grid(True)

plt.show()

plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    train_loss_history,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss_history,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Two-Layer Neural Network Loss"
)

plt.legend()
plt.grid(True)

plt.show()

def train_two_layer_model(
    hidden_dim,
    learning_rate,
    weight_decay,
    batch_size,
    num_epochs
):

    # Create loaders for this batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Create a fresh model for every experiment
    model = TwoLayerNet(
        input_dim,
        hidden_dim,
        num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # TRAINING
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted.eq(labels)
                .sum()
                .item()
            )

        train_loss = (
            running_loss / total
        )

        train_acc = (
            100. * correct / total
        )

        # VALIDATION
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = outputs.max(1)

                val_total += labels.size(0)

                val_correct += (
                    predicted.eq(labels)
                    .sum()
                    .item()
                )

        val_acc = (
            100. * val_correct / val_total
        )


        # Save history
        train_loss_history.append(
            train_loss
        )

        train_acc_history.append(
            train_acc
        )

        val_acc_history.append(
            val_acc
        )


        # Track best validation result
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch + 1


        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train loss: {train_loss:.4f}, "
            f"Train acc: {train_acc:.2f}%, "
            f"Val acc: {val_acc:.2f}%"
        )


    return {
        "model": model,
        "train_loss": train_loss_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch
}

#Experiments:

# Hyperparameter experiments

experiments = [

    # Original configuration
    {
        "hidden_dim": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "num_epochs": 10
    },

    # Higher learning rate
    {
        "hidden_dim": 100,
        "learning_rate": 1e-2,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "num_epochs": 10
    },

    # Larger hidden layer
    {
        "hidden_dim": 200,
        "learning_rate": 1e-2,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "num_epochs": 20
    },

    # Even larger hidden layer
    {
        "hidden_dim": 500,
        "learning_rate": 1e-2,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "num_epochs": 20
    },

    # Different regularization
    {
        "hidden_dim": 200,
        "learning_rate": 1e-2,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "num_epochs": 20
    }
]

experiment_results = []


for i, config in enumerate(experiments):

    print()
    print("=" * 80)
    print(f"EXPERIMENT {i + 1}")
    print("=" * 80)

    print(
        f"Hidden dimension: "
        f"{config['hidden_dim']}"
    )

    print(
        f"Learning rate: "
        f"{config['learning_rate']}"
    )

    print(
        f"Weight decay: "
        f"{config['weight_decay']}"
    )

    print(
        f"Batch size: "
        f"{config['batch_size']}"
    )

    print(
        f"Epochs: "
        f"{config['num_epochs']}"
    )

    print()


    result = train_two_layer_model(

        hidden_dim=
            config["hidden_dim"],

        learning_rate=
            config["learning_rate"],

        weight_decay=
            config["weight_decay"],

        batch_size=
            config["batch_size"],

        num_epochs=
            config["num_epochs"]
    )


    experiment_results.append({

        **config,

        "best_val_acc":
            result["best_val_acc"],

        "best_epoch":
            result["best_epoch"],

        "result":
            result
    })



