from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import Tensor
from time import sleep

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torch.utils.data import random_split
import torch.nn as nn
from timeit import default_timer as timer


class FlowerClassificationModel(nn.Module):
    def __init__(self, input_shape: int = 3, hidden_units: int = 32, output_shape: int = 5, dropout_prob: float = 0.3):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),  # Add dropout after ReLU
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),  # Add dropout after ReLU
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),  # Add dropout after ReLU
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 2, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),  # Add dropout after ReLU
            nn.MaxPool2d(2)
        )
        # Adjust the input size to Linear layer after dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 2 * 61 * 61, out_features=hidden_units * 2),
            # Hidden layer before final classification
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # Add dropout before final classification layer
            nn.Linear(in_features=hidden_units * 2, out_features=output_shape)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)


def train_model(model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                dataloader: torch.utils.data.DataLoader,
                device: torch.device | str):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for (img, label) in tqdm(dataloader, ncols=80, desc='Training', mininterval=1):
        img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        prediction = logits.argmax(axis=1)
        train_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=5).item()

    return train_loss / len(dataloader), train_acc / len(dataloader)


def validate_model(model: torch.nn.Module,
                   loss_fn: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device | str):
    model.eval()
    valid_loss, val_acc = 0.0, 0.0

    with torch.no_grad():
        for (img, label) in tqdm(dataloader, ncols=80, desc='Valid', mininterval=1):
            img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
            logits = model(img)
            loss = loss_fn(logits, label)
            valid_loss += loss.item()
            prediction = logits.argmax(axis=1)
            val_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=5).item()

    return valid_loss / len(dataloader), val_acc / len(dataloader)


def test_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device | str) -> Tuple[Dict[str, float], List[Tensor], List[Tensor]]:
    model.eval()
    results = {'f1': 0, 'acc': 0, 'recall': 0, 'precision': 0}
    all_labels, all_preds = [], []

    with torch.no_grad():
        for (img, label) in tqdm(dataloader, ncols=80, desc='Test', mininterval=1):
            img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
            test_pred_logits = model(img)
            all_labels.append(label.cpu())
            all_preds.append(test_pred_logits.argmax(axis=1).cpu())
            test_pred_labels = test_pred_logits.argmax(dim=1)
            results['acc'] += accuracy(test_pred_labels, label, task='multiclass', num_classes=5).item()
            results['f1'] += f1_score(test_pred_labels, label, average='macro', task="multiclass", num_classes=5).item()
            results['recall'] += recall(test_pred_labels, label, average='macro', task='multiclass',
                                        num_classes=5).item()
            results['precision'] += precision(test_pred_labels, label, task='multiclass', num_classes=5).item()

    num_batches = len(dataloader)
    for key in results:
        results[key] /= num_batches

    return results, all_labels, all_preds


def display_sample_images(dataset, idx_to_label, predictions, correct_indices, incorrect_indices, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    for i in range(num_samples):
        if i >= len(correct_indices) or i >= len(incorrect_indices):
            break

        correct_idx = correct_indices[i]
        incorrect_idx = incorrect_indices[i]

        # Ensure indices are within dataset bounds
        if correct_idx >= len(dataset) or incorrect_idx >= len(dataset):
            continue

        correct_img, _ = dataset.dataset[dataset.indices[correct_idx]]  # Access original dataset
        incorrect_img, _ = dataset.dataset[dataset.indices[incorrect_idx]]  # Access original dataset

        correct_img = correct_img.permute(1, 2, 0).numpy()
        incorrect_img = incorrect_img.permute(1, 2, 0).numpy()

        correct_label = idx_to_label[dataset.dataset.targets[dataset.indices[correct_idx]]]  # Access original dataset
        incorrect_label = idx_to_label[
            dataset.dataset.targets[dataset.indices[incorrect_idx]]]  # Access original dataset

        # Use predictions to get predicted labels
        correct_pred = idx_to_label[predictions[correct_idx]]
        incorrect_pred = idx_to_label[predictions[incorrect_idx]]

        axes[i, 0].imshow(correct_img)
        axes[i, 0].set_title(f'Correct: {correct_label} (Predicted: {correct_pred})')
        axes[i, 1].imshow(incorrect_img)
        axes[i, 1].set_title(f'Incorrect: {incorrect_label} (Predicted: {incorrect_pred})')
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def display_losses(train_losses, val_losses, train_accs, val_accs, epochs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accs)), train_accs, label='Training Accuracy')
    plt.plot(range(len(val_accs)), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),  # Add random horizontal flips
        transforms.RandomRotation(20),  # Add random rotations
        transforms.ToTensor()
    ])

    flowers_path = (R"C:\Users\Michael\OneDrive - University of Wollongong\UOW\Bachelor of Computer "
                    R"Science\Year2\Spring\CSCI218 - Foundations of Artificial "
                    R"Intelligence\Assessments\Assignment\flowers")
    data = ImageFolder(root='flowers', transform=transform)
    idx_to_label = {v: k for k, v in data.class_to_idx.items()}  # Create idx_to_label mapping
    train_set, val_set, test_set = random_split(data, [0.6, 0.2, 0.2])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)  # Set shuffle=False here

    model = FlowerClassificationModel(input_shape=3, hidden_units=30, output_shape=5).to(device)

    print(model)
    sleep(0.5)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=0.001)

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    best_valid_acc = 0
    train_time = 0
    start_time = timer()

    # In your training loop
    for epoch in range(100):
        train_loss, train_acc = train_model(model, loss_fn, optimizer, train_loader, device)
        valid_loss, valid_acc = validate_model(model, loss_fn, val_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        if epoch == 0:
            train_time = timer() - start_time

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            # Ensure the 'models' directory exists
            torch.save(model.state_dict(), './models/model.pt')

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_acc: {valid_acc:.4f}"
        )
        sleep(0.1)

    inference_start_time = timer()
    results, y_test, y_pred = test_model(model, test_loader, device)
    inference_timer = timer() - inference_start_time
    cm = confusion_matrix(torch.cat(y_test), torch.cat(y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Concatenate all labels and predictions
    all_y_test = torch.cat(y_test)
    all_y_pred = torch.cat(y_pred)

    # Find correct and incorrect indices
    correct_indices = [index for index in range(len(all_y_test))
                       if all_y_pred[index] == all_y_test[index]]
    incorrect_indices = [index for index in range(len(all_y_test))
                         if all_y_pred[index] != all_y_test[index]]

    if len(correct_indices) > 0 and len(incorrect_indices) > 0:
        # Randomly select samples
        correct_sample_indices = np.random.choice(correct_indices, min(5, len(correct_indices)), replace=False)
        incorrect_sample_indices = np.random.choice(incorrect_indices, min(5, len(incorrect_indices)), replace=False)

        display_sample_images(test_set, idx_to_label, all_y_pred.numpy(), correct_sample_indices,
                              incorrect_sample_indices, num_samples=5)

    end_time = timer()
    print(f'Time taken: {end_time - start_time:.4f} seconds.')
    print(f"Training time for a single run: {train_time:.4f} seconds")
    print(f"Inference time for a single run: {inference_timer:.4f}\n")
    print(
        f"Test accuracy: {results['acc']:.4f} | "
        f"Test f1: {results['f1']:.4f} | "
        f"Test recall: {results['recall']:.4f} | "
        f"Test precision: {results['precision']:.4f}"
    )

    display_losses(train_losses, valid_losses, train_accs, valid_accs, len(train_losses))
