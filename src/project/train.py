import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import time
from pathlib import Path
import multiprocessing

from project.model import CNN  # относительный импорт работает, если src помечен как Sources Root

# Настройки
class Config:
    batch_size = 64
    epochs = 15
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")


# вычисляем корень репозитория (две папки вверх от этого файла: src/project -> корень)
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
GRAPHICS_DIR = BASE_DIR / "graphics"

# Создаём папки, если их нет
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

# безопасный выбор num_workers
_cpu_count = multiprocessing.cpu_count() if hasattr(multiprocessing, "cpu_count") else 1
NUM_WORKERS = min(2, max(0, _cpu_count - 1))  # 0..2

# Функции даталоадеров
def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader

# Тренировочная и тестовая функции
def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            processed = batch_idx * len(data)
            percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{processed}/{len(train_loader.dataset)} '
                  f'({percent:.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = 100. * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # суммируем loss для корректного среднего
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    n_samples = len(test_loader.dataset)
    test_loss_avg = test_loss / n_samples if n_samples > 0 else 0.0
    accuracy = 100. * correct / n_samples if n_samples > 0 else 0.0

    print(f'\nTest set: Average loss: {test_loss_avg:.4f}, '
          f'Accuracy: {correct}/{n_samples} ({accuracy:.2f}%)\n')

    return test_loss_avg, accuracy

# Визуализация примеров (сохранение в graphics/)
def visualize_samples(test_loader, model, device, num_samples=10):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images[:num_samples])
        preds = outputs.argmax(dim=1)

    images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images_np[i][0], cmap='gray')
        ax.set_title(f'True: {labels_np[i]}, Pred: {preds_np[i]}')
        ax.axis('off')

    plt.tight_layout()
    out_path = GRAPHICS_DIR / "mnist_predictions.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"Сохранены предсказания: {out_path}")
    plt.close()

# Основная логика
def main():
    cfg = Config()

    print("Загрузка данных MNIST...")
    train_loader, test_loader = get_data_loaders()

    print(f"Размер тренировочного датасета: {len(train_loader.dataset)}")
    print(f"Размер тестового датасета: {len(test_loader.dataset)}")
    print(f"Размер батча: {cfg.batch_size}")

    device = cfg.device

    # Создаём модель
    model = CNN().to(device)
    print(f"\nМодель создана: {model.__class__.__name__}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print("\nНачало обучения...")
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        # Тренировка
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)

        # Тестирование
        test_loss, test_acc = test(model, device, test_loader)

        epoch_time = time.time() - epoch_start

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Эпоха {epoch} завершена за {epoch_time:.2f} секунд")
        print(f"Тренировочная точность: {train_acc:.2f}%")
        print(f"Тестовая точность: {test_acc:.2f}%\n")

    total_time = time.time() - start_time
    print(f"Общее время обучения: {total_time:.2f} секунд")

    # визуализация примеров и графики
    print("Визуализация примеров...")
    visualize_samples(test_loader, model, device)

    # Графики обучения
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss during Training')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy during Training')

    plt.tight_layout()
    history_path = GRAPHICS_DIR / "training_history.png"
    plt.savefig(str(history_path), dpi=150, bbox_inches='tight')
    print(f"Сохранён график обучения: {history_path}")
    plt.close()

    # Сохранение модели и метрик в models/
    save_path = MODEL_DIR / "mnist_cnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'config': {
            'batch_size': cfg.batch_size,
            'epochs': cfg.epochs,
            'learning_rate': cfg.learning_rate
        }
    }, str(save_path))

    print(f"Модель сохранена как '{save_path}'")

    # Финальная статистика
    if test_accuracies:
        best_test_acc = max(test_accuracies)
        best_epoch = test_accuracies.index(best_test_acc) + 1
        print(f"\n🏆 Лучшая точность на тесте: {best_test_acc:.2f}% на эпохе {best_epoch}")

# Функция для загрузки и тестирования сохраненной модели
def load_and_test_model():
    cfg = Config()

    _, test_loader = get_data_loaders()
    device = cfg.device

    model = CNN().to(device)
    save_path = MODEL_DIR / "mnist_cnn_model.pth"

    if not save_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {save_path}")

    checkpoint = torch.load(str(save_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Модель загружена. Тестирование...")
    test_loss, test_accuracy = test(model, device, test_loader)

    return test_accuracy


if __name__ == '__main__':
    main()

    # Для проверки загрузки после тренировки можно раскомментировать:
    load_and_test_model()
