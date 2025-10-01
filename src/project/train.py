# src/project/train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
import matplotlib.pyplot as plt

import time
from pathlib import Path
import multiprocessing
from PIL import Image
from torchvision.transforms import functional as TF

from project.model import CNN  # работает, если src помечен как Sources Root

class Config:
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMNIST_SPLIT = "balanced"
    # Поправлять ориентацию EMNIST (обычно нужно): True/False
    EMNIST_ORIENTATION_FIX = True
    # Небольшая аугментация (RandomAffine) — включи, если хочешь повысить робастность
    USE_AUGMENTATION = False
    # Выбор оптимизатора: "adam" или "sgd"
    OPTIMIZER = "adam"
    # Scheduler (StepLR) параметры
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.1

    print(f"Используется устройство: {device}")
    print(f"EMNIST split: {EMNIST_SPLIT}, orientation_fix={EMNIST_ORIENTATION_FIX}, augmentation={USE_AUGMENTATION}")
    print(f"Оптимизатор: {OPTIMIZER}")

# вычисляем корень репозитория (src/project -> корень)
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

# Вспомогательные функции
def emnist_orientation_fix_pil(img: Image.Image) -> Image.Image:
    """
    Надёжная коррекция ориентации для EMNIST, работающая с PIL Image.
    Обычно rotate(-90) + hflip даёт нормальную ориентацию.
    Если результат некорректен — попробуй поменять rotate угол или убрать hflip.
    """
    # Поворачиваем -90 градусов и зеркалим по горизонтали
    img = TF.rotate(img, -90, expand=True)
    img = TF.hflip(img)
    return img


# DataLoaders (EMNIST)
def get_data_loaders():
    # Собираем трансформации
    transforms_list = []

    # Если включена предварительная корректировка, делаем это с PIL
    if Config.EMNIST_ORIENTATION_FIX:
        transforms_list.append(transforms.Lambda(lambda img: emnist_orientation_fix_pil(img)))

    # Небольшая аугментация (опционально)
    if Config.USE_AUGMENTATION:
        aug = transforms.RandomAffine(degrees=8, translate=(0.06, 0.06))
        transforms_list.append(transforms.RandomApply([aug], p=0.5))

    # Перевод в тензор и нормализация (та же, что у MNIST)
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform = transforms.Compose(transforms_list)

    # Загружаем EMNIST
    train_dataset = EMNIST(
        root=str(DATA_DIR),
        split=Config.EMNIST_SPLIT,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = EMNIST(
        root=str(DATA_DIR),
        split=Config.EMNIST_SPLIT,
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

    # Получаем число классов (безопасно)
    try:
        n_classes = len(train_dataset.classes)
    except Exception:
        split_to_classes = {"byclass": 62, "balanced": 47, "bymerge": 47, "letters": 26, "digits": 10, "mnist": 10}
        n_classes = split_to_classes.get(Config.EMNIST_SPLIT, 47)

    return train_loader, test_loader, n_classes

# Train / Test
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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    n_samples = len(test_loader.dataset)
    test_loss_avg = test_loss / n_samples if n_samples > 0 else 0.0
    accuracy = 100. * correct / n_samples if n_samples > 0 else 0.0

    print(f'\nTest set: Average loss: {test_loss_avg:.4f}, '
          f'Accuracy: {correct}/{n_samples} ({accuracy:.2f}%)\n')
    return test_loss_avg, accuracy

# Визуализация примеров
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
    out_path = GRAPHICS_DIR / "emnist_predictions.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"Сохранены предсказания: {out_path}")
    plt.close()

def main():
    cfg = Config()

    print("Загрузка данных EMNIST...")
    train_loader, test_loader, n_classes = get_data_loaders()

    print(f"Размер тренировочного датасета: {len(train_loader.dataset)}")
    print(f"Размер тестового датасета: {len(test_loader.dataset)}")
    print(f"Размер батча: {cfg.batch_size}")
    print(f"Число классов (из датасета): {n_classes}")

    device = cfg.device

    # Создаём модель с нужным числом выходов
    model = CNN(num_classes=n_classes).to(device)
    print(f"\nМодель создана: {model.__class__.__name__}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Выбираем оптимизатор
    if Config.OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.SCHEDULER_STEP, gamma=Config.SCHEDULER_GAMMA)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_acc = 0.0
    best_checkpoint = MODEL_DIR / "best_emnist_cnn.pth"

    print("\nНачало обучения...")
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        # Тренировка
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)

        # Тестирование
        test_loss, test_acc = test(model, device, test_loader)

        # Scheduler step (после валидации)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Эпоха {epoch} завершена за {epoch_time:.2f} секунд")
        print(f"Тренировочная точность: {train_acc:.2f}%")
        print(f"Тестовая точность: {test_acc:.2f}%\n")

        # Сохраняем лучший чекпойнт
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
                'config': {
                    'batch_size': cfg.batch_size,
                    'epochs': cfg.epochs,
                    'learning_rate': cfg.learning_rate,
                    'emnist_split': cfg.EMNIST_SPLIT
                }
            }, str(best_checkpoint))
            print(f"Новый лучший чекпойнт сохранён: {best_checkpoint} (acc={best_acc:.2f}%)")

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

    # Сохранение финальной модели и метрик в models/
    final_save = MODEL_DIR / "emnist_cnn_model_final.pth"
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
            'learning_rate': cfg.learning_rate,
            'emnist_split': cfg.EMNIST_SPLIT
        }
    }, str(final_save))

    print(f"Финальная модель сохранена как '{final_save}'")

    # Финальная статистика
    if test_accuracies:
        best_test_acc = max(test_accuracies)
        best_epoch = test_accuracies.index(best_test_acc) + 1
        print(f"\n🏆 Лучшая точность на тесте: {best_test_acc:.2f}% на эпохе {best_epoch}")


# -----------------------
# Функция для загрузки и тестирования сохраненной модели
# -----------------------
def load_and_test_model():
    cfg = Config()

    _, test_loader, n_classes = get_data_loaders()
    device = cfg.device

    model = CNN(num_classes=n_classes).to(device)
    save_path = MODEL_DIR / "best_emnist_cnn.pth"

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
