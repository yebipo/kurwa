import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import gc

# ==========================================
# 1. УЛЬТРА-ЛЕГКИЙ КОНФИГ
# ==========================================
SEQ_LEN = 16  # Длина памяти
BATCH_SIZE = 64  # Резко снизили батч (был 512)
EPOCHS = 3  # Меньше эпох для теста
DEVICE = torch.device("cpu")  # Принудительно CPU, раз CUDA нет


class EntropyDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len]


def load_data_ultra_lean(filepath, is_test=False, scaler=None):
    print(f"-> Загрузка {filepath}...")
    # Читаем только колонки f1-f4 и таргет (индексы 1-5)
    df = pd.read_csv(filepath, sep=';', decimal=',', usecols=[1, 2, 3, 4, 5],
                     header=None if is_test else 'infer', engine='c')

    # Быстрая очистка
    data = df.apply(
        lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce')).dropna().values.astype(
        np.float32)
    del df
    gc.collect()

    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)

    return data[:, :-1], data[:, -1], scaler


# ==========================================
# 2. КОМПАКТНАЯ МОДЕЛЬ
# ==========================================
class SmallLSTM(nn.Module):
    def __init__(self):
        super(SmallLSTM, self).__init__()
        # 1 слой и 24 скрытых юнита - этого достаточно для начала
        self.lstm = nn.LSTM(4, 24, num_layers=1, batch_first=True)
        self.fc = nn.Linear(24, 3)  # Квантили 0.05, 0.5, 0.95

    def forward(self, x):
        # x: (batch, seq, feat)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Только последний шаг


def quantile_loss(preds, target):
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    return torch.mean(torch.cat(losses, dim=1))


# ==========================================
# 3. ЦИКЛ ОБУЧЕНИЯ С ОЧИСТКОЙ ПАМЯТИ
# ==========================================
def run():
    X_tr, y_tr, scaler = load_data_ultra_lean('million.csv')
    train_loader = DataLoader(EntropyDataset(X_tr, y_tr, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)

    model = SmallLSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print(f"-> Обучение (батч {BATCH_SIZE})...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (b_X, b_y) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(b_X)
            loss = quantile_loss(outputs, b_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Раз в 500 батчей чистим мусор вручную
            if i % 500 == 0:
                gc.collect()

        print(f"Эпоха {epoch + 1} | Loss: {running_loss / len(train_loader):.6f}")

    # Сохранение
    torch.save(model.state_dict(), 'best_lstm.pth')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("-> Модель и скалер сохранены. Готово!")

    # Тест на уникальных строках (упрощенно)
    X_te, y_te, _ = load_data_ultra_lean('unique_rows.csv', is_test=True, scaler=scaler)
    test_loader = DataLoader(EntropyDataset(X_te, y_te, SEQ_LEN), batch_size=BATCH_SIZE)

    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for b_X, b_y in test_loader:
            p = model(b_X).numpy()
            t = b_y.numpy()
            hits += np.sum((t >= p[:, 0]) & (t <= p[:, 2]))
            total += len(t)

    print(f"Финальное покрытие на тесте: {(hits / total) * 100:.2f}%")


if __name__ == "__main__":
    run()