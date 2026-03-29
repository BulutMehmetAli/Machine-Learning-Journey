import numpy as np
import matplotlib.pyplot as plt

# ===== VERİ OLUŞTUR =====
np.random.seed(42)
X = 2 * np.random.rand(100, 1)        # 100 nokta, 0-2 arası
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + gürültü

# ===== PARAMETRELER =====
w = 0.0          # başlangıç ağırlığı (rastgele)
b = 0.0          # başlangıç bias'ı
learning_rate = 0.1
n_iterations = 1000
n = len(X)

# ===== LOSS FONKSİYONU (Mean Squared Error) =====
def compute_loss(X, y, w, b):
    predictions = w * X + b
    loss = (1/n) * np.sum((predictions - y) ** 2)
    return loss

# ===== GRADIENT DESCENT DÖNGÜSÜ =====
loss_history = []

for iteration in range(n_iterations):

    # 1. Tahmin yap
    y_pred = w * X + b

    # 2. Hataları hesapla
    error = y_pred - y

    # 3. Gradient'leri hesapla
    # Loss'un w'ya göre türevi
    dw = (2/n) * np.sum(error * X)
    # Loss'un b'ye göre türevi
    db = (2/n) * np.sum(error)

    # 4. Parametreleri güncelle
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 5. Loss'u kaydet
    loss = compute_loss(X, y, w, b)
    loss_history.append(loss)

    if iteration % 100 == 0:
        print(f"İterasyon {iteration}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(f"\nSonuç → w: {w:.4f} (gerçek: 3), b: {b:.4f} (gerçek: 4)")

# ===== GÖRSELLEŞTİR =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Sol: Veri + öğrenilen doğru
ax1.scatter(X, y, alpha=0.5, label="Veri")
ax1.plot(X, w * X + b, color="red", label=f"y = {w:.2f}x + {b:.2f}")
ax1.set_title("Öğrenilen Doğru")
ax1.legend()

# Sağ: Loss eğrisi
ax2.plot(loss_history, color="orange")
ax2.set_title("Loss Azalması")
ax2.set_xlabel("İterasyon")
ax2.set_ylabel("Loss")

plt.tight_layout()
plt.show()