import json
import matplotlib.pyplot as plt

with open('history.json', 'r') as f:
    h = json.load(f)

epochs = range(1, len(h['train_loss']) + 1)

# Wykres Loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, h['train_loss'], label='Train Loss')
plt.plot(epochs, h['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

# Wykres Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, h['train_acc'], label='Train Acc')
plt.plot(epochs, h['val_acc'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.savefig('wykresy_treningu.png')
plt.show()