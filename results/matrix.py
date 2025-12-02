import os
import pandas as pd
import matplotlib.pyplot as plt

target_species = [
        "Parus major",
        "Turdus merula",
        "Erithacus rubecula",
        "Cyanistes caeruleus",
        "Passer domesticus",
        "Columba palumbus",
        "Sturnus vulgaris",
        "Fringilla coelebs",
        "Streptopelia decaocto",
        "Garrulus glandarius"
    ]

csv_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.csv")
cm = pd.read_csv(csv_path, header=None).values

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap="viridis")
plt.colorbar()

plt.xticks(ticks=range(len(target_species)), labels=target_species, rotation=45)
plt.yticks(ticks=range(len(target_species)), labels=target_species)

plt.xlabel("Species predicted by the model")
plt.ylabel("Actual species")
plt.title("Confusion Matrix")

# Annotate the boxes
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center", color="black" if cm[i, j] > cm.max()/2 else "white")

save_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print("Image enregistrée :", save_path)
