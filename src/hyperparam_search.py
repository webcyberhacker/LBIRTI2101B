from pathlib import Path
from src.train import train

learning_rates = [0.0001] #[0.00001, 0.0001, 0.001, 0.01] au dela de 0.01 ca diverge, O.0001 c'est carré
batch_sizes = [4, 8, 16, 32] #valeurs typiques à tester 
epoch_values = [80] #valeurs max, au dela de 70 epochs ca overfit souvent

base_dir = Path("models")
test2_dir = base_dir / "test_3" # à modifier en fonction de la ou on veut enregistrer les résultats
test2_dir.mkdir(parents=True, exist_ok=True)

def lr_to_code(lr):
    return f"{int(lr * 10000):05d}"

for lr in learning_rates:
    for bs in batch_sizes:
        for ep in epoch_values:

            name = f"{lr_to_code(lr)}_{bs}_{ep}"
            out_dir = test2_dir / name
            out_dir.mkdir(exist_ok=True)

            print(f"Running lr={lr}, batch={bs}, epochs={ep}")
            print(f"Saving to {out_dir}")

            train(
                data_dir="data/dataset",
                epochs=ep,
                batch_size=bs,
                learning_rate=lr,
                output_dir=str(out_dir),
                device="mps"
            )



            