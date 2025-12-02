from src.xenocanto import XenoCantoAPI, download_recordings
from src.xenocanto.utils import get_bird_name
import time

def main():
    api = XenoCantoAPI()

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

    max_per_species = 300
    qualities = ["A", "B", "C", "D", "E"]

    print(f"=== Début du téléchargement pour {len(target_species)} espèces ===")

    for species in target_species:
        print(f"\n--- Espèce : {species} ---")

        all_recordings = []

        # On effectue 2 recherches : qualité A puis qualité B…
        for q in qualities:
            query = f'sp:"{species}" q:{q} type:song'
            print(f"Requête: {query}")

            data = api.search(query, per_page=max_per_species)
            recs = data.get("recordings", [])

            print(f"{len(recs)} sons trouvés en qualité {q}")

            all_recordings.extend(recs)

            # Si on a atteint ou dépassé 200, on arrête
            if len(all_recordings) >= max_per_species:
                break

            time.sleep(0.5)

        if not all_recordings:
            print("Aucun enregistrement trouvé.")
            continue

        # On tronque à 200 si on a dépassé
        selected = all_recordings[:max_per_species]
        print(f"Total retenus : {len(selected)}")

        # Nom du dossier à partir du premier fichier
        bird_name = get_bird_name(selected[0])

        downloaded, failed = download_recordings(selected, bird_name)
        print(f"Téléchargés : {downloaded}, Échecs : {failed}")

        time.sleep(1)

    print("\n=== Téléchargement terminé ===")

if __name__ == "__main__":
    main()
