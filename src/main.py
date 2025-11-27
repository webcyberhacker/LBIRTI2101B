from src.xenocanto import XenoCantoAPI, download_recordings
from src.xenocanto.utils import get_bird_name
import time  # Nécessaire pour faire une petite pause entre les oiseaux

def main():
    """Main function to search and download recordings for multiple species."""
    # Initialize API client
    api = XenoCantoAPI()
    
    # Ta liste des 10 espèces cibles
    target_species = [
        "Parus major",          # Mésange charbonnière
        "Turdus merula",        # Merle noir
        "Erithacus rubecula",   # Rouge-gorge familier
        "Cyanistes caeruleus",  # Mésange bleue
        "Passer domesticus",    # Moineau domestique
        "Columba palumbus",     # Pigeon ramier
        "Sturnus vulgaris",     # Étourneau sansonnet
        "Fringilla coelebs",    # Pinson des arbres
        "Streptopelia decaocto",# Tourterelle turque
        "Pica pica"             # Pie bavarde
    ]

    print(f"=== Démarrage du téléchargement pour {len(target_species)} espèces ===")
    
    for species in target_species:
        # On construit la requête dynamiquement pour chaque espèce
        # Je garde 'cnt:belgium' pour rester fidèle à ton script, mais tu peux l'enlever pour avoir le monde entier
        query = f'sp:"{species}" q:A type:song'
        print(f"\n--- Traitement de : {species} ---")
        print(f"Searching: {query}")
        
        # J'ai mis 20 fichiers par espèce (per_page=20) pour que tu aies un peu de matière pour l'IA
        # Tu peux remettre 5 si c'est juste pour tester
        data = api.search(query, per_page=50)
        
        if not data or not data.get('recordings'):
            print(f"No recordings found for {species}.")
            continue
        
        recordings = data.get('recordings', [])
        total = data.get('numRecordings', 0)
        
        print(f"Found {total} recording(s). Downloading {len(recordings)}...")
        
        # Get bird name and download
        if recordings:
            # On récupère le nom formaté du dossier via le premier fichier trouvé
            bird_name = get_bird_name(recordings[0])
            downloaded, failed = download_recordings(recordings, bird_name)
            print(f"Downloaded: {downloaded}, Failed: {failed}")
        
        # Petite pause de 1 seconde pour être poli avec le serveur
        time.sleep(1)

    print("\n=== The download is complete ===")

if __name__ == "__main__":
    main()