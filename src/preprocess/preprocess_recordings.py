"""Script to preprocess all recordings for CNN training."""

import argparse # package permet de modifier les arguments depuis un terminal
from pathlib import Path # pr mieux manipuler les chemins fichiers
from .preprocessing import AudioPreprocessor # importe la fonction Audioprocessor depuis le script preprocessing.py
from src.xenocanto.config import RECORDINGS_DIR # défini un chemin par défaut


def main():
    """Preprocess all recordings in the recordings directory.""" # parser = analyseur
    parser = argparse.ArgumentParser(description="Preprocess audio recordings for CNN training")
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default=RECORDINGS_DIR,
        help="Directory containing raw recordings",
    )#localisation des sons bruts
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preprocessed",
        help="Directory to save preprocessed data",
    )# localisation des sons traités
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate",
    )# qualité du son (fréquence d'échantillonage économise de la place)
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel filter banks",
    )# conversion en spectrogramme + hauteur de l'image (+ haut = + précis)
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration in seconds",
    )# durée de chaque enregistrment = longueur de l'image
    
    args = parser.parse_args() #récupère les réglages choisis
    
    recordings_dir = Path(args.recordings_dir)
    output_dir = Path(args.output_dir)
    # vérifie l'existence du dossier source
    if not recordings_dir.exists():
        print(f"Error: Recordings directory {recordings_dir} does not exist")
        return
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        duration=args.duration,
    )
    
    # Process each bird species directory, cherche les sous-dossier pour chaque espèce
    bird_dirs = [d for d in recordings_dir.iterdir() if d.is_dir()]
    
    if not bird_dirs:
        print(f"No bird directories found in {recordings_dir}")
        return
    
    print(f"Found {len(bird_dirs)} bird species directories")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    total_processed = 0
    
    for bird_dir in bird_dirs:
        bird_name = bird_dir.name
        print(f"\nProcessing {bird_name}...")
        
        saved_paths = preprocessor.process_directory(
            recordings_dir=bird_dir,
            output_dir=output_dir,
            bird_name=bird_name,
        )
        
        total_processed += len(saved_paths)
        print(f"  Processed {len(saved_paths)} files")
    
    print("\n" + "=" * 50)
    print(f"Total files processed: {total_processed}")
    print(f"Preprocessed data saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Use src.dataset.BirdDataset to load the preprocessed data")
    print("2. Create train/val/test splits using sklearn.model_selection.train_test_split")
    print("3. Train your CNN model!")


if __name__ == "__main__":
    main()

