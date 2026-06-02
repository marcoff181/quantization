import os
import csv
import torch
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

# ==========================================
# CONFIGURAZIONE (Modifica questi parametri)
# ==========================================
IMAGE_FOLDER = "/media/SSD_4TB/crispy_storage/comparative_images_img2img/"  # Inserisci il percorso della tua cartella
OUTPUT_CSV = "analisi_dettagliata_parametri.csv"
# ==========================================

def estrai_parametri(nome_file):
    """
    Estrae i parametri dal nome file formattato come:
    00000_sd3_fp8_seed123_s50_g6.5_str0.6.png
    """
    nome_senza_estensione = nome_file.rsplit('.', 1)[0]
    parti = nome_senza_estensione.split('_')
    
    # Se il file non rispetta esattamente la struttura a 7 parti, restituiamo campi vuoti
    if len(parti) >= 7:
        try:
            return {
                "ID_Immagine": parti[0],
                "Modello": parti[1],
                "Precisione": parti[2],
                "Seed": parti[3].replace('seed', ''),
                "Steps": int(parti[4].replace('s', '')),
                "Guidance_CFG": float(parti[5].replace('g', '')),
                "Strength": float(parti[6].replace('str', ''))
            }
        except ValueError:
            pass
            
    # Fallback per file con nomi diversi
    return {
        "ID_Immagine": "N/D", "Modello": "N/D", "Precisione": "N/D", 
        "Seed": "N/D", "Steps": "N/D", "Guidance_CFG": "N/D", "Strength": "N/D"
    }

def main():
    print("Caricamento del giudice AI (Cafe Aesthetic)...")
    
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("GPU rilevata! L'analisi sarà veloce.")
    else:
        print("Nessuna GPU rilevata. Utilizzo della CPU.")
    
    scorer = pipeline("image-classification", model="cafeai/cafe_aesthetic", device=device)
    
    estensioni_valide = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(estensioni_valide)]
    
    if not image_files:
        print(f"ERRORE: Nessuna immagine trovata nella cartella {IMAGE_FOLDER}")
        return
        
    print(f"\nTrovate {len(image_files)} immagini. Inizio l'analisi e l'estrazione dei dati...")
    
    risultati = []
    
    for filename in tqdm(image_files, desc="Valutazione"):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            voti = scorer(img)
            
            punteggio_estetico = next(item['score'] for item in voti if item['label'] == 'aesthetic')
            punteggio_finale = round(punteggio_estetico * 100, 2)
            
            # Estraiamo i dati dal nome del file
            dati_estratti = estrai_parametri(filename)
            
            # Creiamo la riga completa unendo nome, voto e parametri estratti
            riga = {
                "Nome_File": filename,
                "Voto_Estetico_su_100": punteggio_finale
            }
            riga.update(dati_estratti) # Aggiunge Modello, Seed, Steps, ecc.
            
            risultati.append(riga)
            
        except Exception as e:
            print(f"\nImpossibile leggere l'immagine {filename}. Errore: {e}")
    
    # Ordina dal voto più alto al più basso
    risultati.sort(key=lambda x: x["Voto_Estetico_su_100"], reverse=True)
    
    # Intestazioni delle colonne per il file CSV
    colonne = [
        "Nome_File", "Voto_Estetico_su_100", "Modello", 
        "Precisione", "Seed", "Steps", "Guidance_CFG", "Strength", "ID_Immagine"
    ]
    
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=colonne)
        writer.writeheader()
        writer.writerows(risultati)
        
    print(f"\nAnalisi completata! File salvato come: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()