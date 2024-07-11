import os

def rinomina_file(cartella):
    # Ottieni la lista di tutti i file nella cartella specificata
    lista_file = os.listdir(cartella)

    # Itera su ciascun file e rinominalo
    for nome_file in lista_file:
        # Costruisci il nuovo nome del file con il suffisso _000000
        nuovo_nome = os.path.splitext(nome_file)[0] + "_000000" + os.path.splitext(nome_file)[1]

        # Rinomina il file
        vecchio_percorso = os.path.join(cartella, nome_file)
        nuovo_percorso = os.path.join(cartella, nuovo_nome)
        os.rename(vecchio_percorso, nuovo_percorso)

        print(f"File rinominato: {nuovo_nome}")

print("aa")

# Specifica la cartella in cui desideri rinominare i file
cartella_da_rinominare = "../BOP ROOT/laryngoscope/train/000000/mask"
rinomina_file(cartella_da_rinominare)

cartella_da_rinominare = "../BOP ROOT/laryngoscope/train/000000/mask_visib"
rinomina_file(cartella_da_rinominare)