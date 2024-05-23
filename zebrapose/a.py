import socket

import cv2
import numpy as np
import pdb

# Definisci l'indirizzo IP e la porta su cui il server ascolterà
HOST = '127.0.0.1'  # Indirizzo IP localhost
PORT = 6543  # Porta arbitraria non utilizzata

# Crea un oggetto socket TCP/IP
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # Associa il socket all'indirizzo IP e alla porta definiti
        s.bind((HOST, PORT))
        # Metti il server in ascolto per una connessione in entrata
        s.listen()
        print("Server in ascolto...")
        # Accetta la connessione quando viene ricevuta
        conn, addr = s.accept()
        with conn:
            print('Connesso da:', addr)
            while True:
                # Ricevi dati dal client
                n = conn.recv(1024)
                if not n:
                    break
                n = int.from_bytes(n, "little", signed=True)
                #---bad method of read all data---
                data = bytes()
                pdb.set_trace()
                while len(data) < n:
                    moreData = conn.recv(n - len(data))
                    if not moreData:
                        break
                    data += moreData
                #----- --- -- -- -- --- --- -- ----

                # Stampa i dati ricevuti
                nparr = np.frombuffer(data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imshow("prova", image)
                conn.send("OK".encode())
    except Exception as e:
        print(e)
        s.close()
