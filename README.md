# Pietrastante.py
e se il tuo agente si evolvesse? Questo è uno script per essere informato se il tuo agente modifica i pesi da solo .

🪨 Pietrastante (Everstone)

Sentinella contro l'auto-modifica e la deriva non autorizzata dei pesi nei modelli di Intelligenza Artificiale.

Ispirato alla celebre Pietrastante dei Pokémon (lo strumento tenuto che blocca l'evoluzione spontanea), questo strumento:

Legge e salva l'impronta iniziale (hash SHA-256) dei pesi (supporta .safetensors, .gguf, .pt, .bin, intere cartelle di checkpoint o modelli PyTorch in memoria).
Sorveglia periodicamente a intervalli regolari (es. ogni 10 o 60 secondi).
Se i pesi cambiano: emette un allarme dettagliato (orario, target mutato, impronta precedente vs attuale) e scrive su log (pietrastante_alerts.log).
Se sono identici: rimane completamente silenzioso (zero rumore a terminale).
🚀 Utilizzo Rapido (CLI)
1. Salva l'impronta iniziale dei pesi
bash
python pietrastante.py snapshot /percorso/al/modello.safetensors

(Crea un file di baseline leggero .modello.safetensors.pietrastante.json)

2. Controllo Periodico Automatico (Watch)
bash
# Ricontrolla ogni 30 secondi; allerta solo se mutano
python pietrastante.py watch /percorso/al/modello.safetensors --interval 30
# Con blocco protettivo del file in sola lettura in caso di mutazione:
python pietrastante.py watch /percorso/al/modello.gguf --interval 10 --lock
3. Controllo Singolo Istantaneo (Check)
bash
python pietrastante.py check /percorso/al/modello.safetensors
# Uscita 0 se integro (silenzioso)
# Uscita 2 con banner di allarme se mutato
🐍 Utilizzo in Script Python / Agent Loop

Puoi importare Pietrastante direttamente nel codice in cui gira l'agente o il modello:

python
from pietrastante import Pietrastante
# 1. Utilizzo come Context Manager durante l'inferenza o il loop dell'agente:
with Pietrastante(target_path="weights.safetensors", check_interval=5.0):
    run_autonomous_agent_loop()
    # Se l'agente prova a sovrascrivere i pesi, scatta l'allerta immediata!
# 2. Oppure per modelli PyTorch in memoria (VRAM/RAM):
sentinel = Pietrastante(model_obj=my_pytorch_model, check_interval=15.0)
sentinel.snapshot()
sentinel.start_background()
# ... Il modello esegue le sue operazioni ...
sentinel.stop_background()
