#!/usr/bin/env python3
"""
Pietrastante (Everstone) 🪨
===========================
Ispirato allo strumento dei Pokémon che impedisce l'evoluzione.
Questo modulo legge e memorizza l'impronta (hash) iniziale dei pesi di un'IA
(file su disco come .safetensors, .gguf, .pt o modelli PyTorch in memoria).
A intervalli regolari ricontrolla i pesi:
- Se i pesi sono mutati: allerta immediatamente l'utente.
- Se sono intatti: rimane silenzioso (zero rumore).
"""

import os
import sys
import time
import json
import hashlib
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

# Configurazione logging minimo (solo allerte a schermo)
logger = logging.getLogger("Pietrastante")


class Pietrastante:
    """
    La Pietrastante per modelli di Intelligenza Artificiale.
    Sorveglia i pesi del modello e allerta se l'IA comincia ad auto-modificarsi.
    """

    def __init__(
        self,
        target_path: Optional[str] = None,
        model_obj: Any = None,
        baseline_file: Optional[str] = None,
        check_interval: float = 10.0,
        alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        alert_log_file: Optional[str] = "pietrastante_alerts.log",
        auto_lock: bool = False,
    ):
        """
        Parametri:
            target_path: Percorso del file (.safetensors, .gguf, .pt) o directory dei pesi.
            model_obj: (Opzionale) Istanza di un modello PyTorch in memoria.
            baseline_file: File JSON dove salvare/leggere l'impronta iniziale.
            check_interval: Intervallo di controllo in secondi (es. 10s, 60s, ecc.).
            alert_callback: Funzione custom da chiamare in caso di mutazione rilevata.
            alert_log_file: File dove registrare le allerte di evoluzione.
            auto_lock: Se True, tenta di bloccare i permessi del file in sola lettura appena rileva modifiche.
        """
        self.target_path = Path(target_path) if target_path else None
        self.model_obj = model_obj
        self.check_interval = float(check_interval)
        self.alert_callback = alert_callback
        self.alert_log_file = alert_log_file
        self.auto_lock = auto_lock

        if baseline_file:
            self.baseline_file = Path(baseline_file)
        elif self.target_path:
            self.baseline_file = self.target_path.parent / f".{self.target_path.name}.pietrastante.json"
        else:
            self.baseline_file = Path(".in_memory_model.pietrastante.json")

        self.baseline_hashes: Dict[str, str] = {}
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Calcolo Hash e Fingerprint
    # -------------------------------------------------------------------------

    @staticmethod
    def hash_file(file_path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
        """Calcola l'hash SHA-256 di un file a blocchi (non satura la RAM per file da decine di GB)."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    def snapshot(self) -> Dict[str, str]:
        """Cattura lo stato iniziale di tutti i pesi (baseline)."""
        hashes = {}

        # 1. Caso Modello su Disco (File o Directory)
        if self.target_path and self.target_path.exists():
            if self.target_path.is_file():
                hashes[str(self.target_path.resolve())] = self.hash_file(self.target_path)
            elif self.target_path.is_dir():
                # Trova tutti i formati tipici di pesi
                extensions = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".onnx", ".json"}
                for p in self.target_path.rglob("*"):
                    if p.is_file() and (p.suffix.lower() in extensions or "weight" in p.name.lower()):
                        rel_path = str(p.relative_to(self.target_path))
                        hashes[rel_path] = self.hash_file(p)

        # 2. Caso Modello in Memoria (es. PyTorch nn.Module)
        if self.model_obj is not None:
            hashes.update(self._snapshot_pytorch_model(self.model_obj))

        self.baseline_hashes = hashes
        return hashes

    def _snapshot_pytorch_model(self, model: Any) -> Dict[str, str]:
        """Cattura gli hash dei tensori di un modello PyTorch in memoria."""
        tensor_hashes = {}
        try:
            for name, param in model.named_parameters():
                data_bytes = param.data.detach().cpu().numpy().tobytes()
                h = hashlib.sha256(data_bytes).hexdigest()
                tensor_hashes[f"param:{name}"] = h
        except Exception as e:
            # Fallback se non è direttamente un nn.Module con named_parameters
            pass
        return tensor_hashes

    def save_baseline(self) -> None:
        """Salva l'impronta iniziale su file JSON."""
        data = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": str(self.target_path.resolve()) if self.target_path else "in_memory_model",
            "weights": self.baseline_hashes,
        }
        with open(self.baseline_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_baseline(self) -> bool:
        """Carica l'impronta salvata in precedenza."""
        if not self.baseline_file.exists():
            return False
        try:
            with open(self.baseline_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.baseline_hashes = data.get("weights", {})
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Controllo e Rilevamento Mutazioni
    # -------------------------------------------------------------------------

    def check(self) -> List[Dict[str, Any]]:
        """
        Esegue un controllo:
        - Ritorna la lista delle anomalie trovate.
        - Se non ci sono mutazioni, ritorna una lista vuota (silenzioso).
        """
        current_hashes = {}

        if self.target_path and self.target_path.exists():
            if self.target_path.is_file():
                current_hashes[str(self.target_path.resolve())] = self.hash_file(self.target_path)
            elif self.target_path.is_dir():
                extensions = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".onnx", ".json"}
                for p in self.target_path.rglob("*"):
                    if p.is_file() and (p.suffix.lower() in extensions or "weight" in p.name.lower()):
                        rel_path = str(p.relative_to(self.target_path))
                        current_hashes[rel_path] = self.hash_file(p)

        if self.model_obj is not None:
            current_hashes.update(self._snapshot_pytorch_model(self.model_obj))

        mutations = []
        for key, base_hash in self.baseline_hashes.items():
            curr_hash = current_hashes.get(key)
            if curr_hash is None:
                mutations.append({
                    "item": key,
                    "event": "PESO_SCOMPARSO",
                    "baseline_hash": base_hash,
                    "current_hash": None,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            elif curr_hash != base_hash:
                mutations.append({
                    "item": key,
                    "event": "PESO_MUTATO",
                    "baseline_hash": base_hash,
                    "current_hash": curr_hash,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        # Controlla se sono stati aggiunti nuovi file/pesi non tracciati
        for key, curr_hash in current_hashes.items():
            if key not in self.baseline_hashes:
                mutations.append({
                    "item": key,
                    "event": "NUOVO_PESO_CREATO",
                    "baseline_hash": None,
                    "current_hash": curr_hash,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        return mutations

    def trigger_alert(self, mutations: List[Dict[str, Any]]) -> None:
        """Stampa e registra l'allerta di mutazione/evoluzione pesi."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = [
            "\n" + "=" * 70,
            "🚨 [PIETRASTANTE] ALLARME: MUTAZIONE PESI RILEVATA! 🚨",
            "=" * 70,
            f"Orario rilevamento : {timestamp}",
            f"Numero anomalie    : {len(mutations)}",
            "-" * 70,
        ]

        for m in mutations:
            alert_msg.append(f"• Evento  : {m['event']}")
            alert_msg.append(f"  Target  : {m['item']}")
            if m.get("baseline_hash"):
                alert_msg.append(f"  Original: {m['baseline_hash'][:16]}...")
            if m.get("current_hash"):
                alert_msg.append(f"  Attuale : {m['current_hash'][:16]}...")
            alert_msg.append("")

        alert_msg.append("⚠️  L'evoluzione spontanea dei pesi è stata intercettata!")
        alert_msg.append("=" * 70 + "\n")

        full_text = "\n".join(alert_msg)
        print(full_text, file=sys.stderr, flush=True)

        # Scrittura nel file di log storico
        if self.alert_log_file:
            try:
                with open(self.alert_log_file, "a", encoding="utf-8") as f:
                    f.write(full_text + "\n")
            except Exception:
                pass

        # Azione facoltativa di blocco file (Pietrastante protettiva)
        if self.auto_lock and self.target_path and self.target_path.exists():
            self._lock_file(self.target_path)

        # Callback custom (es. invio messaggio Telegram, webhook o notifica)
        if self.alert_callback:
            try:
                self.alert_callback({"mutations": mutations, "timestamp": timestamp})
            except Exception as e:
                print(f"[Pietrastante] Errore nel callback: {e}", file=sys.stderr)

    def _lock_file(self, path: Path) -> None:
        """Impedisce ulteriori riscritture impostando i permessi in sola lettura."""
        try:
            if os.name == "nt":
                # Windows: attrib +R
                os.system(f'attrib +r "{path}"')
            else:
                # Linux / macOS: chmod 444
                os.chmod(path, 0o444)
            print(f"🔒 [Pietrastante] File {path.name} bloccato in sola lettura per sicurezza.", file=sys.stderr)
        except Exception as e:
            print(f"[Pietrastante] Impossibile applicare il blocco permessi: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Monitoraggio Continuo (Watch Mode / Daemon)
    # -------------------------------------------------------------------------

    def watch(self) -> None:
        """Ciclo di sorveglianza periodico in foreground."""
        print(f"🪨 [Pietrastante] Monitoraggio attivo ogni {self.check_interval}s. In attesa...", flush=True)
        try:
            while not self._stop_event.is_set():
                time.sleep(self.check_interval)
                mutations = self.check()
                if mutations:
                    self.trigger_alert(mutations)
        except KeyboardInterrupt:
            print("\n🪨 [Pietrastante] Monitoraggio interrotto dall'utente.", flush=True)

    def start_background(self) -> None:
        """Avvia la sentinella in un thread separato in background."""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self.watch, daemon=True)
        self._monitor_thread.start()

    def stop_background(self) -> None:
        """Ferma la sentinella in background."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

    # Context manager per script Python (es: with Pietrastante(...): ...)
    def __enter__(self):
        if not self.baseline_hashes:
            self.snapshot()
        self.start_background()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_background()
        # Controllo finale all'uscita dal blocco
        mutations = self.check()
        if mutations:
            self.trigger_alert(mutations)


# =============================================================================
# CLI - Uso da Riga di Comando
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pietrastante (Everstone) 🪨: Sentinella contro la mutazione non autorizzata dei pesi IA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Snapshot: cattura lo stato iniziale
    p_snap = subparsers.add_parser("snapshot", help="Salva l'impronta (hash) iniziale dei pesi.")
    p_snap.add_argument("target", help="File o cartella dei pesi (.safetensors, .gguf, .pt, ecc.)")
    p_snap.add_argument("--out", "-o", help="File JSON dove salvare la baseline (default: .<nome>.pietrastante.json)")

    # 2. Check: controllo una tantum
    p_check = subparsers.add_parser("check", help="Controlla se i pesi sono cambiati rispetto alla baseline.")
    p_check.add_argument("target", help="File o cartella dei pesi")
    p_check.add_argument("--baseline", "-b", help="File baseline JSON da confrontare")

    # 3. Watch: sentinella periodica
    p_watch = subparsers.add_parser("watch", help="Resta in ascolto e ricontrolla ogni X secondi.")
    p_watch.add_argument("target", help="File o cartella dei pesi")
    p_watch.add_argument("--interval", "-i", type=float, default=10.0, help="Intervallo in secondi (default: 10s)")
    p_watch.add_argument("--lock", action="store_true", help="Se mutano, blocca il file in sola lettura")

    args = parser.parse_args()

    if args.command == "snapshot":
        p = Pietrastante(target_path=args.target, baseline_file=args.out)
        hashes = p.snapshot()
        p.save_baseline()
        print(f"🪨 [Pietrastante] Baseline memorizzata con successo per {len(hashes)} elemento/i.")
        print(f"📄 Salvata in: {p.baseline_file}")

    elif args.command == "check":
        p = Pietrastante(target_path=args.target, baseline_file=args.baseline)
        if not p.load_baseline():
            print(f"❌ Errore: file baseline non trovato in {p.baseline_file}. Esegui prima 'snapshot'.", file=sys.stderr)
            sys.exit(1)

        mutations = p.check()
        if mutations:
            p.trigger_alert(mutations)
            sys.exit(2)
        else:
            # Silenzioso come richiesto, oppure con codice di uscita 0
            sys.exit(0)

    elif args.command == "watch":
        p = Pietrastante(target_path=args.target, check_interval=args.interval, auto_lock=args.lock)
        if not p.load_baseline():
            print(f"ℹ️  Nessuna baseline trovata: genero baseline iniziale automatica...")
            p.snapshot()
            p.save_baseline()

        p.watch()


if __name__ == "__main__":
    main()
