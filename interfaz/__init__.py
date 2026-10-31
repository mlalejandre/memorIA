# interfaz/__init__.py
import atexit
import os
import sys
import threading

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
# [PARCHE22] siempre desde el repositorio: una instalación editable (pip install -e .) solo conoce los módulos que
# existían al instalarla, y los añadidos después (descomponer, refinar...) fallaban con ModuleNotFoundError.
if os.path.isdir(SRC_DIR):
    for p in (SRC_DIR, ROOT_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)

from agente import Agent
from servidor import start_all_servers, stop_all_servers
from .frontend import run_server
from config import UI_PORT

def _boot(agent: Agent) -> None:
    try:
        agent.log("Comprobando servidores en GPU 0 (LTM) y GPU 1 (STM)...")
        start_all_servers()
        agent.server_ready = True
        agent.log("✓ Ambas GPUs verificadas y listas.")
    except Exception as exc:
        agent.log(f"❌ Error al iniciar servidores: {exc}")
    try:   # [PARCHE31] nombre del modelo de cada servidor para la cabecera
        import modelos
        modelos.refresh_cache()
    except Exception:
        pass
    
    agent.start_background_tasks()
    agent.log(f"Memoria a corto plazo cargada: {agent.stm.total_tokens()} tokens.")
    agent.log(f"Memoria a largo plazo cargada: {agent.ltm.summary()}")

def iniciar():
    agent = Agent()
    atexit.register(stop_all_servers)
    threading.Thread(target=_boot, args=(agent,), daemon=True).start()
    run_server(agent, port=UI_PORT)

__all__ = ["iniciar"]