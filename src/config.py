# Configuración central.
# Todo se puede sobrescribir con variables de entorno o con un fichero .env
# (copia .env.example a .env). Las variables de entorno reales tienen prioridad.
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Cargador mínimo de .env (sin dependencias). No pisa variables ya definidas."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(PROJECT_ROOT / ".env")
_load_dotenv(Path.cwd() / ".env")


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        print(f"⚠️ {name}={raw!r} no es un entero válido; uso {default}", file=sys.stderr)
        return int(default)


def _float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ {name}={raw!r} no es un número válido; uso {default}", file=sys.stderr)
        return float(default)


# ============================================================
# RUTAS
# ============================================================
_hmem_home = os.getenv("HMEM_HOME", "").strip()
if _hmem_home:
    BODY = Path(_hmem_home).expanduser().resolve()
elif (PROJECT_ROOT / "pyproject.toml").exists():  # checkout del repositorio / pip install -e
    BODY = PROJECT_ROOT / "body"
else:  # instalacion normal: no escribir dentro de site-packages
    BODY = Path.home() / ".hmem"
LTM_DIR = BODY / "largo_plazo"
MEM_DIR = LTM_DIR  # Alias para compatibilidad hacia atrás
STM_DIR = BODY / "corto_plazo"
LOG_DIR = BODY / "logs"
BACKUP_DIR = BODY / "backups"
CARTRIDGES_DIR = BODY / "cartuchos"

# ============================================================
# INTERFAZ WEB
# ============================================================
UI_PORT = _int("UI_PORT", "8765")

# ============================================================
# INFRAESTRUCTURA LLM (GPU 0 = LTM, GPU 1 = STM + Agente)
# Si SERVER_IP es 127.0.0.1/localhost los comandos se ejecutan en local;
# en cualquier otro caso se lanzan por SSH (clave sin contraseña).
# Con SERVER_AUTOSTART=0 (por defecto) solo se comprueba que haya un
# endpoint OpenAI-compatible escuchando: sirve llama-server, vLLM, Ollama...
# ============================================================
SERVER_USER = os.getenv("SERVER_USER", os.getenv("USER", ""))
SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", "llama-server")
LLAMA_SERVER_EXTRA = os.getenv("LLAMA_SERVER_EXTRA", "")
REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "")
REMOTE_MODEL_ALIAS = os.getenv("REMOTE_MODEL_ALIAS", "local-model")
REMOTE_THREADS = _int("REMOTE_THREADS", "8")
SERVER_AUTOSTART = _flag("SERVER_AUTOSTART", "0")
SERVER_KEEP_ALIVE = _flag("SERVER_KEEP_ALIVE", "0")
SERVER_HEALTH_TIMEOUT_S = _float("SERVER_HEALTH_TIMEOUT_S", "180.0")

# GPU 0: Memoria a Largo Plazo
GPU0_INDEX = os.getenv("GPU0_INDEX", "0")
SERVER_PORT_GPU0 = _int("SERVER_PORT_GPU0", "8087")
REMOTE_PID_GPU0 = os.getenv("REMOTE_PID_GPU0", "/tmp/hmem-gpu0.pid")
REMOTE_LOG_GPU0 = os.getenv("REMOTE_LOG_GPU0", "/tmp/hmem-gpu0.log")
REMOTE_CTX_GPU0 = _int("REMOTE_CTX_GPU0", "32768")
LLM_LTM_URL = os.getenv("LLM_LTM_URL", f"http://{SERVER_IP}:{SERVER_PORT_GPU0}/v1")

# GPU 1: Memoria a Corto Plazo y Respuestas de Agente
GPU1_INDEX = os.getenv("GPU1_INDEX", "1")
SERVER_PORT_GPU1 = _int("SERVER_PORT_GPU1", "8088")
REMOTE_PID_GPU1 = os.getenv("REMOTE_PID_GPU1", "/tmp/hmem-gpu1.pid")
REMOTE_LOG_GPU1 = os.getenv("REMOTE_LOG_GPU1", "/tmp/hmem-gpu1.log")
SLOT_CTX_GPU1 = _int("STM_SLOT_CTX", "40960")
PARALLEL_SLOTS_GPU1 = _int("STM_SLOTS", "2")
LLM_STM_URL = os.getenv("LLM_STM_URL", f"http://{SERVER_IP}:{SERVER_PORT_GPU1}/v1")

# Defaults para LLM
LLM_BASE_URL = LLM_LTM_URL
LLM_MODEL = REMOTE_MODEL_ALIAS
REQUEST_TIMEOUT_S = _int("REQUEST_TIMEOUT_S", "240")
THINKING = _flag("THINKING", "0")

# ============================================================
# MEMORIA A CORTO PLAZO (STM)
# ============================================================
STM_MAX_TOKENS = _int("STM_MAX_TOKENS", "30000")
STM_TRIGGER_RATIO = _float("STM_TRIGGER_RATIO", "0.90")
STM_TARGET_RATIO = _float("STM_TARGET_RATIO", "0.60")
STM_KEEP_RECENT = _int("STM_KEEP_RECENT", "6")
STM_CHUNK_TOKENS = _int("STM_CHUNK_TOKENS", "6000")
STM_SUMMARY_RATIO = _float("STM_SUMMARY_RATIO", "0.25")
STM_MIN_SUMMARY_TOKENS = _int("STM_MIN_SUMMARY_TOKENS", "120")
STM_MAX_COMPACT_ROUNDS = _int("STM_MAX_COMPACT_ROUNDS", "12")
STM_COMPACT_POLL_S = _float("STM_COMPACT_POLL_S", "5")
STM_PROMPT_MODE = os.getenv("STM_PROMPT_MODE", "retrieval").strip().lower()
STM_PROMPT_RECENT = _int("STM_PROMPT_RECENT", "12")
STM_LOOKUP_K = _int("STM_LOOKUP_K", "6")
STM_EXACT_TOKENS = _flag("STM_EXACT_TOKENS", "1")
STM_CHARS_PER_TOKEN = _float("STM_CHARS_PER_TOKEN", "3.2")
STM_MSG_OVERHEAD = _int("STM_MSG_OVERHEAD", "6")
# 4096 (antes 1500): con 1500 los textos largos (p. ej. un cuento de 1000 palabras) se cortaban.
ANSWER_MAX_TOKENS = _int("STM_ANSWER_MAX_TOKENS", "4096")
ANSWER_TEMPERATURE = _float("STM_TEMPERATURE", "0.4")

# ============================================================
# MEMORIA A LARGO PLAZO (LTM)
# ============================================================
MEM_MAX_TOKENS = _int("MEM_MAX_TOKENS", "1500")
MEM_REQUEST_TIMEOUT_S = _int("MEM_REQUEST_TIMEOUT_S", "180")
MEM_THINKING = _flag("MEM_THINKING", "0")
MEM_TAU = _float("MEM_TAU", "20")
MEM_ARCHIVE_BELOW = _float("MEM_ARCHIVE_BELOW", "0.5")
MEM_MIN_AGE = _int("MEM_MIN_AGE", "10")
MEM_BEAM = _int("MEM_BEAM", "2")  # anchura del beam en la navegacion por el arbol
MEM_NAV_MAX_CALLS = _int("MEM_NAV_MAX_CALLS", "40")  # max. llamadas LLM por consulta (solo con MEM_NAV_LLM=1)
MEM_MAX_CHILDREN = _int("MEM_MAX_CHILDREN", "12")
MEM_LIST_LIMIT = _int("MEM_LIST_LIMIT", "30")
MEM_DESCRIBE_BATCH = _int("MEM_DESCRIBE_BATCH", "4")
MEM_REVISIT_TICKS = _int("MEM_REVISIT_TICKS", "1000")
MEM_IDLE_SLEEP_S = _float("MEM_IDLE_SLEEP_S", "3")
MEM_IDLE_MAX_SLEEP_S = _float("MEM_IDLE_MAX_SLEEP_S", "60")
MEM_BACKUP_EVERY = _int("MEM_BACKUP_EVERY", "200")
MEM_BACKUP_KEEP = _int("MEM_BACKUP_KEEP", "10")
MEM_RECALL_EXPAND = _flag("MEM_RECALL_EXPAND", "1")   # expansión semántica con LLM si FTS5 no halla nada
RRF_K = _int("RRF_K", "60")                            # constante de Reciprocal Rank Fusion

# Integración inter-memorias
LTM_ENABLED = _flag("LTM_ENABLED", "1")
LTM_DIRECT_RETRIEVAL = _flag("LTM_DIRECT_RETRIEVAL", "1")   # 1 = inyecta los hechos recuperados sin llamada LLM de sintesis (ver README: latencia medida)
# [PARCHE13] hechos inyectados en modo directo (antes fijo en 6). En LoCoMo, 110 de 620 fallos tenían la evidencia
# en los puestos 7-18. [PARCHE14] 10 por defecto: en LoCoMo, correctas 59,7 % → 61,9 % (p = 0,003) sin subir la
# invención (6,3 % → 7,0 %, p = 0,65); corpus propio 100 % / 100 % / 0 %.
LTM_DIRECT_TOP_K = _int("LTM_DIRECT_TOP_K", "10")
LTM_MIN_CHARS = _int("LTM_MIN_CHARS", "6")
LTM_INGEST_USER = _flag("LTM_INGEST_USER", "1")         # extraer hechos de los mensajes del usuario
LTM_INGEST_ASSISTANT = _flag("LTM_INGEST_ASSISTANT", "0")
LTM_INGEST_QUEUE_MAX = _int("LTM_INGEST_QUEUE_MAX", "200")

LTM_DB_PATH = LTM_DIR / "memoria.db"


# ============================================================
# NAVEGACIÓN POR EL ÁRBOL Y MEMORIA EPISÓDICA
# ============================================================
MEM_NAV = _flag("MEM_NAV", "1")            # navegación por el árbol en cada consulta (≈1 llamada LLM por nivel)
EPI_ENABLED = _flag("EPI_ENABLED", "1")    # registro literal de toda la conversación (SQLite + FTS5)
EPI_LOOKUP_K = _int("EPI_LOOKUP_K", "22")   # episodios literales candidatos para el prompt ([PARCHE17] antes 6)
# [PARCHE16] episodios que llegan al prompt (antes fijo en 5 en agente._build_system). Debe ser ≤ EPI_LOOKUP_K
# (los repetidos con el corto plazo se descartan). [PARCHE17] 20 por defecto: en LoCoMo, correctas 61,9 % → 70,2 %
# (p < 0,001; multi-salto 35,5 → 47,5 %) sin subir la invención (7,0 → 5,8 %, p = 0,46). Coste: prompt ≈ +1.800 tokens.
EPI_PROMPT_K = _int("EPI_PROMPT_K", "20")
# [PARCHE19] descomposición de la pregunta en subconsultas (1 llamada LLM al servidor de memoria larga) y fusión
# RRF de lo recuperado con cada una, SIN subir el presupuesto (LTM_DIRECT_TOP_K hechos, EPI_PROMPT_K episodios).
# 0 por defecto hasta que un experimento pareado en LoCoMo lo justifique (ver src/descomponer.py).
LTM_DECOMPOSE = _flag("LTM_DECOMPOSE", "0")
LTM_DECOMPOSE_MAX = _int("LTM_DECOMPOSE_MAX", "3")
# [PARCHE20] (ver src/refinar.py) · ambas desactivadas por defecto hasta un experimento pareado en LoCoMo
EPI_DIVERSITY = os.getenv("EPI_DIVERSITY", "off").strip().lower()   # off | dia (≤ EPI_PER_DAY episodios por día)
EPI_PER_DAY = _int("EPI_PER_DAY", "3")
EPI_DIVERSITY_POOL = _int("EPI_DIVERSITY_POOL", "3")    # se buscan EPI_LOOKUP_K × este factor candidatos
LTM_SECOND_PASS = _flag("LTM_SECOND_PASS", "0")         # 2.ª pasada: 1 llamada LLM con lo recuperado


def _episode_limits(lookup: int, prompt: int) -> tuple[int, int]:  # [PARCHE18]
    """Garantiza que se BUSCAN al menos los episodios que se muestran (+2 por los repetidos con el corto plazo).
    Un .env antiguo con EPI_LOOKUP_K=6 limitaba en silencio el prompt a 6 episodios aunque EPI_PROMPT_K fuese 20."""
    prompt = max(1, prompt)
    return (lookup if lookup >= prompt else prompt + 2), prompt


_lookup_cfg = EPI_LOOKUP_K
EPI_LOOKUP_K, EPI_PROMPT_K = _episode_limits(EPI_LOOKUP_K, EPI_PROMPT_K)
if EPI_LOOKUP_K != _lookup_cfg:
    print(f"⚠️ EPI_LOOKUP_K={_lookup_cfg} es menor que EPI_PROMPT_K={EPI_PROMPT_K}; uso {EPI_LOOKUP_K} "
          "(revisa tu .env: probablemente es un valor antiguo)", file=sys.stderr)


# ============================================================
# EMBEDDINGS (GPU 2)
# ============================================================
EMB_ENABLED = _flag("EMB_ENABLED", "1")
GPU2_INDEX = os.getenv("GPU2_INDEX", "2")
SERVER_PORT_GPU2 = _int("SERVER_PORT_GPU2", "8089")
REMOTE_PID_GPU2 = os.getenv("REMOTE_PID_GPU2", "/tmp/hmem-gpu2.pid")
REMOTE_LOG_GPU2 = os.getenv("REMOTE_LOG_GPU2", "/tmp/hmem-gpu2.log")
EMB_MODEL_PATH = os.getenv("EMB_MODEL_PATH", "")            # .gguf de bge-m3 (solo con SERVER_AUTOSTART=1)
EMB_MODEL_ALIAS = os.getenv("EMB_MODEL_ALIAS", "bge-m3")
EMB_SERVER_EXTRA = os.getenv("EMB_SERVER_EXTRA", "")
EMB_CTX = _int("EMB_CTX", "8192")
EMB_URL = os.getenv("EMB_URL", f"http://{SERVER_IP}:{SERVER_PORT_GPU2}/v1")
EMB_TIMEOUT_S = _float("EMB_TIMEOUT_S", "20")
EMB_BATCH = _int("EMB_BATCH", "16")                         # textos por petición
EMB_BATCH_PER_TICK = _int("EMB_BATCH_PER_TICK", "32")       # vectores nuevos por tick del ciclo de sueño
EMB_TOP_K = _int("EMB_TOP_K", "25")
EMB_MIN_SIM = _float("EMB_MIN_SIM", "0.38")                 # coseno mínimo para entrar en el ranking (calibra con scripts/evaluar.py)
EMB_PLACE = _flag("EMB_PLACE", "1")                         # colocar hechos nuevos junto a su vecino más parecido
EMB_PLACE_SIM = _float("EMB_PLACE_SIM", "0.62")             # >= : se coloca junto al vecino sin llamar al LLM
EMB_PLACE_MID = _float("EMB_PLACE_MID", "0.45")             # >= : comparte los primeros EMB_PLACE_SHARE niveles
EMB_PLACE_SHARE = _int("EMB_PLACE_SHARE", "3")              # 0 desactiva el reparto parcial
EMB_DEDUPE_SIM = _float("EMB_DEDUPE_SIM", "0.88")           # coseno que dispara la comprobación de duplicado con LLM
EMB_NAV_MARGIN = _float("EMB_NAV_MARGIN", "0.10")           # ventaja de coseno con la que se evita la llamada LLM al navegar


# ============================================================
# PROFUNDIDAD ADAPTATIVA DEL ÁRBOL
# ============================================================
MEM_ADAPTIVE_DEPTH = _flag("MEM_ADAPTIVE_DEPTH", "1")   # 0 = comportamiento antiguo (siempre 7 niveles)
MEM_MAX_DEPTH = _int("MEM_MAX_DEPTH", "5")               # tope de niveles al CREAR ramas (task_split puede bajar hasta 7)

# ============================================================
# BLINDAJE SEMÁNTICO DE FUSIONES
# ============================================================
MEM_MERGE_MIN_SIM = _float("MEM_MERGE_MIN_SIM", "0.50")  # Coseno mínimo para evaluar fusión

# 0 = navegación 100% vectorial/léxica (Zero-LLM, ultra-rápida)
MEM_NAV_LLM = _flag("MEM_NAV_LLM", "0")


# ============================================================
# PESOS DE LA FUSIÓN RRF POR CANAL
# ============================================================
RRF_W_EXACT = _float("RRF_W_EXACT", "0.6")   # Coincidencia literal fuerte
RRF_W_FTS = _float("RRF_W_FTS", "0.25")      # Atenuado para no canibalizar embeddings (antes 0.5)
RRF_W_EMB = _float("RRF_W_EMB", "2.5")       # Canal principal preponderante (antes 1.0)
RRF_W_SEM = _float("RRF_W_SEM", "0.3")       # Expansión semántica LLM (antes 0.5)
RRF_W_TREE = _float("RRF_W_TREE", "0.1")     # Árbol (antes 0.3)
RRF_W_SYN = _float("RRF_W_SYN", "0.3")       # Sinapsis hebbianas (valor efectivo medido en el benchmark)

# Umbral de abstención para la compuerta de relevancia
EMB_ABSTAIN_SIM = _float("EMB_ABSTAIN_SIM", "0.58")
ABSTAIN_OV1_SIM = _float("ABSTAIN_OV1_SIM", "0.48")   # 1 término de contenido compartido: similitud mínima para responder
ABSTAIN_OV2_SIM = _float("ABSTAIN_OV2_SIM", "0.35")
# Umbral inferior para la banda tentativa (Modo C de dos niveles)
EMB_TENTATIVE_SIM = _float("EMB_TENTATIVE_SIM", "0.40")   # >=2 términos compartidos: similitud mínima para responder

# ============================================================
# ENLACES ASOCIATIVOS HORIZONTALES (SINAPSIS HEBBIANAS)
# ============================================================
SYN_ENABLED = _flag("SYN_ENABLED", "1")              # Activar red sináptica asociativa
SYN_SPREAD_LIMIT = _int("SYN_SPREAD_LIMIT", "6")     # Máximas células activadas por propagación
SYN_MIN_WEIGHT = _float("SYN_MIN_WEIGHT", "1.2")     # Peso mínimo para propagar activación
SYN_TAU = _float("SYN_TAU", "80.0")                  # Constante de tiempo para decaimiento sináptico
SYN_PRUNE_BELOW = _float("SYN_PRUNE_BELOW", "0.3")   # Umbral de poda para sinapsis inactivas

# ============================================================
# PERFIL DE RECUPERACIÓN (ablación del 27/10/2026)  [RETRIEVAL_PROFILE_V2]
# "embeddings" (defecto): FIJA los pesos RRF (solo canal vectorial) y desactiva la expansión LLM.
#     Ignora RRF_W_* / MEM_RECALL_EXPAND del entorno o de .env (y lo avisa por stderr).
# "completo" / "personalizado": usa tus RRF_W_* y MEM_RECALL_EXPAND (entorno, .env o valores por defecto).
# Si una consulta no obtiene candidatos vectoriales, retrieval.py reactiva el resto de canales.
# ============================================================
RETRIEVAL_PROFILE = os.getenv("RETRIEVAL_PROFILE", "embeddings").strip().lower()
if RETRIEVAL_PROFILE == "embeddings":
    _ignorados = [k for k in ("RRF_W_EXACT", "RRF_W_FTS", "RRF_W_SEM", "RRF_W_TREE", "RRF_W_SYN",
                              "MEM_RECALL_EXPAND") if k in os.environ]
    if _ignorados:
        print("⚠ RETRIEVAL_PROFILE=embeddings ignora " + ", ".join(_ignorados)
              + " (usa RETRIEVAL_PROFILE=personalizado para respetarlos)", file=sys.stderr)
    RRF_W_EXACT = RRF_W_FTS = RRF_W_SEM = RRF_W_TREE = RRF_W_SYN = 0.0
    MEM_RECALL_EXPAND = False

# ============================================================
# RERANKER  [PARCHE4]
#   off   = sin reranker (compuerta clásica: embeddings + solapamiento)
#   llm   = juez LLM: el modelo de la memoria larga decide qué candidatos responden (1 llamada por consulta)
#   cross = cross-encoder (bge-reranker-v2-m3) servido por llama-server --reranking en RERANK_URL
# ============================================================
# [PARCHE8] off por defecto: a 10.108 hechos off y llm dan el mismo OBJETIVO (97,3 % · usuario 100 % · inventa 3,2 %)
# y llm cuesta 2,2× de latencia + 1 llamada LLM por consulta. RERANK_MODE=llm sigue disponible en .env.
RERANK_MODE = os.getenv("RERANK_MODE", "off").strip().lower()
SERVER_PORT_RERANK = _int("SERVER_PORT_RERANK", "8090")
RERANK_URL = os.getenv("RERANK_URL", f"http://{SERVER_IP}:{SERVER_PORT_RERANK}/v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_TOP_K = _int("RERANK_TOP_K", "8")             # candidatos que se reordenan
RERANK_MIN = _float("RERANK_MIN", "0.5")             # cross: puntuación 0-1 mínima para que un recuerdo "responda"
RERANK_NONE = os.getenv("RERANK_NONE", "tentative").strip().lower()   # si nada responde: tentative | abstain
RERANK_SIGMOID = os.getenv("RERANK_SIGMOID", "1")     # llama-server devuelve logits: se pasan a 0-1
RERANK_TIMEOUT_S = _float("RERANK_TIMEOUT_S", "60")
