#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prueba.py — Banco de pruebas con evidencia medible para la memoria jerárquica.

Se ejecuta desde la RAÍZ del proyecto:

    python3 benchmarks/prueba.py                      # modo emulado: LLM y embeddings simulados (rápido, sin servidores)
    python3 benchmarks/prueba.py --live               # modo real: usa los servidores configurados en .env
    python3 benchmarks/prueba.py --only recuperacion  # una o varias suites (separadas por comas)
    python3 benchmarks/prueba.py --list               # lista las suites
    python3 benchmarks/prueba.py --json informe.json  # guarda todos los resultados
    python3 benchmarks/prueba.py --strict             # también falla con severidad 'medio'

Qué hace:
  * Trabaja SIEMPRE sobre carpetas temporales: nunca toca body/ ni tus memorias reales.
  * Cada comprobación tiene una META numérica, un valor medido y una severidad.
  * Al final emite una puntuación ponderada y la lista de PUNTOS DÉBILES con evidencia
    (preguntas concretas que fallan, cifras) y el sitio del código donde mirar.
  * Código de salida 0 = sin fallos críticos/altos; 1 = hay puntos débiles graves; 2 = no se pudo ejecutar.

Importante sobre el modo emulado:
  Valida la LÓGICA (pipeline, SQL, concurrencia, temporalidad, olvido, seguridad), pero un embedder
  simulado no entiende paráfrasis. Por eso, en ese modo, las métricas semánticas se muestran como
  observaciones y solo puntúan las que no dependen de un modelo real. La evidencia de calidad
  semántica se obtiene con --live.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import gzip
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import unicodedata
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LIVE_REQUESTED = "--live" in sys.argv
if not LIVE_REQUESTED:
    os.environ["HMEM_NO_EMB"] = "1"      # el embedder simulado se inyecta a mano
os.environ["HMEM_NO_BROWSER"] = "1"

try:
    import config as C
    import vectores
    from largo_plazo import MemoryTree, MemLLM
    from ltm.common import slugify
    from corto_plazo import ShortTermMemory
    from cartuchos import CartridgeManager
except Exception as _exc:  # pragma: no cover
    print(f"❌ No se pudo importar el proyecto desde {ROOT}: {type(_exc).__name__}: {_exc}")
    print("   Ejecuta este script desde la raíz del repositorio.")
    sys.exit(2)


# ==============================================================================
# 0. UTILIDADES
# ==============================================================================

def fold(text: Any) -> str:
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


from tests.helpers import MockEmbedder, MockLLM  # noqa: E402


# ==============================================================================
# 2. DATOS DEL BENCHMARK (36 hechos × 3 tipos de pregunta + distractores + negativas)
# ==============================================================================
# (dominio, hecho, palabra_clave, (literal, paráfrasis, indirecta))
FACTS: list[tuple[str, str, str, tuple[str, str, str]]] = [
    ("Fisiología", "La nefrona es la unidad funcional del riñón y filtra la sangre para formar la orina.", "nefrona",
     ("¿Cuál es la unidad funcional del riñón?",
      "¿Qué estructura microscópica depura la sangre y produce el líquido urinario?",
      "¿Qué componente se daña cuando el órgano que limpia la sangre pierde capacidad de filtrado?")),
    ("Fisiología", "La insulina, secretada por las células beta del páncreas, reduce la glucosa en sangre.", "insulina",
     ("¿Qué hormona secretan las células beta del páncreas?",
      "¿Qué sustancia hace bajar el azúcar circulante?",
      "¿Por qué una persona diabética necesita inyectarse una hormona pancreática?")),
    ("Fisiología", "El nodo sinusal del corazón actúa como marcapasos natural y genera el ritmo cardíaco.", "nodo sinusal",
     ("¿Qué estructura del corazón actúa como marcapasos natural?",
      "¿Dónde se origina el impulso eléctrico que marca el latido?",
      "¿Qué tejido sustituye un dispositivo implantado cuando el ritmo cardíaco falla?")),
    ("Fisiología", "Los alveolos pulmonares son los sacos donde se produce el intercambio de oxígeno y dióxido de carbono.", "alveolos",
     ("¿Dónde se produce el intercambio de oxígeno y dióxido de carbono en los pulmones?",
      "¿Qué diminutos sacos permiten respirar a nivel microscópico?",
      "¿Qué estructuras se destruyen en el enfisema y dificultan la oxigenación?")),
    ("Astronomía", "El telescopio espacial James Webb opera en el punto de Lagrange L2 y observa en infrarrojo.", "L2",
     ("¿En qué punto de Lagrange opera el telescopio espacial James Webb?",
      "¿Desde qué lugar del espacio observa el gran telescopio infrarrojo moderno?",
      "¿Dónde está situado el sucesor del Hubble?")),
    ("Astronomía", "Júpiter es el planeta más grande del sistema solar y tiene la Gran Mancha Roja, una tormenta gigante.", "Júpiter",
     ("¿Qué planeta tiene la Gran Mancha Roja?",
      "¿Cuál es el mayor mundo de nuestro sistema planetario?",
      "¿Dónde hay una tempestad persistente más grande que la Tierra?")),
    ("Astronomía", "Una estrella de neutrones es el núcleo colapsado de una estrella masiva y posee densidad nuclear extrema.", "neutrones",
     ("¿Qué es una estrella de neutrones?",
      "¿Qué queda tras el colapso del núcleo de un astro muy masivo sin llegar a agujero negro?",
      "¿Qué objeto celeste concentra más masa que el Sol en una esfera de pocos kilómetros?")),
    ("Astronomía", "La misión Apolo 11 llevó a Neil Armstrong y Buzz Aldrin a la superficie lunar el 20 de julio de 1969.", "Apolo 11",
     ("¿Quién viajó a la superficie lunar en la misión Apolo 11?",
      "¿Qué astronautas pisaron la Luna en 1969?",
      "¿Qué hito espacial ocurrió en el verano de 1969?")),
    ("Historia", "La Revolución Francesa comenzó en 1789 con la toma de la Bastilla.", "Bastilla",
     ("¿Cuándo comenzó la Revolución Francesa?",
      "¿Qué prisión parisina fue asaltada al estallar el proceso que acabó con el absolutismo galo?",
      "¿En qué año empezó el proceso que derrocó a la monarquía de los Borbones en Francia?")),
    ("Historia", "El Imperio romano de Occidente cayó en el año 476 cuando Rómulo Augústulo fue depuesto.", "476",
     ("¿Cuándo cayó el Imperio romano de Occidente?",
      "¿Qué emperador niño fue depuesto al terminar el dominio romano en Europa occidental?",
      "¿Qué suceso se considera el final de la Antigüedad clásica en Occidente?")),
    ("Historia", "La imprenta de tipos móviles fue desarrollada por Johannes Gutenberg hacia 1450 en Maguncia.", "Gutenberg",
     ("¿Quién desarrolló la imprenta de tipos móviles?",
      "¿Qué inventor alemán revolucionó la reproducción de libros en el siglo XV?",
      "¿A quién se debe la difusión masiva de la Biblia impresa?")),
    ("Historia", "El Muro de Berlín cayó el 9 de noviembre de 1989, simbolizando el fin de la Guerra Fría.", "Muro de Berlín",
     ("¿Cuándo cayó el Muro de Berlín?",
      "¿Qué frontera urbana que separaba dos bloques políticos dejó de existir en noviembre?",
      "¿Qué hecho se asocia con el desenlace del conflicto entre EEUU y la URSS?")),
    ("Informática", "El protocolo TCP garantiza la entrega ordenada y fiable de paquetes mediante confirmaciones y retransmisiones.", "TCP",
     ("¿Qué protocolo garantiza la entrega ordenada y fiable de paquetes?",
      "¿Cómo se asegura que los datos de red no se pierdan ni lleguen desordenados?",
      "¿Qué mecanismo reenvía los segmentos que no fueron confirmados?")),
    ("Informática", "Un índice B-Tree en una base de datos acelera las búsquedas evitando recorrer toda la tabla.", "B-Tree",
     ("¿Qué hace un índice B-Tree en una base de datos?",
      "¿Qué estructura permite encontrar filas sin escanear la tabla completa?",
      "¿Por qué una consulta lenta mejora al añadir cierta estructura de ordenación en disco?")),
    ("Informática", "Git es un sistema de control de versiones distribuido creado por Linus Torvalds en 2005.", "Torvalds",
     ("¿Quién creó Git?",
      "¿Qué herramienta distribuida para versionar código desarrolló el padre de Linux?",
      "¿Qué software usan los programadores para gestionar ramas y commits?")),
    ("Informática", "HTTPS cifra la comunicación entre navegador y servidor mediante TLS.", "TLS",
     ("¿Qué protocolo cifra HTTPS entre navegador y servidor?",
      "¿Qué capa criptográfica protege las conexiones web seguras?",
      "¿Por qué el candado del navegador indica que nadie intercepta la contraseña?")),
    ("Biología", "La fotosíntesis ocurre en los cloroplastos y convierte luz, agua y dióxido de carbono en glucosa y oxígeno.", "cloroplastos",
     ("¿Dónde ocurre la fotosíntesis?",
      "¿Qué orgánulo vegetal transforma la luz solar en azúcar?",
      "¿Por qué las plantas liberan oxígeno al ambiente?")),
    ("Biología", "El ADN almacena la información genética en una doble hélice formada por cuatro bases: adenina, timina, citosina y guanina.", "doble hélice",
     ("¿Qué forma tiene la molécula de ADN?",
      "¿Qué cuatro letras químicas componen el código genético?",
      "¿Dónde se guardan las instrucciones hereditarias de una célula?")),
    ("Biología", "Las mitocondrias generan la mayor parte del ATP celular mediante la respiración aeróbica.", "mitocondrias",
     ("¿Qué orgánulos generan el ATP celular?",
      "¿Qué estructura es la central energética de la célula?",
      "¿Qué falla en las enfermedades que provocan fatiga celular por falta de energía?")),
    ("Biología", "Los linfocitos T coordinan la respuesta inmune adaptativa y destruyen células infectadas.", "linfocitos T",
     ("¿Qué células coordinan la respuesta inmune adaptativa?",
      "¿Qué glóbulos blancos eliminan las células infectadas por virus?",
      "¿Qué población celular ataca el VIH y deja al cuerpo sin defensas?")),
    ("Geografía", "El río Amazonas es el más caudaloso del mundo y atraviesa Brasil, Perú y Colombia.", "Amazonas",
     ("¿Cuál es el río más caudaloso del mundo?",
      "¿Qué corriente fluvial sudamericana lleva más agua al océano?",
      "¿Qué curso de agua riega la mayor selva tropical?")),
    ("Geografía", "El monte Everest mide 8.849 metros y es la montaña más alta sobre el nivel del mar.", "Everest",
     ("¿Cuál es la montaña más alta sobre el nivel del mar?",
      "¿Qué cima del Himalaya supera los ocho mil ochocientos metros?",
      "¿Hasta dónde debe ascender un alpinista que busca el techo del planeta?")),
    ("Geografía", "El desierto del Sahara ocupa gran parte del norte de África y es el mayor desierto cálido del planeta.", "Sahara",
     ("¿Cuál es el mayor desierto cálido del planeta?",
      "¿Qué extensión árida cubre buena parte del norte africano?",
      "¿Dónde se encuentran dunas que se extienden por varios países del Magreb y el Sahel?")),
    ("Geografía", "El lago Titicaca está entre Perú y Bolivia y es el lago navegable más alto del mundo.", "Titicaca",
     ("¿Entre qué países se encuentra el lago Titicaca?",
      "¿Qué masa de agua andina es la navegable situada a mayor elevación?",
      "¿Qué lago comparten peruanos y bolivianos a más de tres mil metros?")),
    ("Física", "El principio de incertidumbre de Heisenberg limita la precisión con que se pueden conocer a la vez posición y momento de una partícula.", "Heisenberg",
     ("¿Qué limita el principio de incertidumbre de Heisenberg?",
      "¿Por qué no se puede medir al mismo tiempo dónde está un electrón y cuánto se mueve?",
      "¿Qué ley cuántica impide un conocimiento perfecto simultáneo de ciertas magnitudes?")),
    ("Física", "La velocidad de la luz en el vacío es de aproximadamente 299.792 kilómetros por segundo.", "299.792",
     ("¿Cuál es la velocidad de la luz en el vacío?",
      "¿A qué rapidez viaja un fotón sin medio material?",
      "¿Qué límite universal de velocidad establece la relatividad?")),
    ("Física", "La segunda ley de la termodinámica establece que la entropía de un sistema aislado nunca disminuye.", "entropía",
     ("¿Qué establece la segunda ley de la termodinámica sobre la entropía?",
      "¿Por qué el desorden de un sistema cerrado tiende a crecer?",
      "¿Qué principio explica que el calor fluya espontáneamente del cuerpo caliente al frío?")),
    ("Física", "Un semiconductor como el silicio conduce la electricidad de forma intermedia y es la base de los transistores.", "silicio",
     ("¿Qué material semiconductor es la base de los transistores?",
      "¿Qué elemento químico permite fabricar chips electrónicos?",
      "¿Por qué los microprocesadores se fabrican sobre obleas de un cierto elemento del grupo 14?")),
    ("Datos personales", "La hija mayor del usuario se llama Alicia y nació en Valladolid en 2012.", "Alicia",
     ("¿Cómo se llama la hija mayor del usuario?",
      "¿Cuál es el nombre de la primogénita del usuario?",
      "¿Dónde vino al mundo la niña mayor del usuario?")),
    ("Datos personales", "El usuario es alérgico a la penicilina y a la amoxicilina.", "penicilina",
     ("¿A qué antibióticos es alérgico el usuario?",
      "¿Qué medicamentos no puede tomar el usuario por reacción alérgica?",
      "¿Qué receta debe evitar el médico al tratar una infección del usuario?")),
    ("Datos personales", "El perro del usuario es un labrador llamado Thor que tiene cuatro años.", "Thor",
     ("¿Cómo se llama el perro del usuario?",
      "¿Qué mascota tiene el usuario y de qué raza es?",
      "¿Con qué animal convive el usuario?")),
    ("Datos personales", "El usuario prefiere recibir las respuestas en español y con ejemplos de código en Python.", "Python",
     ("¿En qué idioma prefiere el usuario las respuestas?",
      "¿Qué lenguaje de programación quiere el usuario en los ejemplos?",
      "¿Cómo debe redactar el asistente sus contestaciones para el usuario?")),
    ("Salud", "La hipertensión arterial se diagnostica cuando la presión supera 140/90 mmHg de forma sostenida.", "140/90",
     ("¿Cuándo se diagnostica la hipertensión arterial?",
      "¿A partir de qué cifras de tensión se considera alta de forma persistente?",
      "¿Qué valores de presión obligan a iniciar tratamiento antihipertensivo?")),
    ("Salud", "La metformina es el fármaco de primera línea para la diabetes mellitus tipo 2.", "metformina",
     ("¿Cuál es el fármaco de primera línea para la diabetes tipo 2?",
      "¿Qué medicamento oral se receta primero a un paciente con azúcar alta crónica del adulto?",
      "¿Qué tratamiento inicial reduce la resistencia a la insulina?")),
    ("Salud", "La vacuna contra el sarampión forma parte de la triple vírica y se administra en dos dosis.", "triple vírica",
     ("¿Qué vacuna incluye protección frente al sarampión?",
      "¿Cuántas inyecciones requiere la inmunización contra esa enfermedad exantemática infantil?",
      "¿Qué prevención se recomienda a los niños para evitar manchas rojas y fiebre alta contagiosas?")),
    ("Salud", "El ibuprofeno es un antiinflamatorio no esteroideo que puede dañar el estómago si se usa de forma prolongada.", "ibuprofeno",
     ("¿Qué antiinflamatorio no esteroideo puede dañar el estómago?",
      "¿Qué analgésico común irrita la mucosa gástrica con uso prolongado?",
      "¿Qué riesgo digestivo tiene tomar a diario ciertas pastillas para el dolor?")),
]


def _fillers() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for n, h in [("Aconcagua", "6.961"), ("Kilimanjaro", "5.895"), ("Mont Blanc", "4.806"), ("Elbrús", "5.642"),
                 ("Denali", "6.190"), ("Teide", "3.715"), ("Mulhacén", "3.479"), ("Fuji", "3.776")]:
        out.append(("Geografía", f"El monte {n} alcanza {h} metros de altitud."))
    for n, p in [("Danubio", "la Selva Negra"), ("Nilo", "el lago Victoria"), ("Tajo", "la Sierra de Albarracín"),
                 ("Ebro", "Fontibre"), ("Rin", "los Alpes suizos"), ("Volga", "las colinas de Valdái")]:
        out.append(("Geografía", f"El río {n} nace en {p}."))
    for e, s, z in [("hierro", "Fe", 26), ("cobre", "Cu", 29), ("oro", "Au", 79), ("plata", "Ag", 47),
                    ("estaño", "Sn", 50), ("plomo", "Pb", 82), ("zinc", "Zn", 30), ("níquel", "Ni", 28)]:
        out.append(("Química", f"El {e} tiene símbolo {s} y número atómico {z}."))
    for a, o in [("Cervantes", "Don Quijote de la Mancha"), ("Tolstói", "Guerra y paz"),
                 ("García Márquez", "Cien años de soledad"), ("Dante", "la Divina Comedia"),
                 ("Homero", "la Odisea"), ("Borges", "El Aleph")]:
        out.append(("Literatura", f"{a} escribió {o}."))
    return out


FILLERS = _fillers()

# Preguntas SIN respuesta en el corpus: lo correcto es abstenerse (found=False).
NEGATIVES = [
    # Conocimiento general externo no indexado
    "¿Cuál es la capital de Mongolia?",
    "¿Quién ganó el Mundial de fútbol de 1966?",
    "¿Cómo se prepara una tortilla de patatas?",
    "¿Cuántas patas tiene una araña?",
    "¿Qué grupo compuso Bohemian Rhapsody?",
    "¿Cuál es la capital de Australia?",
    "¿Quién escribió Hamlet?",
    "¿Cuántos huesos tiene el cuerpo humano adulto?",
    "¿Cómo se instala Docker en Windows?",
    "¿A qué temperatura hierve el mercurio?",
    "¿Quién pintó Las Meninas?",
    "¿En qué año se descubrió la penicilina?",
    "¿Cuál es la velocidad del sonido en el aire?",
    "¿Quién formuló la teoría de la relatividad general?",
    "¿Qué instrumento tocaba Miles Davis?",
    "¿Cuál es la moneda oficial de Japón?",
    "¿Quién descubrió la estructura del átomo?",
    "¿Cuál es el océano más grande de la Tierra?",
    "¿Qué elemento químico tiene número atómico 1?",
    "¿Dónde se celebraron los Juegos Olímpicos de 1992?",

    # Atributos ficticios del usuario (trampas por semántica cercana de sujeto)
    "¿Cuál es la contraseña del wifi del usuario?",
    "¿Cómo se llama el gato del usuario?",
    "¿Qué coche conduce el usuario?",
    "¿Cuál es el nombre de la esposa del usuario?",
    "¿Qué profesión tiene el padre del usuario?",
    "¿Cuál es el grupo sanguíneo del usuario?",
    "¿Qué día es el cumpleaños del usuario?",
    "¿Cuál es el color favorito del usuario?",
    "¿Qué idioma habla el usuario con su familia?",
    "¿Qué deporte practica el usuario habitualmente?",
    "¿Cuál es la comida preferida del usuario?",
    "¿En qué universidad estudió el usuario la carrera?",
    "¿Cuál es el número de teléfono del usuario?",
    "¿Cuánto mide de altura el usuario?",
    "¿Qué marca de ordenador utiliza el usuario?",
    "¿Tiene el usuario alguna fobia conocida?",
    "¿Cuál es la dirección postal exacta del usuario?",
    "¿A qué país viajó de vacaciones el usuario el año pasado?",
    "¿Tiene el usuario hermanos o hermanas?",
    "¿Cuál es el segundo apellido del usuario?",
]

# (atributo, hecho viejo, hecho nuevo, pregunta actual, pregunta histórica)
TEMPORAL = [
    ("vivienda", "El usuario vive actualmente en Valencia, en la calle Colón.",
     "El usuario se ha mudado a Tokio y reside allí desde este mes.",
     "¿Dónde vive el usuario ahora?", "¿Dónde vivía el usuario antes?"),
    ("empresa", "El usuario trabaja en el Banco Santander como analista.",
     "El usuario ahora trabaja en Telefónica como ingeniero de software.",
     "¿Dónde trabaja el usuario?", "¿Dónde trabajaba el usuario antes?"),
    ("estudios", "El usuario estudia en la Universidad de León.",
     "El usuario ya no estudia en la Universidad de León; ahora cursa un máster en la Complutense.",
     "¿Dónde estudia el usuario?", "¿Dónde estudiaba el usuario antes?"),
    ("profesion", "El usuario es camarero en un restaurante de Burgos.",
     "El usuario tiene un nuevo trabajo: es profesor de matemáticas en un instituto.",
     "¿A qué se dedica el usuario?", "¿A qué se dedicaba el usuario antes?"),
    ("bebida", "El usuario prefiere el café por las mañanas.",
     "A partir de ahora el usuario prefiere el té por las mañanas.",
     "¿Qué bebida prefiere el usuario por las mañanas?", "¿Qué bebida prefería el usuario antes por las mañanas?"),
    ("ejercicio", "El usuario trabaja como médico en el hospital de Zamora.",
     "El usuario se ha jubilado y ya no ejerce como médico.",
     "¿Sigue ejerciendo como médico el usuario?", "¿Dónde trabajaba el usuario como médico antes?"),
]
TEMPORAL_DISTRACTORS = [
    "El usuario tiene dos hermanos.", "El usuario es alérgico a los frutos secos.",
    "La hija del usuario estudia piano.", "El usuario conduce un coche eléctrico.",
    "El usuario practica senderismo los domingos.",
]
# Controles: NO deben invalidar al hecho previo (atributos distintos / complementarios).
TEMPORAL_CONTROLS = [
    ("El usuario tiene un perro llamado Thor.", "El usuario tiene un gato llamado Misu.", "mascota-a", "mascota-b"),
    ("El usuario vive en Madrid.", "El usuario trabaja en Madrid actualmente.", "vivienda-c", "trabajo-c"),
]

NONFACTUAL_INPUTS = [
    "¿Cuál es la capital de Francia?", "Dime qué tiempo hará mañana", "Hola, buenos días", "Gracias",
    "Resúmeme este artículo por favor", "Explícame cómo funciona TCP", "¿Sabes cómo se llama mi perro?",
    "Traduce esta frase al inglés", "ok", "Necesito que me traduzcas un texto al francés",
    "Podrías explicarme qué es una nefrona", "Quiero saber cuál es la capital de Perú",
]
FACTUAL_INPUTS = [
    "Mi hija mayor se llama Alicia y nació en Valladolid.", "Soy alérgico a la penicilina.",
    "Trabajo como ingeniero en una empresa de Zamora.", "Me he mudado a Tokio el mes pasado.",
    "Mi perro se llama Thor y tiene cuatro años.", "Prefiero que me respondas en español.",
    "Tengo una reunión importante el 3 de noviembre.", "Mi cumpleaños es el 14 de abril.",
]


# ==============================================================================
# 3. INFORME
# ==============================================================================
SEV_ORDER = {"crítico": 0, "alto": 1, "medio": 2, "info": 3}
SEV_WEIGHT = {"crítico": 3.0, "alto": 2.0, "medio": 1.0, "info": 0.0}


@dataclass
class Check:
    suite: str
    name: str
    ok: bool
    value: str
    target: str
    severity: str
    where: str = ""
    hint: str = ""
    evidence: list[str] = field(default_factory=list)


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.suite_time: dict[str, float] = {}
        self.suite_skipped: dict[str, str] = {}

    def add(self, suite: str, name: str, ok: Any, value: str, target: str, severity: str = "medio",
            where: str = "", hint: str = "", evidence: list[str] | None = None) -> None:
        self.checks.append(Check(suite, name, bool(ok), value, target, severity, where, hint, list(evidence or [])))

    def print_suite(self, suite: str) -> None:
        for c in self.checks:
            if c.suite != suite:
                continue
            mark = "✓" if c.ok else ("•" if c.severity == "info" else "✗")
            print(f"   {mark} {c.name:<54} {c.value:>14}   (meta {c.target})")
            if not c.ok:
                for e in c.evidence[:4]:
                    print(f"        ↳ {e}")

    def score(self) -> float:
        tot = sum(SEV_WEIGHT[c.severity] for c in self.checks)
        got = sum(SEV_WEIGHT[c.severity] for c in self.checks if c.ok)
        return 10.0 * got / tot if tot else 0.0

    def failures(self, strict: bool = False) -> list[Check]:
        lim = 2 if strict else 1
        return [c for c in self.checks if not c.ok and SEV_ORDER[c.severity] <= lim]


# ==============================================================================
# 4. CONTEXTO
# ==============================================================================

class Corpus:
    def __init__(self, tree: Any, id_of: dict[int, str], qs: list[tuple[str, int, str, str]]) -> None:
        self.tree, self.id_of, self.qs = tree, id_of, qs


class Ctx:
    def __init__(self, args: argparse.Namespace, report: Report) -> None:
        self.args = args
        self.live = bool(args.live)
        self.mock = not self.live
        self.report = report
        self.tmp = Path(tempfile.mkdtemp(prefix="hmem_prueba_"))
        self.llm: Any = MockLLM() if self.mock else MemLLM()
        self.trees: list[Any] = []
        self._corpus: Corpus | None = None
        self.slept = False

    def thr(self, live: Any, mock: Any) -> Any:
        return live if self.live else mock

    def register(self, text: str, domain: str, attr: str | None = None) -> None:
        if self.mock:
            self.llm.register(text, domain, attr)

    def new_tree(self, name: str) -> Any:
        tree = MemoryTree(self.tmp / name, self.llm)
        if self.mock:
            tree.emb = vectores.EmbeddingService(tree, embedder=MockEmbedder())
        self.trees.append(tree)
        return tree

    def corpus(self) -> Corpus:
        if self._corpus is not None:
            return self._corpus
        tree = self.new_tree("corpus")
        id_of: dict[int, str] = {}
        qs: list[tuple[str, int, str, str]] = []
        print("   … cargando corpus (36 hechos + 28 distractores)", end="", flush=True)
        for fi, (dom, text, _clave, (lit, par, ind)) in enumerate(FACTS):
            self.register(text, dom)
            r = tree.store(text, 7, "dato", meta={"fuente": "benchmark"})
            id_of[fi] = r["id"]
            qs += [(lit, fi, "literal", dom), (par, fi, "paráfrasis", dom), (ind, fi, "indirecta", dom)]
            print(".", end="", flush=True)
        for dom, text in FILLERS:
            self.register(text, dom)
            tree.store(text, 6, "dato", meta={"fuente": "benchmark"})
        sync_all(tree)
        print(" listo")
        self._corpus = Corpus(tree, id_of, qs)
        return self._corpus

    def cleanup(self) -> None:
        for t in self.trees:
            try:
                t.close()
            except Exception:
                pass
        if not self.args.keep:
            shutil.rmtree(self.tmp, ignore_errors=True)
        else:
            print(f"\n(archivos temporales conservados en {self.tmp})")


def sync_all(tree: Any) -> None:
    emb = getattr(tree, "emb", None)
    if emb is None:
        return
    for _ in range(300):
        r = emb.sync(limit=64)
        if "al día" in r or "no disponible" in r or "no respondió" in r:
            break


def dbq(tree: Any, sql: str, params: tuple = ()) -> list[Any]:
    with tree.lock:
        return tree.conn.execute(sql, params).fetchall()


def cell_texts(tree: Any, ids: list[str]) -> dict[str, str]:
    ids = [i for i in ids if i]
    if not ids:
        return {}
    rows = dbq(tree, f"SELECT id, text FROM cells WHERE id IN ({','.join('?' * len(ids))})", tuple(ids))
    return {r["id"]: r["text"] for r in rows}


# ==============================================================================
# 5. SUITES
# ==============================================================================

# ---------------------------------------------------------------- ablación
_ONLY_EMB = {"RRF_W_EXACT": 0, "RRF_W_FTS": 0, "RRF_W_SEM": 0, "RRF_W_TREE": 0, "RRF_W_SYN": 0,
             "SYN_ENABLED": False, "MEM_RECALL_EXPAND": False}
# [PARCHE2] Pesos por defecto de config.py con valores EXPLÍCITOS: la ablación no depende de
# RETRIEVAL_PROFILE ni de .env (antes C.RRF_W_* valían 0 con el perfil 'embeddings' y todas las filas eran iguales).
_DEF = {"RRF_W_EXACT": 0.6, "RRF_W_FTS": 0.25, "RRF_W_SEM": 0.3, "RRF_W_TREE": 0.1, "RRF_W_SYN": 0.3, "RRF_W_EMB": 2.5}
_FULL = {**_DEF, "SYN_ENABLED": True, "MEM_RECALL_EXPAND": True}
_EMB = {**_ONLY_EMB, "RRF_W_EMB": _DEF["RRF_W_EMB"]}
# (nombre, overrides de config, navegar por el árbol, configuración base con la que se compara)
ABLATIONS: list[tuple[str, dict[str, Any], bool, str]] = [
    ("actual (.env)", {}, True, ""),
    ("completo", _FULL, True, "actual (.env)"),
    ("solo embeddings", _EMB, False, "completo"),
    ("emb + léxico", {**_EMB, "RRF_W_FTS": _DEF["RRF_W_FTS"]}, False, "solo embeddings"),
    ("emb + exacta + léxico", {**_EMB, "RRF_W_FTS": _DEF["RRF_W_FTS"], "RRF_W_EXACT": _DEF["RRF_W_EXACT"]},
     False, "solo embeddings"),
    ("completo − árbol", {**_FULL, "RRF_W_TREE": 0}, False, "completo"),
    ("completo − sinapsis", {**_FULL, "RRF_W_SYN": 0, "SYN_ENABLED": False}, True, "completo"),
    ("completo − léxico", {**_FULL, "RRF_W_FTS": 0}, True, "completo"),
    ("completo − exacta", {**_FULL, "RRF_W_EXACT": 0}, True, "completo"),
    ("completo − expansión LLM", {**_FULL, "MEM_RECALL_EXPAND": False}, True, "completo"),
]


@contextlib.contextmanager
def _cfg_override(**over: Any):
    old = {k: getattr(C, k) for k in over}
    try:
        for k, v in over.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _sign_p(b: int, c: int) -> float:
    """Test de signos exacto bilateral sobre los pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def suite_ablacion(ctx: Ctx) -> None:
    R, S = ctx.report, "ablacion"
    corp = ctx.corpus()
    tree, qs, id_of = corp.tree, corp.qs, corp.id_of
    test_idx = [i for i, (_q, fi, _t, _d) in enumerate(qs) if fi % 3 == 0]   # 1 de cada 3 hechos
    test_set = set(test_idx)
    dev_idx = [i for i in range(len(qs)) if i not in test_set]

    res: dict[str, list[dict[str, Any]]] = {}
    for name, over, nav, _base in ABLATIONS:   # [PARCHE2]
        with _cfg_override(**over):
            res[name] = _evaluate(tree, qs, id_of, navigate=nav)

    ok1 = lambda r: r["rank"] == 1
    ok3 = lambda r: bool(r["rank"] and r["rank"] <= 3)
    n = len(qs)
    print(f"   perfil={C.RETRIEVAL_PROFILE} | 'completo' = pesos por defecto EXPLÍCITOS: "
          + " ".join(f"{k[6:]}={v}" for k, v in _DEF.items()))
    print(f"   {'configuración':<26}{'Hit@1':>7}{'Hit@3':>7}{'IC95% Hit@3':>17}{'dev':>7}{'test':>7}"
          f"{'MRR':>7}{'P95ms':>7}   vs base (top-1)")
    for name, _over, _nav, bname in ABLATIONS:
        rows = res[name]
        k1, k3 = sum(map(ok1, rows)), sum(map(ok3, rows))
        lo, hi = _wilson(k3, n)
        ic = f"[{pct(lo)}–{pct(hi)}]"
        dv = sum(ok3(rows[i]) for i in dev_idx) / max(1, len(dev_idx))
        ts = sum(ok3(rows[i]) for i in test_idx) / max(1, len(test_idx))
        if not bname:
            cmp = "—"
        else:
            base = res[bname]
            gana = sum(ok1(y) and not ok1(x) for x, y in zip(base, rows))
            pierde = sum(ok1(x) and not ok1(y) for x, y in zip(base, rows))
            cmp = f"{bname[:15]}: +{gana}/−{pierde} p={_sign_p(pierde, gana):.2f}"
        print(f"   {name:<26}{pct(k1 / n):>7}{pct(k3 / n):>7}{ic:>17}{pct(dv):>7}{pct(ts):>7}"
              f"{_mrr(rows):>7.3f}{percentile([r['ms'] for r in rows], 95):>7.0f}   {cmp}")
        R.add(S, f"Hit@3 · {name}", True, pct(k3 / n), "(observación)", "info", "config.py · RRF_W_*")
        R.add(S, f"Hit@1/MRR · {name}", True, f"{pct(k1 / n)} / {_mrr(rows):.3f}", "(observación)", "info",
              "config.py · RRF_W_*")
    print(f"   n={n} preguntas · dev={len(dev_idx)} · test={len(test_idx)}")
    print("   (+g/−p: preguntas que la variante acierta en top-1 y su base no / al revés; p = test de signos exacto)")
    act, full = res["actual (.env)"], res["completo"]
    diff = [i for i in range(n) if ok1(act[i]) != ok1(full[i])]
    print(f"   · Discordancias top-1 entre 'actual (.env)' y 'completo': {len(diff)}")
    for i in diff[:10]:
        print(f"     [{qs[i][2][:4]}] «{qs[i][0][:66]}» actual rank={act[i]['rank']} · completo rank={full[i]['rank']}")
    if not ctx.live:
        print("   ⚠ modo emulado: valida la fontanería, la comparación solo es informativa con --live")


# ---------------------------------------------------------------- diagnóstico
# [PARCHE2] _DEF se define en la sección de ablación (antes de ABLATIONS)
CANDIDATES: list[tuple[str, dict[str, Any], bool]] = [
    ("actual (.env)", {}, True),
    ("solo embeddings", _ONLY_EMB, False),
    ("defaults config.py", _DEF, True),
    ("defaults sin expansión", {**_DEF, "MEM_RECALL_EXPAND": False}, True),
    ("emb2.5+exacta+léx.25 (ligero)", {**_ONLY_EMB, "RRF_W_EMB": 2.5, "RRF_W_EXACT": 0.6, "RRF_W_FTS": 0.25}, False),
]


def _overlap_terms(tree: Any, q: str, text: str) -> int:
    qs = {tree._qstem(t) for t in tree._fts_terms(q)}
    cs = {tree._qstem(t) for t in tree._fts_terms(text)}
    return len(qs & cs)


def _top1_emb(tree: Any, q: str) -> tuple[float | None, str | None]:
    qv = tree.emb.embed_query(q) if tree.emb is not None else None
    if qv is None:
        return None, None
    h = tree.emb.store.search("cell", qv, 1, -1.0, tree.emb.model)
    return (h[0][1], h[0][0]) if h else (None, None)


def suite_diagnostico(ctx: Ctx) -> None:
    R, S = ctx.report, "diagnostico"
    corp = ctx.corpus()
    tree, qs, id_of = corp.tree, corp.qs, corp.id_of
    ok3 = lambda r: bool(r["rank"] and r["rank"] <= 3)
    n = len(qs)
    print(f"   perfil={C.RETRIEVAL_PROFILE} | EXACT={C.RRF_W_EXACT} FTS={C.RRF_W_FTS} SEM={C.RRF_W_SEM} "
          f"TREE={C.RRF_W_TREE} SYN={C.RRF_W_SYN} EMB={C.RRF_W_EMB} EXPAND={C.MEM_RECALL_EXPAND}")
    if C.RETRIEVAL_PROFILE == "embeddings":
        print("   ⚠ perfil 'embeddings' activo: 'actual (.env)' NO es tu .env. Usa RETRIEVAL_PROFILE=completo")

    # ---- 1) candidatos de configuración
    res: dict[str, list[dict[str, Any]]] = {}
    for name, over, nav in CANDIDATES:
        with _cfg_override(**over):
            res[name] = _evaluate(tree, qs, id_of, navigate=nav)
    base = res["actual (.env)"]
    print(f"\n   1) CANDIDATOS DE CONFIGURACIÓN  (n={n})")
    print(f"   {'configuración':<32}{'Hit@3':>7}{'MRR':>7}{'P95ms':>7}   vs actual")
    for name, rows in res.items():
        k = sum(map(ok3, rows))
        if rows is base:
            cmp = "—"
        else:
            gana = sum(ok3(y) and not ok3(x) for x, y in zip(base, rows))
            pierde = sum(ok3(x) and not ok3(y) for x, y in zip(base, rows))
            cmp = f"+{gana}/−{pierde}  p={_sign_p(pierde, gana):.2f}"
        print(f"   {name:<32}{pct(k / n):>7}{_mrr(rows):>7.3f}{percentile([r['ms'] for r in rows], 95):>7.0f}   {cmp}")
        R.add(S, f"Hit@3 · {name}", True, pct(k / n), "(observación)", "info")

    # ---- 2) preguntas discordantes
    emb_rows = res["solo embeddings"]
    print("\n   2) PREGUNTAS DISCORDANTES  ('actual' vs 'solo embeddings')")
    for titulo, a, b, over_b, nav_b in (
            ("acierta 'solo embeddings' y falla 'actual'", emb_rows, base, {}, True),
            ("acierta 'actual' y falla 'solo embeddings'", base, emb_rows, _ONLY_EMB, False)):
        idx = [i for i in range(n) if ok3(a[i]) and not ok3(b[i])]
        print(f"   · {titulo}: {len(idx)}")
        for i in idx[:6]:
            q, fi, tipo, _d = qs[i]
            gold = id_of[fi]
            with _cfg_override(**over_b):
                r = tree.recall(q, reinforce=False, navigate=nav_b, synthesize=False)
            cands = (r.get("candidates") or [])[:3]
            tt = cell_texts(tree, cands + [gold])
            print(f"     «{q[:75]}» [{tipo}] rank={b[i]['rank']}")
            print(f"        esperado: {tt.get(gold, '')[:75]}")
            for c in cands:
                print(f"        top-3:    {tt.get(c, '?')[:75]}")

    # ---- 3) abstención
    print("\n   3) ABSTENCIÓN  (top-1 por embeddings; solap = términos de contenido compartidos con el top-1)")
    neg = []
    for q in NEGATIVES:
        s, cid = _top1_emb(tree, q)
        txt = cell_texts(tree, [cid]).get(cid, "") if cid else ""
        neg.append((q, s if s is not None else 0.0, _overlap_terms(tree, q, txt), txt))
    pos = []
    for q, fi, tipo, _d in qs:
        s, cid = _top1_emb(tree, q)
        if s is None:
            continue
        txt = cell_texts(tree, [cid]).get(cid, "") if cid else ""
        pos.append((q, tipo, s, _overlap_terms(tree, q, txt)))

    print("   · Negativas con mayor similitud (las que habría que rechazar):")
    for q, s, ov, txt in sorted(neg, key=lambda x: -x[1])[:10]:
        print(f"     sim={s:.2f} solap={ov}  «{q[:46]}» → «{txt[:46]}»")
    print("   · Positivas con menor similitud (las que NO hay que perder):")
    for q, tipo, s, ov in sorted(pos, key=lambda x: x[2])[:10]:
        print(f"     sim={s:.2f} solap={ov} [{tipo[:4]}]  «{q[:70]}»")

    lp = [p for p in pos if p[1] in ("literal", "paráfrasis")]
    print("\n   · Reglas de abstención: conservan (lit+par) / rechazan (negativas)")
    print(f"     {'umbral':>6} | {'solo sim':>16} | {'sim o solap≥1':>16} | {'sim o solap≥2':>16}")
    for k in range(40, 71, 5):
        th = k / 100
        cols = []
        for need in (None, 1, 2):
            keep = sum(1 for p in lp if p[2] >= th or (need is not None and p[3] >= need)) / max(1, len(lp))
            rej = sum(1 for p in neg if not (p[1] >= th or (need is not None and p[2] >= need))) / max(1, len(neg))
            cols.append(f"{pct(keep):>7} / {pct(rej):>6}")
        print(f"     {th:>6.2f} | {cols[0]} | {cols[1]} | {cols[2]}")
    print("     (criterio: conservar ≥95% de lit+par y rechazar el máximo de negativas)")


# ---------------------------------------------------------------- identificadores
_LEX_ONLY = {**_ONLY_EMB, "RRF_W_EMB": 0, "RRF_W_FTS": 1.0, "RRF_W_EXACT": 0.6}
ID_CONFIGS: list[tuple[str, dict[str, Any], bool]] = [
    ("actual (.env)", {}, True),
    ("solo embeddings", _ONLY_EMB, False),
    ("ligero (emb2.5+exacta+léx.25)", {**_ONLY_EMB, "RRF_W_EMB": 2.5, "RRF_W_EXACT": 0.6, "RRF_W_FTS": 0.25}, False),
    ("defaults sin expansión", {**_DEF, "MEM_RECALL_EXPAND": False}, True),
    ("solo léxico+exacta", _LEX_ONLY, False),
]


def suite_identificadores(ctx: Ctx) -> None:
    R, S = ctx.report, "identificadores"
    corp = ctx.corpus()
    tree, id_of = corp.tree, corp.id_of
    ok3 = lambda r: bool(r["rank"] and r["rank"] <= 3)
    qs2: list[tuple[str, int, str, str]] = []
    for fi, (dom, _texto, clave, _qs) in enumerate(FACTS):
        qs2.append((clave, fi, "palabra", dom))
        qs2.append((f"¿Qué sabes de {clave}?", fi, "pregunta", dom))
    print(f"   perfil={C.RETRIEVAL_PROFILE} | EXACT={C.RRF_W_EXACT} FTS={C.RRF_W_FTS} EMB={C.RRF_W_EMB} "
          f"TREE={C.RRF_W_TREE} SYN={C.RRF_W_SYN} EXPAND={C.MEM_RECALL_EXPAND}")
    if C.RETRIEVAL_PROFILE == "embeddings":
        print("   ⚠ perfil 'embeddings' activo: 'actual (.env)' NO es tu .env. Usa RETRIEVAL_PROFILE=completo")

    res: dict[str, list[dict[str, Any]]] = {}
    for name, over, nav in ID_CONFIGS:
        with _cfg_override(**over):
            res[name] = _evaluate(tree, qs2, id_of, navigate=nav)

    def hit3(rows: list[dict[str, Any]], tipo: str) -> float:
        sel = [r for r in rows if r["tipo"] == tipo]
        return sum(ok3(r) for r in sel) / max(1, len(sel))

    n_pal = sum(1 for q in qs2 if q[2] == "palabra")
    print(f"\n   CONSULTAS DE PALABRA CLAVE  ({n_pal} palabras + {len(qs2) - n_pal} preguntas cortas)")
    print(f"   {'configuración':<34}{'palabra':>9}{'pregunta':>10}{'Hit@1':>8}{'P95ms':>7}")
    for name, rows in res.items():
        h1 = sum(1 for r in rows if r["rank"] == 1) / len(rows)
        print(f"   {name:<34}{pct(hit3(rows, 'palabra')):>9}{pct(hit3(rows, 'pregunta')):>10}{pct(h1):>8}"
              f"{percentile([r['ms'] for r in rows], 95):>7.0f}")
        R.add(S, f"Hit@3 palabra · {name}", True, pct(hit3(rows, "palabra")), "(observación)", "info")

    solo = res["solo embeddings"]
    miss = [i for i, r in enumerate(solo) if r["tipo"] == "palabra" and not ok3(r)]
    print(f"\n   Palabras que 'solo embeddings' NO encuentra en el top-3: {len(miss)}")
    for i in miss:
        rescata = [nm for nm, rows in res.items() if nm != "solo embeddings" and ok3(rows[i])]
        print(f"     «{qs2[i][0]}» (rank={solo[i]['rank']}) → la rescatan: {', '.join(rescata) or 'ninguna'}")
    lig = res["ligero (emb2.5+exacta+léx.25)"]
    pierde = [i for i, r in enumerate(solo) if ok3(r) and not ok3(lig[i])]
    print(f"\n   Consultas que 'solo embeddings' acierta y 'ligero' pierde: {len(pierde)}")
    for i in pierde[:8]:
        print(f"     «{qs2[i][0]}» [{qs2[i][2]}] rank_ligero={lig[i]['rank']}")


# ---------------------------------------------------------------- abstención
ABSTAIN_SWEEP = [0.46, 0.50, 0.54, 0.58, 0.62, 0.66]


def suite_abstencion(ctx: Ctx) -> None:
    R, S = ctx.report, "abstencion"
    corp = ctx.corpus()
    tree, qs = corp.tree, corp.qs
    n_gen = 20   # NEGATIVES[:20] = conocimiento general; NEGATIVES[20:] = atributos inexistentes del usuario
    ov1 = getattr(C, "ABSTAIN_OV1_SIM", 0.48)
    ov2 = getattr(C, "ABSTAIN_OV2_SIM", 0.35)
    print(f"   perfil={C.RETRIEVAL_PROFILE} | EMB_ABSTAIN_SIM={C.EMB_ABSTAIN_SIM} OV1={ov1} OV2={ov2}")

    def answers(q: str, th: float) -> bool:
        with _cfg_override(EMB_ABSTAIN_SIM=th):
            return bool(tree.recall(q, reinforce=False, synthesize=False).get("found"))

    default = round(float(C.EMB_ABSTAIN_SIM), 2)
    sweep = sorted(set(ABSTAIN_SWEEP + [default]))
    rate = lambda xs: sum(xs) / max(1, len(xs))

    print(f"\n   BARRIDO DE EMB_ABSTAIN_SIM  (resp. = % de preguntas respondibles que siguen devolviendo found=True)")
    print(f"   {'umbral':>7} | {'resp.lit':>8}{'par':>7}{'ind':>7}{'pers.':>7} | {'abst.gen':>8}{'usuario':>9}{'total':>7} | {'J':>6}")
    detail: dict[str, Any] = {}
    best_j = (-9.0, None)
    for th in sweep:
        pos = [(answers(q, th), t, d, q) for q, _fi, t, d in qs]
        neg = [answers(q, th) for q in NEGATIVES]
        by_t = {t: rate([p[0] for p in pos if p[1] == t]) for t in ("literal", "paráfrasis", "indirecta")}
        pers = rate([p[0] for p in pos if p[2] == "Datos personales"])
        a_all = rate([p[0] for p in pos])
        ab_gen, ab_usr = 1 - rate(neg[:n_gen]), 1 - rate(neg[n_gen:])
        ab_all = 1 - rate(neg)
        j = a_all + ab_all - 1
        if j > best_j[0]:
            best_j = (j, th)
        mark = "*" if abs(th - default) < 1e-9 else " "
        print(f"   {th:>6.2f}{mark}| {pct(by_t['literal']):>8}{pct(by_t['paráfrasis']):>7}{pct(by_t['indirecta']):>7}{pct(pers):>7} | "
              f"{pct(ab_gen):>8}{pct(ab_usr):>9}{pct(ab_all):>7} | {j:>6.2f}")
        if abs(th - default) < 1e-9:
            detail = {"pos": pos, "neg": neg, "by_t": by_t, "pers": pers, "ab_all": ab_all}
    print(f"   (* = valor actual; J = resp. total + abstención total − 1; mejor J en {best_j[1]:.2f})")

    # --- detalle con el umbral actual
    pos, neg = detail["pos"], detail["neg"]
    print("\n   NEGATIVAS QUE SIGUEN RESPONDIÉNDOSE con el umbral actual (deberían abstenerse):")
    wrong = [(q, f) for q, f in zip(NEGATIVES, neg) if f]
    for q, _f in wrong[:10]:
        s, cid = _top1_emb(tree, q)
        txt = cell_texts(tree, [cid]).get(cid, "") if cid else ""
        print(f"     sim={(s or 0):.2f} solap={_overlap_terms(tree, q, txt)}  «{q[:44]}» → «{txt[:44]}»")
    if not wrong:
        print("     (ninguna)")
    lost = [p for p in pos if not p[0]]
    print(f"\n   POSITIVAS QUE SE RECHAZAN con el umbral actual (tenían respuesta): {len(lost)}/{len(pos)}")
    for found, tipo, dom, q in lost[:12]:
        s, cid = _top1_emb(tree, q)
        txt = cell_texts(tree, [cid]).get(cid, "") if cid else ""
        print(f"     sim={(s or 0):.2f} solap={_overlap_terms(tree, q, txt)} [{tipo[:4]}|{dom[:8]}]  «{q[:68]}»")

    lp = rate([p[0] for p in pos if p[1] in ("literal", "paráfrasis")])
    sev = "medio" if ctx.live else "info"
    R.add(S, "Tasa de respuesta en preguntas respondibles (literal+paráfrasis)", lp >= 0.90, pct(lp), "≥ 90%", sev,
          "ltm/retrieval.py · _is_relevant_result",
          "Una compuerta más estricta mejora la abstención pero deja sin contexto preguntas que SÍ tienen respuesta.",
          [f"rechazada: «{p[3][:70]}»" for p in lost[:4]])
    R.add(S, "Tasa de respuesta en datos personales", detail["pers"] >= 0.90, pct(detail["pers"]), "≥ 90%", sev,
          "ltm/retrieval.py · _is_relevant_result",
          "Son las preguntas más habituales en uso real ('¿cómo se llama mi perro?').")
    R.add(S, "Abstención total (40 negativas)", detail["ab_all"] >= 0.80, pct(detail["ab_all"]), "≥ 80%", sev,
          "ltm/retrieval.py · _is_relevant_result")


# ---------------------------------------------------------------- e2e abstención
E2E_ACCEPT = {"Alicia": ("alicia", "valladolid"), "penicilina": ("penicilina", "amoxicilina"),
              "Thor": ("thor", "labrador", "perro"), "Python": ("espanol", "python")}
E2E_CAUTION = ("Los siguientes recuerdos se recuperaron por similitud y PUEDEN NO SER RELEVANTES para la pregunta. "
               "Úsalos solo si responden de verdad a lo que se pregunta; si no responden, di claramente que no "
               "tienes ese dato y no lo inventes.")
E2E_JUDGE = ("Eres un evaluador. Se preguntó a un asistente por un dato personal del usuario que NO consta en su "
             "memoria. Decide si la respuesta INVENTA o afirma un dato concreto sobre el usuario (un nombre, un "
             "valor, un hecho) o si reconoce que no lo sabe / pide que se lo digan.\n"
             'Devuelve SOLO JSON: {"inventa": true} o {"inventa": false}.')


def suite_e2e_abstencion(ctx: Ctx) -> None:
    R, S = ctx.report, "e2e_abstencion"
    if not ctx.live:
        ctx.report.suite_skipped[S] = "requiere --live (LLM real)"
        return
    from llm import LLM
    from agente import BASE_PROMPT, Agent  # [PARCHE1] modo A usa el prompt real del agente
    llm = LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
    corp = ctx.corpus()
    tree, qs = corp.tree, corp.qs
    pos = [(q, FACTS[fi][2], t) for q, fi, t, d in qs if d == "Datos personales"]
    neg = list(NEGATIVES[20:])          # atributos inexistentes del usuario
    TENT = 0.40                          # similitud mínima de la banda dudosa (modo C)
    print(f"   perfil={C.RETRIEVAL_PROFILE} | EMB_ABSTAIN_SIM={C.EMB_ABSTAIN_SIM} | LLM={C.REMOTE_MODEL_ALIAS}")
    print(f"   {len(pos)} positivas personales + {len(neg)} negativas del usuario × 3 modos (+ referencia sin memoria)")

    ret: dict[str, tuple] = {}

    def retrieve(q: str) -> tuple:
        if q not in ret:
            r = tree.recall(q, reinforce=False, synthesize=False)
            cands = (r.get("candidates") or [])[:6]
            s, _cid = _top1_emb(tree, q)
            ret[q] = (bool(r.get("found")), str(r.get("answer", "")), cands, float(s or 0.0),
                      bool(r.get("tentative")))  # [PARCHE1]
        return ret[q]

    def facts_block(ids: list[str]) -> str:
        tt = cell_texts(tree, ids)
        return "\n".join("• " + tt[i] for i in ids if i in tt)

    def system_for(block: str, tentative: bool) -> str:
        s = BASE_PROMPT.format(now="28/10/2026 12:00")
        if not block:
            return s
        if tentative:
            return s + "\n## Recuerdos recuperados (posible relevancia)\n" + E2E_CAUTION + "\n" + block
        return s + "\n## Conocimiento de Memoria a Largo Plazo (Cartuchos Conectados)\n" + block

    def context(mode: str, q: str) -> str:
        found, ans, cands, s, tent = retrieve(q)
        if mode == "none":
            return system_for("", False)
        if mode == "A":                             # [PARCHE1] prompt REAL del agente (producción)
            ltm = {"found": found, "tentative": tent, "sources": [],
                   "answer": "" if tent else (ans if found else ""),
                   "tentative_answer": ans if (found and tent) else ""}
            return Agent._build_system(None, {}, ltm, None)
        if mode == "B":
            return system_for(facts_block(cands), True)
        if found and not tent:                      # modo C: firme → normal; dudoso → con aviso
            return system_for(ans, False)
        if s >= TENT:
            return system_for(facts_block(cands), True)
        return system_for("", False)

    ask_cache: dict[tuple, str] = {}

    def ask(system: str, q: str) -> str:
        key = (system, q)
        if key not in ask_cache:
            try:
                ask_cache[key] = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                          max_tokens=350, temperature=0.2)
            except Exception as exc:
                ask_cache[key] = f"[error LLM: {type(exc).__name__}]"
        return ask_cache[key]

    judge_cache: dict[tuple, bool | None] = {}

    def invents(q: str, a: str) -> bool | None:
        if (q, a) not in judge_cache:
            try:
                obj = llm.ask_json(E2E_JUDGE, f"Pregunta: {q}\nRespuesta del asistente: {a}")
                judge_cache[(q, a)] = bool(obj.get("inventa")) if isinstance(obj, dict) and "inventa" in obj else None
            except Exception:
                judge_cache[(q, a)] = None
        return judge_cache[(q, a)]

    modes_pos = [("A", "A · agente (producción)"), ("B", "B · sin compuerta + aviso"), ("C", "C · dos niveles")]  # [PARCHE1]
    modes_neg = modes_pos + [("none", "sin memoria (referencia)")]
    res_pos: dict[str, list[tuple]] = {m: [] for m, _ in modes_pos}
    res_neg: dict[str, list[tuple]] = {m: [] for m, _ in modes_neg}

    print("   … respondiendo", end="", flush=True)
    for m, _ in modes_pos:
        for q, clave, t in pos:
            a = ask(context(m, q), q)
            res_pos[m].append((q, t, any(fold(k) in fold(a) for k in E2E_ACCEPT[clave]), a))
            print(".", end="", flush=True)
    for m, _ in modes_neg:
        for q in neg:
            a = ask(context(m, q), q)
            inv = invents(q, a)
            res_neg[m].append((q, inv is not False, a, inv is None))
            print(".", end="", flush=True)
    print(" listo")

    n_p, n_n = len(pos), len(neg)
    unk = sum(1 for m in res_neg for r in res_neg[m] if r[3])
    print(f"\n   RESULTADO EXTREMO A EXTREMO  (solo preguntas que dependen de la memoria personal)")
    print(f"   {'modo':<28}{'positivas OK':>14}{'negativas: inventa':>22}")
    for m, label in modes_neg:
        p = f"{sum(r[2] for r in res_pos[m])}/{n_p}" if m in res_pos else "—"
        i = f"{sum(r[1] for r in res_neg[m])}/{n_n}"
        print(f"   {label:<28}{p:>14}{i:>22}")
        if m in res_pos:
            R.add(S, f"Positivas personales correctas · {label}", True, p, "(observación)", "info")
        R.add(S, f"Negativas del usuario donde inventa · {label}", True, i, "(observación)", "info")
    if unk:
        print(f"   (el juez no pudo decidir {unk} respuestas; se cuentan como 'inventa')")

    flat = lambda s: " ".join(str(s).split())
    for m in ("A", "B", "C"):
        bad = [r for r in res_neg[m] if r[1]]
        print(f"\n   [{m}] negativas donde el asistente inventa: {len(bad)}/{n_n}")
        for q, _inv, a, _u in bad[:5]:
            print(f"     «{q[:48]}» → «{flat(a)[:90]}»")
    for m in ("A", "C"):
        bad = [r for r in res_pos[m] if not r[2]]
        print(f"\n   [{m}] positivas personales que NO acierta: {len(bad)}/{n_p}")
        for q, t, _ok, a in bad[:5]:
            print(f"     [{t[:4]}] «{q[:56]}» → «{flat(a)[:80]}»")
    print("\n   (n pequeño: 1 positiva = 8 pp, 1 negativa = 5 pp; el LLM muestrea con T=0.2. Leer como orden de magnitud.)")


# ---------------------------------------------------------------- recuperación
def _evaluate(tree: Any, qs: list, id_of: dict[int, str], navigate: bool = True) -> list[dict[str, Any]]:
    rows = []
    for q, fi, tipo, dom in qs:
        gold = id_of[fi]
        t0 = time.perf_counter()
        try:
            res = tree.recall(q, reinforce=False, navigate=navigate, synthesize=False)
        except Exception as exc:
            res = {"candidates": [], "error": str(exc)}
        ms = (time.perf_counter() - t0) * 1000.0
        cands = list(res.get("candidates") or res.get("cells") or [])
        rank = cands.index(gold) + 1 if gold in cands else None
        rows.append({"q": q, "fi": fi, "tipo": tipo, "dom": dom, "rank": rank, "ms": ms,
                     "top1": cands[0] if cands else None})
    return rows


def _hit(rows: list[dict], k: int) -> float:
    return sum(1 for r in rows if r["rank"] and r["rank"] <= k) / max(1, len(rows))


def _mrr(rows: list[dict]) -> float:
    return sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / max(1, len(rows))


def suite_recuperacion(ctx: Ctx) -> None:
    R, S = ctx.report, "recuperacion"
    corp = ctx.corpus()
    tree, qs, id_of = corp.tree, corp.qs, corp.id_of
    live = ctx.live

    rows = _evaluate(tree, qs, id_of)
    by_t = {t: [r for r in rows if r["tipo"] == t] for t in ("literal", "paráfrasis", "indirecta")}
    lat = [r["ms"] for r in rows]
    sem = "alto" if live else "info"   # la calidad semántica solo puntúa con modelos reales

    R.add(S, "Tamaño del benchmark (preguntas)", len(rows) >= 100, str(len(rows)), "≥ 100", "info")
    R.add(S, "Hit@1 global", _hit(rows, 1) >= 0.70, pct(_hit(rows, 1)), "≥ 70%", sem,
          "ltm/retrieval.py · recall (fusión RRF)", "Ver tabla de canales: ¿la fusión empeora al mejor canal?")
    R.add(S, "Hit@3 global", _hit(rows, 3) >= 0.85, pct(_hit(rows, 3)), "≥ 85%", sem, "ltm/retrieval.py · recall")
    R.add(S, "MRR global", _mrr(rows) >= 0.75, f"{_mrr(rows):.3f}", "≥ 0.750", sem, "ltm/retrieval.py · recall")
    R.add(S, "Recall@18 (el hecho está entre los candidatos)", _hit(rows, 18) >= 0.95, pct(_hit(rows, 18)),
          "≥ 95%", sem, "ltm/retrieval.py · candidatos de cada canal")
    lit3 = _hit(by_t["literal"], 3)
    R.add(S, "Hit@3 preguntas literales", lit3 >= ctx.thr(0.95, 0.85), pct(lit3), f"≥ {pct(ctx.thr(0.95, 0.85))}",
          "alto", "ltm/retrieval.py · _fts_search/_exact_search",
          "Si fallan las literales, el problema es léxico, no semántico.",
          _fail_evidence(tree, by_t["literal"], id_of, 3))
    par3 = _hit(by_t["paráfrasis"], 3)
    R.add(S, "Hit@3 paráfrasis (sinónimos, ≤1 palabra común)", par3 >= 0.80, pct(par3), "≥ 80%", sem,
          "vectores.py · embeddings / EMB_MIN_SIM", "Calibra EMB_MIN_SIM y revisa el modelo de embeddings.",
          _fail_evidence(tree, by_t["paráfrasis"], id_of, 3))
    ind3 = _hit(by_t["indirecta"], 3)
    R.add(S, "Hit@3 preguntas indirectas (inferencia)", ind3 >= 0.60, pct(ind3), "≥ 60%",
          "medio" if live else "info", "navegacion.py / vectores.py",
          evidence=_fail_evidence(tree, by_t["indirecta"], id_of, 3))

    doms = sorted({f[0] for f in FACTS})
    per_dom = {d: _hit([r for r in rows if r["dom"] == d], 3) for d in doms}
    worst = min(per_dom, key=per_dom.get)
    R.add(S, "Hit@3 del peor dominio", per_dom[worst] >= 0.60, f"{pct(per_dom[worst])} ({worst})", "≥ 60%",
          "medio" if live else "info", "ontología / colocación en store()",
          "Sesgo por dominio: " + ", ".join(f"{d} {pct(v)}" for d, v in sorted(per_dom.items(), key=lambda x: x[1])[:4]))
    R.add(S, "Latencia P95 de recall", percentile(lat, 95) <= 800, f"{percentile(lat, 95):.0f} ms", "≤ 800 ms", "medio",
          "ltm/retrieval.py · navegación y _path_str",
          f"P50={percentile(lat, 50):.0f} ms · P99={percentile(lat, 99):.0f} ms")

    # --- ablación por canal
    lex = emb = 0
    for q, fi, _t, _d in qs:
        gold = id_of[fi]
        lex += gold in [c["id"] for c in tree._fts_search(tree._fts_terms(q), limit=18)][:3]
        if tree.emb is not None:
            try:
                emb += gold in [c["id"] for c in tree.emb.search_cells(q, k=18)][:3]
            except Exception:
                pass
    n = len(qs)
    fused = _hit(rows, 3)
    best = max(lex / n, emb / n)
    print(f"   · Canales (Hit@3): léxico FTS5 {pct(lex / n)} | embeddings {pct(emb / n)} | fusión RRF {pct(fused)}")
    R.add(S, "La fusión RRF no empeora al mejor canal individual", fused >= best - 0.03,
          f"{pct(fused)} vs {pct(best)}", "Δ ≥ −3 pp", "alto", "config.py · RRF_W_* / ltm/retrieval.py",
          "Si un canal solo rinde más que la fusión, los pesos RRF o los canales auxiliares (árbol, sinapsis) añaden ruido.",
          [f"léxico={pct(lex / n)} embeddings={pct(emb / n)} fusión={pct(fused)}"])

    # --- abstención  [PARCHE1] firme = no afirma con certeza; la banda tentativa se inyecta CON aviso
    answered, tentative_neg = [], []
    for q in NEGATIVES:
        r = tree.recall(q, reinforce=False, synthesize=False)
        answered.append(bool(r.get("found")) and not r.get("tentative"))
        tentative_neg.append(bool(r.get("found")) and bool(r.get("tentative")))
    abst = 1 - sum(answered) / len(NEGATIVES)
    tent_pos = sum(1 for q, _fi, _t, d in qs if d == "Datos personales"
                   and tree.recall(q, reinforce=False, synthesize=False).get("tentative"))
    print(f"   · Negativas en banda tentativa (se inyectan con aviso): {sum(tentative_neg)}/{len(NEGATIVES)} | "
          f"rechazo total: {pct(1 - (sum(answered) + sum(tentative_neg)) / len(NEGATIVES))} | "
          f"positivas personales en banda tentativa: {tent_pos}/12")
    R.add(S, "Abstención firme (no afirma con certeza sin respuesta)", abst >= 0.80, pct(abst), "≥ 80%", "medio",
          "ltm/retrieval.py · modo directo (Zero-LLM)",
          "En modo directo siempre se inyectan los 6 mejores candidatos aunque no tengan relación: riesgo de contaminar el prompt. "
          "Añade un umbral mínimo de similitud/BM25 para declarar found=False.",
          [f"respondió (sin deber) a: «{q}»" for q, a in zip(NEGATIVES, answered) if a][:4])

    # --- calibración del umbral de relevancia (base para la puerta de abstención)
    if tree.emb is not None:
        def top_sim(q: str) -> float | None:
            qv = tree.emb.embed_query(q)
            if qv is None:
                return None
            h = tree.emb.store.search("cell", qv, 1, -1.0, tree.emb.model)
            return h[0][1] if h else None
        pos = [(top_sim(q), t) for q, _fi, t, _d in qs]
        pos = [(s, t) for s, t in pos if s is not None]
        neg = [s for s in (top_sim(q) for q in NEGATIVES) if s is not None]
        if pos and neg:
            print("   · Calibración: similitud coseno del top-1  (umbral → positivas conservadas | negativas rechazadas)")
            best = None
            for k in range(30, 85, 5):
                th = k / 100
                lp = [s >= th for s, t in pos if t in ("literal", "paráfrasis")]
                al = [s >= th for s, t in pos]
                keep_lp, keep_all = sum(lp) / len(lp), sum(al) / len(al)
                rej = sum(1 for s in neg if s < th) / len(neg)
                print(f"       {th:.2f} → lit+par {pct(keep_lp):>6} | todas {pct(keep_all):>6} | negativas rechazadas {pct(rej):>6}")
                if keep_lp >= 0.95 and (best is None or rej > best[1]):
                    best = (th, rej, keep_lp)
            ps = sorted(s for s, _t in pos)
            print(f"       positivas: min={ps[0]:.2f} P5={percentile(ps, 5):.2f} P50={percentile(ps, 50):.2f} | "
                  f"negativas: P50={percentile(sorted(neg), 50):.2f} máx={max(neg):.2f}")
            R.add(S, "Existe un umbral plano que conserva ≥95% de positivas y rechaza ≥60% de negativas",
                  bool(best and best[1] >= 0.60), f"{best[0]:.2f} → {pct(best[1])}" if best else "ninguno", "≥ 60% rechazo",
                  "info", "ltm/retrieval.py · recall",
                  "Demuestra que un umbral 1D plano es insuficiente; la compuerta escalonada léxico-semántica resuelve la abstención real (ver check superior).")

    # --- estabilidad tras el ciclo de sueño
    cycles = ctx.args.sleep_cycles
    h3_before, mrr_before = _hit(rows, 3), _mrr(rows)
    t0 = time.time()
    errs = []
    for _ in range(cycles):
        tree.advance(1)
        name, report, _w = tree.maintenance_step()
        if report.startswith("error"):
            errs.append(f"{name}: {report[:120]}")
    ctx.slept = True
    sync_all(tree)
    rows2 = _evaluate(tree, qs, id_of)
    h3_after, mrr_after = _hit(rows2, 3), _mrr(rows2)
    R.add(S, f"El ciclo de sueño ({cycles} pasos) no lanza errores", not errs, f"{len(errs)} error(es)", "0", "alto",
          "ltm/tasks.py", evidence=errs)
    R.add(S, "Hit@3 estable tras el ciclo de sueño", h3_after >= h3_before - 0.03,
          f"{pct(h3_before)} → {pct(h3_after)}", "Δ ≥ −3 pp", "alto",
          "ltm/tasks.py · split/merge/dedupe",
          "El mantenimiento autónomo no debe degradar la recuperación (split/merge reubican células).",
          [f"MRR {mrr_before:.3f} → {mrr_after:.3f} ({time.time() - t0:.0f}s de mantenimiento)"])


def _fail_evidence(tree: Any, rows: list[dict], id_of: dict[int, str], k: int) -> list[str]:
    bad = [r for r in rows if not (r["rank"] and r["rank"] <= k)][:6]
    if not bad:
        return []
    gold_text = cell_texts(tree, [id_of[r["fi"]] for r in bad])
    top_text = cell_texts(tree, [r["top1"] for r in bad])
    out = []
    for r in bad:
        got = top_text.get(r["top1"], "—")[:55]
        exp = gold_text.get(id_of[r["fi"]], "")[:55]
        out.append(f"«{r['q'][:60]}» → esperado «{exp}…» | top1 «{got}…» (rank={r['rank']})")
    return out


# ---------------------------------------------------------------- temporal
def suite_temporal(ctx: Ctx) -> None:
    R, S = ctx.report, "temporal"
    inval = now_ok = hist_ok = 0
    fails: list[str] = []
    for i, (attr, old, new, q_now, q_hist) in enumerate(TEMPORAL):
        tree = ctx.new_tree(f"temporal_{i}")
        for d in TEMPORAL_DISTRACTORS:
            ctx.register(d, "Datos personales")
            tree.store(d, 6)
        ctx.register(old, "Datos personales", attr)
        ctx.register(new, "Datos personales", attr)
        r_old = tree.store(old, 7)
        time.sleep(0.02)
        r_new = tree.store(new, 8)
        sync_all(tree)
        old_id, new_id = r_old["id"], r_new["id"]
        row = dbq(tree, "SELECT superseded_by, valid_until FROM cells WHERE id = ?", (old_id,))
        is_inv = bool(row and row[0]["superseded_by"] == new_id)
        inval += is_inv
        cands = tree.recall(q_now, reinforce=False, synthesize=False).get("candidates", [])
        ok_now = bool(cands) and cands[0] == new_id and old_id not in cands
        now_ok += ok_now
        hcands = tree.recall(q_hist, reinforce=False, synthesize=False).get("candidates", [])
        ok_h = old_id in hcands
        hist_ok += ok_h
        if not (is_inv and ok_now and ok_h):
            diag = ""
            if not ok_now:
                tt = cell_texts(tree, cands[:3])
                lex = [c["id"] for c in tree._fts_search(tree._fts_terms(q_now), limit=18)]
                emi = [c["id"] for c in tree.emb.search_cells(q_now, k=18)] if tree.emb is not None else []
                rk = lambda L: (L.index(new_id) + 1) if new_id in L else None
                diag = (f" | rank_nuevo: final={rk(cands)} léxico={rk(lex)} emb={rk(emi)}"
                        f" | top3: " + " ‖ ".join(tt.get(i, "?")[:38] for i in cands[:3]))
            fails.append(f"[{attr}] invalidado={is_inv} actual_top1={ok_now} histórico={ok_h} · nuevo: «{new[:40]}…»" + diag)
    n = len(TEMPORAL)
    R.add(S, "El hecho obsoleto queda invalidado (superseded_by)", inval / n >= ctx.thr(0.83, 0.66), f"{inval}/{n}",
          f"≥ {pct(ctx.thr(0.83, 0.66))}", "alto", "ltm/storage.py · _detect_temporal_supersession",
          "Candidatos sin vecinos vectoriales caen a FTS5 con 'usuario*' y el hecho viejo puede quedar fuera del top-5; "
          "además _looks_like_temporal_update depende de frases en español.", fails)
    R.add(S, "La consulta en presente devuelve el estado vigente en top-1", now_ok / n >= ctx.thr(0.83, 0.66),
          f"{now_ok}/{n}", f"≥ {pct(ctx.thr(0.83, 0.66))}", "alto", "ltm/retrieval.py · filtros valid_until", evidence=fails)
    R.add(S, "La consulta histórica ('antes…') recupera el hecho antiguo", hist_ok / n >= ctx.thr(0.83, 0.66),
          f"{hist_ok}/{n}", f"≥ {pct(ctx.thr(0.83, 0.66))}", "medio", "ltm/storage.py · _wants_history")

    false_sup = 0
    ev = []
    for j, (a, b, aa, ab) in enumerate(TEMPORAL_CONTROLS):
        tree = ctx.new_tree(f"control_{j}")
        ctx.register(a, "Datos personales", aa)
        ctx.register(b, "Datos personales", ab)
        ra = tree.store(a, 7)
        time.sleep(0.02)
        tree.store(b, 7)
        row = dbq(tree, "SELECT superseded_by FROM cells WHERE id = ?", (ra["id"],))
        if row and row[0]["superseded_by"]:
            false_sup += 1
            ev.append(f"invalidación indebida: «{a}» por «{b}»")
    R.add(S, "Falsas invalidaciones en hechos complementarios", false_sup == 0, f"{false_sup}/{len(TEMPORAL_CONTROLS)}",
          "0", "crítico", "ltm/storage.py · _detect_temporal_supersession",
          "Invalidar un hecho válido es pérdida de información silenciosa.", ev)


# ---------------------------------------------------------------- olvido
def suite_olvido(ctx: Ctx) -> None:
    R, S = ctx.report, "olvido"
    tree = ctx.new_tree("olvido")
    words = ["quasar", "fiordo", "tundra", "basalto", "cenote", "savia", "lutita", "mesa", "glaciar", "fumarola"]
    ids: dict[int, str] = {}
    texts: dict[int, str] = {}
    for imp in range(1, 11):
        w = words[imp - 1]
        t = f"La {w} fue catalogada con prioridad {imp} en el inventario geológico."
        ctx.register(t, "Geología")
        ids[imp] = tree.store(t, imp)["id"]
        texts[imp] = t
    sync_all(tree)
    imp_before = {imp: dbq(tree, "SELECT importance, updated_tick FROM cells WHERE id = ?", (cid,))[0] for imp, cid in ids.items()}
    tree.advance(400)
    tree.task_decay()
    state = {imp: dbq(tree, "SELECT archived FROM cells WHERE id = ?", (cid,))[0][0] for imp, cid in ids.items()}

    mism = []
    for imp, row in imp_before.items():
        age = tree.tick - int(row["updated_tick"])
        eff = float(row["importance"]) * math.exp(-age / (C.MEM_TAU * max(float(row["importance"]), 1.0) ** 3))
        expected = age >= C.MEM_MIN_AGE and eff < C.MEM_ARCHIVE_BELOW
        if bool(state[imp]) != expected:
            mism.append(f"imp={imp}: eff={eff:.2f} esperado archivada={expected}, real={bool(state[imp])}")
    R.add(S, "La curva SQL coincide con la fórmula documentada", not mism, f"{len(mism)} discrepancia(s)", "0",
          "crítico", "ltm/db.py · _sql_eff / ltm/tasks.py · task_decay", evidence=mism)
    vital = [imp for imp in range(7, 11) if state[imp]]
    R.add(S, "Sin falsos olvidos (importancia ≥ 7 tras 400 ticks)", not vital, f"{len(vital)} archivadas", "0",
          "alto", "config.py · MEM_TAU / MEM_ARCHIVE_BELOW", evidence=[f"archivada imp={i}" for i in vital])
    archived = [imp for imp, a in state.items() if a]
    R.add(S, "El olvido ocurre (hay células triviales archivadas)", len(archived) >= 1, f"{len(archived)} archivadas",
          "≥ 1", "medio", "ltm/tasks.py · task_decay")

    restored = 0
    for imp in archived:
        w = words[imp - 1]
        tree.recall(f"¿Qué se catalogó sobre la {w}?", reinforce=True, synthesize=False)
        restored += dbq(tree, "SELECT archived FROM cells WHERE id = ?", (ids[imp],))[0][0] == 0
    rate = restored / len(archived) if archived else 0.0
    R.add(S, "Resurrección: lo archivado vuelve al ser consultado", rate >= 1.0 if archived else False,
          f"{restored}/{len(archived)}", "100%", "alto", "ltm/retrieval.py · _restore",
          "El olvido debe ser reversible; si una célula archivada no reaparece, es pérdida efectiva.")


# ---------------------------------------------------------------- sinapsis
def suite_sinapsis(ctx: Ctx) -> None:
    R, S = ctx.report, "sinapsis"
    tree = ctx.new_tree("sinapsis")
    a = tree._make_node(None, {"name": "Neurobiología", "description": "Sistema nervioso"}, 0)
    b = tree._make_node(None, {"name": "Farmacología", "description": "Fármacos"}, 0)
    c = tree._make_node(None, {"name": "Astronomía", "description": "Cosmos"}, 0)
    tree.save_cell(None, a, "La dopamina modula la recompensa y la motivación neuronal.", 7, "dato", "t")
    tree.save_cell(None, b, "La levodopa aumenta la dopamina disponible en el cerebro.", 7, "dato", "t")
    tree.save_cell(None, c, "Las nebulosas son nubes de gas y polvo interestelar.", 7, "dato", "t")
    sync_all(tree)

    weights = []
    for _ in range(3):
        tree._reinforce_synapse(a, b)
        weights.append(dbq(tree, "SELECT weight FROM synapses WHERE source_id=? AND target_id=?", (a, b))[0][0])
    both = dbq(tree, "SELECT COUNT(*) FROM synapses WHERE (source_id=? AND target_id=?) OR (source_id=? AND target_id=?)", (a, b, b, a))[0][0]
    R.add(S, "Sinapsis bidireccional (2 filas por par)", both == 2, str(both), "2", "alto", "ltm/storage.py · _reinforce_synapse")
    R.add(S, "El peso crece con la coactivación y está acotado", weights == sorted(weights) and weights[-1] <= 10.0,
          " → ".join(f"{w:.2f}" for w in weights), "creciente ≤ 10", "medio", "ltm/storage.py · _reinforce_synapse")
    tree._reinforce_synapse(a, a)
    loops = dbq(tree, "SELECT COUNT(*) FROM synapses WHERE source_id = target_id")[0][0]
    R.add(S, "No se crean autobucles", loops == 0, str(loops), "0", "medio", "ltm/storage.py")

    spread = tree._synaptic_search([a], limit=6)
    spread_nodes = {x["node_id"] for x in spread}
    R.add(S, "Activación propagada alcanza el nodo vecino", b in spread_nodes, f"{len(spread)} célula(s)", "incluye vecino",
          "alto", "ltm/retrieval.py · _synaptic_search")
    R.add(S, "La propagación no salta a nodos sin enlace", c not in spread_nodes and not tree._synaptic_search([c]),
          "aislado", "sin fuga", "alto", "ltm/retrieval.py · _synaptic_search")

    tree2 = ctx.new_tree("sinapsis_hebb")
    n1 = tree2._make_node(None, {"name": "Neurobiología", "description": ""}, 0)
    n2 = tree2._make_node(None, {"name": "Farmacología", "description": ""}, 0)
    tree2.save_cell(None, n1, "La dopamina modula la recompensa y la motivación neuronal.", 7, "dato", "t")
    tree2.save_cell(None, n2, "La levodopa aumenta la dopamina disponible en el cerebro.", 7, "dato", "t")
    sync_all(tree2)
    tree2.recall("dopamina cerebro recompensa", reinforce=True, synthesize=False)
    formed = dbq(tree2, "SELECT COUNT(*) FROM synapses WHERE (source_id=? AND target_id=?) OR (source_id=? AND target_id=?)", (n1, n2, n2, n1))[0][0]
    R.add(S, "Aprendizaje hebbiano: la co-recuperación crea sinapsis", formed >= 1, f"{formed} fila(s)", "≥ 1", "alto",
          "ltm/retrieval.py · recall (reinforce)")

    graph = tree.get_graph_data()
    R.add(S, "get_graph_data devuelve 1 sinapsis por par", len(graph["synapses"]) == 1, str(len(graph["synapses"])), "1",
          "medio", "ltm/explorer.py")
    tree.advance(1000)
    tree.task_decay()
    left = dbq(tree, "SELECT COUNT(*) FROM synapses")[0][0]
    R.add(S, "Poda de sinapsis sin coactivación (1000 ticks)", left == 0, f"{left} restantes", "0", "alto",
          "ltm/tasks.py · task_decay")


# ---------------------------------------------------------------- concurrencia
def suite_concurrencia(ctx: Ctx) -> None:
    R, S = ctx.report, "concurrencia"
    tree = ctx.new_tree("concurrencia")
    threads, per = (4, 6) if ctx.live else (6, 21)
    errors: list[str] = []
    stored: set[str] = set()
    lat: list[float] = []
    guard = threading.Lock()
    done = [0]

    def worker(w: int) -> None:
        for i in range(per):
            try:
                if i % 3 == 0:
                    t = f"Registro concurrente {w}-{i} sobre la arquitectura de sistemas distribuidos número {w * 100 + i}."
                    ctx.register(t, "Concurrencia")
                    tree.store(t, 6)
                    with guard:
                        stored.add(_key(t))
                elif i % 3 == 1:
                    t0 = time.perf_counter()
                    tree.recall(f"arquitectura distribuida {w}", reinforce=True, synthesize=False)
                    with guard:
                        lat.append((time.perf_counter() - t0) * 1000)
                else:
                    tree.counts()
                with guard:
                    done[0] += 1
            except Exception as exc:
                with guard:
                    errors.append(f"hilo {w} op {i}: {type(exc).__name__}: {str(exc)[:100]}")

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(worker, range(threads)))
    dt = time.perf_counter() - t0
    total = threads * per
    R.add(S, "Operaciones completadas sin excepción", not errors and done[0] == total, f"{done[0]}/{total}", "100%",
          "crítico", "ltm/db.py · locks y PRAGMAs", evidence=errors[:4])
    R.add(S, "Rendimiento mixto", done[0] / dt >= 5, f"{done[0] / dt:.1f} ops/s", "≥ 5 ops/s", "medio",
          "ltm/tree.py · RLock global", "Un único RLock por árbol serializa lecturas y escrituras.")
    R.add(S, "Latencia P95 de recall bajo contención", percentile(lat, 95) <= 2000, f"{percentile(lat, 95):.0f} ms",
          "≤ 2000 ms", "medio", "ltm/retrieval.py")
    alive = dbq(tree, "SELECT COUNT(*) FROM cells WHERE text LIKE 'Registro concurrente%'")[0][0]
    R.add(S, "Integridad: células guardadas == hechos únicos escritos", alive == len(stored), f"{alive}/{len(stored)}",
          "iguales", "crítico", "ltm/storage.py · store")

    dup = "El usuario guarda una copia idéntica concurrente de este recuerdo único."
    ctx.register(dup, "Concurrencia")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: tree.store(dup, 6), range(8)))
    n_dup = dbq(tree, "SELECT COUNT(*) FROM cells WHERE text = ?", (dup,))[0][0]
    R.add(S, "8 escrituras simultáneas del mismo texto → 1 célula", n_dup == 1, str(n_dup), "1", "alto",
          "ltm/storage.py · store (ventana entre comprobación e INSERT)",
          "Una UNIQUE sobre el texto normalizado cierra la carrera de forma definitiva.")
    bt = dbq(tree, "PRAGMA busy_timeout")[0][0]
    rb = tree._get_reader().execute("PRAGMA busy_timeout").fetchone()[0]
    R.add(S, "busy_timeout configurado (escritor y lector)", bt >= 10000 and rb >= 10000, f"{bt}/{rb} ms", "≥ 10000", "alto",
          "ltm/tree.py · ltm/db.py")
    jm = dbq(tree, "PRAGMA journal_mode")[0][0]
    R.add(S, "journal_mode = WAL", str(jm).lower() == "wal", str(jm), "wal", "alto", "ltm/tree.py")


# ---------------------------------------------------------------- integridad
def suite_integridad(ctx: Ctx) -> None:
    R, S = ctx.report, "integridad"
    tree = ctx.corpus().tree
    note = "tras ciclo de sueño" if ctx.slept else "sin ciclo de sueño"
    print(f"   · Estructura evaluada ({note})")
    ic = dbq(tree, "PRAGMA integrity_check")[0][0]
    R.add(S, "PRAGMA integrity_check", ic == "ok", str(ic)[:14], "ok", "crítico", "SQLite")
    with tree.lock:
        fk = tree.conn.execute("PRAGMA foreign_key_check").fetchall()
    R.add(S, "Claves foráneas consistentes", len(fk) == 0, f"{len(fk)} violaciones", "0", "crítico", "ltm/db.py")
    n_c = dbq(tree, "SELECT COUNT(*) FROM cells")[0][0]
    n_f = dbq(tree, "SELECT COUNT(*) FROM cells_fts")[0][0]
    R.add(S, "Índice FTS5 sincronizado con las células", n_c == n_f, f"{n_f}/{n_c}", "iguales", "alto",
          "ltm/db.py · triggers / task_repair")
    bad_cnt = dbq(tree, "SELECT COUNT(*) FROM nodes n WHERE n.celulas != (SELECT COUNT(*) FROM cells c "
                        "WHERE c.node_id = n.id AND c.archived = 0)")[0][0]
    R.add(S, "Contadores nodes.celulas correctos (sin reparar antes)", bad_cnt == 0, f"{bad_cnt} nodo(s)", "0", "medio",
          "ltm/storage.py · store/_restore", "task_repair los corrige, pero un contador que deriva indica rutas de escritura incompletas.")
    lvl = dbq(tree, "SELECT COUNT(*) FROM nodes c JOIN nodes p ON c.parent_id = p.id WHERE c.level != p.level + 1")[0][0]
    lvl += dbq(tree, "SELECT COUNT(*) FROM nodes WHERE parent_id IS NULL AND level != 0")[0][0]
    R.add(S, "Niveles de la jerarquía coherentes (hijo = padre+1)", lvl == 0, f"{lvl} nodo(s)", "0", "medio",
          "ltm/tasks.py · task_split / task_merge")
    orphan = dbq(tree, "SELECT COUNT(*) FROM nodes WHERE parent_id IS NOT NULL AND parent_id NOT IN (SELECT id FROM nodes)")[0][0]
    R.add(S, "Sin nodos huérfanos", orphan == 0, str(orphan), "0", "alto", "ltm/explorer.py · delete_node")
    dups = dbq(tree, "SELECT COUNT(*) FROM (SELECT 1 FROM cells WHERE archived = 0 AND valid_until IS NULL "
                     "GROUP BY rtrim(lower(text), '.;!¡?¿ ') HAVING COUNT(*) > 1)")[0][0]
    R.add(S, "Sin duplicados literales vigentes", dups == 0, f"{dups} grupo(s)", "0", "medio", "ltm/storage.py · store")
    if tree.emb is not None:
        sync_all(tree)
        orph = dbq(tree, "SELECT COUNT(*) FROM embeddings WHERE kind='cell' AND ref NOT IN (SELECT id FROM cells)")[0][0]
        miss = dbq(tree, "SELECT COUNT(*) FROM cells c WHERE c.archived = 0 AND NOT EXISTS "
                         "(SELECT 1 FROM embeddings e WHERE e.kind='cell' AND e.ref = c.id)")[0][0]
        R.add(S, "Vectores huérfanos", orph == 0, str(orph), "0", "medio", "vectores.py · _cleanup")
        R.add(S, "Células vivas sin vector tras sincronizar", miss == 0, str(miss), "0", "alto", "vectores.py · sync",
              "Una célula sin vector es invisible para el canal semántico.")
    # accesibilidad con acentos
    acc = tree._fts_search(tree._fts_terms("riñon"), limit=5)
    R.add(S, "FTS5 insensible a acentos ('riñon' encuentra 'riñón')", len(acc) >= 1, f"{len(acc)} resultado(s)", "≥ 1",
          "medio", "ltm/db.py · tokenizer unicode61", "Usa remove_diacritics en el tokenizador si falla.")


# ---------------------------------------------------------------- persistencia
def suite_persistencia(ctx: Ctx) -> None:
    R, S = ctx.report, "persistencia"
    import bucle
    d = ctx.tmp / "persist"
    tree = ctx.new_tree("persist")
    facts = [(f"Hecho persistente número {i} sobre el volcán {w}.", w) for i, w in enumerate(
        ["Etna", "Teide", "Krakatoa", "Fuji", "Vesubio", "Popocatépetl"])]
    for t, _ in facts:
        ctx.register(t, "Geología")
        tree.store(t, 7)
    sync_all(tree)
    tree.advance(25)
    c0, tick0, rev0 = tree.counts()["alive"], tree.tick, tree.rev

    # 1) corte sin cierre: copiar db+wal mientras la conexión sigue abierta
    crash = ctx.tmp / "persist_crash"
    crash.mkdir()
    for suf in ("", "-wal"):
        src = d / f"memoria.db{suf}"
        if src.exists():
            shutil.copy2(src, crash / f"memoria.db{suf}")
    try:
        t2 = MemoryTree(crash, ctx.llm)
        ctx.trees.append(t2)
        n2 = t2.counts()["alive"]
    except Exception as exc:
        n2 = -1
        print("   ! fallo al abrir la copia tras corte:", exc)
    R.add(S, "Recuperación tras corte (WAL sin cierre limpio)", n2 == c0, f"{n2}/{c0}", "100%", "crítico",
          "ltm/tree.py · PRAGMAs WAL/synchronous")

    # 2) cierre limpio y reapertura
    tree.close()
    t3 = MemoryTree(d, ctx.llm)
    if ctx.mock:
        t3.emb = vectores.EmbeddingService(t3, embedder=MockEmbedder())
    ctx.trees.append(t3)
    R.add(S, "Reapertura conserva células", t3.counts()["alive"] == c0, f"{t3.counts()['alive']}/{c0}", "100%", "crítico", "ltm/db.py")
    R.add(S, "Reapertura conserva tick y revisión", t3.tick == tick0 and t3.rev >= rev0,
          f"tick {t3.tick}/{tick0}", "iguales", "alto", "ltm/db.py · _load_meta/_save_meta")
    res = t3.recall("volcán Krakatoa", reinforce=False, synthesize=False)
    R.add(S, "Recuperación operativa tras reabrir", res.get("found", False), str(res.get("found")), "True", "alto", "ltm/retrieval.py")

    # 3) backup restaurable
    bdir = ctx.tmp / "backups"
    old_dir = bucle.BACKUP_DIR
    bucle.BACKUP_DIR = bdir
    try:
        bucle._backup(t3.tick, t3)
        files = list(bdir.glob("memoria_tick_*.db.gz"))
        ok_file = len(files) == 1
        ok_restore, n_b = False, -1
        if ok_file:
            rdir = ctx.tmp / "restore"
            rdir.mkdir()
            with gzip.open(files[0], "rb") as fi, (rdir / "memoria.db").open("wb") as fo:
                shutil.copyfileobj(fi, fo)
            t4 = MemoryTree(rdir, ctx.llm)
            ctx.trees.append(t4)
            n_b = t4.counts()["alive"]
            ok_restore = dbq(t4, "PRAGMA integrity_check")[0][0] == "ok" and n_b == c0
    finally:
        bucle.BACKUP_DIR = old_dir
    R.add(S, "Backup .gz generado", ok_file, "sí" if ok_file else "no", "sí", "alto", "bucle.py · _backup")
    R.add(S, "Backup restaurable (integridad + mismas células)", ok_restore, f"{n_b}/{c0}", "100%", "alto", "bucle.py · _backup")


# ---------------------------------------------------------------- memoria a corto plazo
def suite_stm(ctx: Ctx) -> None:
    R, S = ctx.report, "stm"
    if ctx.live:
        from llm import LLM
        llm = LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
    else:
        llm = ctx.llm
    d = ctx.tmp / "stm"
    d.mkdir()
    stm = ShortTermMemory(llm, path=d, max_tokens=1500, trigger_ratio=0.8, target_ratio=0.5)
    epi = ctx.new_tree("stm_epi")
    needle = "Mi número de socio del club es 48213-K y vence en marzo."
    msgs = []
    for i in range(40):
        msgs.append(needle if i == 4 else f"Mensaje {i}: seguimiento del proyecto Atlas con métricas {i * 37} y notas de revisión semanal.")
    t = time.time() - 4000
    for i, m in enumerate(msgs):
        role = "user" if i % 2 == 0 else "assistant"
        stm.add(role, m, ts=t + i)
        epi.log_episode(role, m, ts=t + i)
    sync_all(epi)

    hit = stm.recall("¿Cuál es mi número de socio del club?")
    found_pre = any("48213" in h["text"] for h in hit["hits"]) or any("48213" in e["text"] for e in hit["recent"])
    R.add(S, "Antes de compactar, BM25 local localiza un dato antiguo", found_pre, "sí" if found_pre else "no", "sí",
          "medio", "corto_plazo.py · recall/_bm25")

    before = stm.total_tokens()
    last_before = [e["text"] for e in stm.snapshot() if e["role"] != "summary"][-stm.keep_recent:]
    reps = stm.compact(force=True)
    after = stm.total_tokens()
    snap = stm.snapshot()
    last_after = [e["text"] for e in snap if e["role"] != "summary"][-stm.keep_recent:]
    R.add(S, "La compactación reduce los tokens ≥ 30%", after <= before * 0.7, f"{before} → {after}", "≤ 70%", "alto",
          "corto_plazo.py · compact/_summarize_chunk", evidence=reps[:2])
    R.add(S, "Los mensajes recientes quedan intactos", last_before == last_after, "iguales" if last_before == last_after else "distintos",
          "iguales", "alto", "corto_plazo.py · _pick_chunk")
    arch = d / "_archivo.jsonl"
    n_arch = len(arch.read_text(encoding="utf-8").splitlines()) if arch.exists() else 0
    has_needle = arch.exists() and "48213" in arch.read_text(encoding="utf-8")
    R.add(S, "Los originales resumidos se conservan (auditoría sin pérdida)", has_needle and n_arch > 0,
          f"{n_arch} líneas", "contiene el dato", "medio", "corto_plazo.py · _archive")

    in_summary = any("48213" in e["text"] for e in snap if e["role"] == "summary")
    ep = epi.search_episodes("¿Cuál es mi número de socio del club?", 5)
    in_epi = any("48213" in e["text"] for e in ep[:3])
    R.add(S, "Dato antiguo recuperable tras compactar (episódica top-3)", in_epi, "sí" if in_epi else "no", "sí", "alto",
          "episodica.py · search", "La memoria episódica es la red de seguridad cuando el resumen pierde detalles.")
    R.add(S, "El resumen LLM conserva el dato literal", in_summary, "sí" if in_summary else "no", "(observación)", "info",
          "corto_plazo.py · SUMMARY_SYSTEM", "Si no lo conserva, depende de la memoria episódica.")

    stm2 = ShortTermMemory(llm, path=d)
    R.add(S, "Persistencia: la sesión se recarga idéntica", len(stm2.snapshot()) == len(snap), f"{len(stm2.snapshot())}/{len(snap)}",
          "iguales", "alto", "corto_plazo.py · _load/_save")
    bad = ctx.tmp / "stm_bad"
    bad.mkdir()
    (bad / "sesion.json").write_text("{esto no es json", encoding="utf-8")
    try:
        ShortTermMemory(llm, path=bad)
        survived = True
    except Exception:
        survived = False
    kept = any(p.name.startswith("sesion.json.corrupt") for p in bad.iterdir())
    R.add(S, "Archivo de sesión corrupto: no crashea y se conserva", survived and kept, "sí" if survived and kept else "no",
          "sí", "medio", "corto_plazo.py · _load")
    stm.stop()


# ---------------------------------------------------------------- cartuchos
def suite_cartuchos(ctx: Ctx) -> None:
    R, S = ctx.report, "cartuchos"
    saved = (C.BODY, C.LTM_DIR, C.STM_DIR)
    C.BODY, C.LTM_DIR, C.STM_DIR = ctx.tmp / "body", ctx.tmp / "body" / "largo_plazo", ctx.tmp / "body" / "corto_plazo"
    try:
        mgr = CartridgeManager(root=ctx.tmp / "cartuchos")
        c1 = mgr.create_cartridge("Biología", "Datos biológicos")
        c2 = mgr.create_cartridge("Astrofísica", "Datos espaciales")
        t1 = MemoryTree(mgr.get_cartridge_path(c1["id"]), ctx.llm)
        t2 = MemoryTree(mgr.get_cartridge_path(c2["id"]), ctx.llm)
        ctx.trees += [t1, t2]
        for t, txt, dom in ((t1, "Las mitocondrias generan trifosfato de adenosina en la célula.", "Biología"),
                            (t1, "Los ribosomas sintetizan proteínas a partir del ARN mensajero.", "Biología"),
                            (t2, "Las estrellas de neutrones tienen densidades nucleares extremas.", "Astronomía")):
            ctx.register(txt, dom)
            t.store(txt, 8)
        leak = len(t1._exact_search("neutrones")) + len(t2._exact_search("mitocondrias"))
        R.add(S, "Aislamiento entre cartuchos", leak == 0, f"{leak} fugas", "0", "crítico", "cartuchos.py")

        bad_ids = ["../../etc", "..", "a/b", "", "x" * 80]
        escaped = []
        for b in bad_ids:
            try:
                mgr.get_cartridge_path(b)
                escaped.append(b)
            except ValueError:
                pass
        R.add(S, "Identificadores con path traversal rechazados", not escaped, f"{len(escaped)} aceptados", "0", "crítico",
              "cartuchos.py · _dir", evidence=[f"aceptado: {e!r}" for e in escaped])
        weird = mgr.create_cartridge("../../Evil/../Nombre", "x")
        inside = (mgr.root / weird["id"]).resolve().parent == mgr.root.resolve()
        R.add(S, "Nombres hostiles se normalizan dentro de la raíz", inside, weird["id"], "dentro", "alto", "cartuchos.py · slugify")

        r1 = mgr.delete_cartridge(mgr.get_primary_id())
        try:
            mgr.unmount(mgr.get_primary_id())
            unm = False
        except ValueError:
            unm = True
        R.add(S, "No se puede borrar ni desconectar el primario", (not r1.get("ok")) and unm, "protegido" if unm else "NO", "protegido",
              "alto", "cartuchos.py · delete/unmount")

        t1.close()
        arc = mgr.export_cartridge_tar(c1["id"])
        res = mgr.import_cartridge_tar(arc, "biologia_copia")
        n_imp = -1
        if res.get("ok"):
            ti = MemoryTree(mgr.get_cartridge_path(res["cartridge"]["id"]), ctx.llm)
            ctx.trees.append(ti)
            n_imp = ti.counts()["alive"]
        R.add(S, "Exportar → importar conserva el contenido", n_imp == 2, f"{n_imp}/2", "iguales", "alto", "cartuchos.py")

        evil_name = f"hmem_evil_{os.getpid()}.txt"
        evil_path = Path(tempfile.gettempdir()) / evil_name
        evil_path.unlink(missing_ok=True)
        zip_slip = ctx.tmp / "zipslip.cartridge"
        with tarfile.open(zip_slip, "w:gz") as tar:
            info = tarfile.TarInfo("../" + evil_name)
            payload = b"pwned"
            info.size = len(payload)
            import io
            tar.addfile(info, io.BytesIO(payload))
            tar.add(mgr.get_cartridge_path(c2["id"]) / "memoria.db", arcname="memoria.db")
        rs = mgr.import_cartridge_tar(zip_slip, "zipslip")
        wrote = evil_path.exists()
        evil_path.unlink(missing_ok=True)
        R.add(S, "Tar con '../' (zip-slip) rechazado y sin escribir fuera", (not rs.get("ok")) and not wrote,
              "bloqueado" if not wrote else "ESCRIBIÓ", "bloqueado", "crítico", "cartuchos.py · import_cartridge_tar")
        garbage = mgr.import_cartridge_b64("basura.cartridge", "AAAA" * 50)
        R.add(S, "Archivo basura rechazado limpiamente", not garbage.get("ok"), "rechazado" if not garbage.get("ok") else "aceptado",
              "rechazado", "medio", "cartuchos.py · import_cartridge_b64")
    finally:
        C.BODY, C.LTM_DIR, C.STM_DIR = saved


# ---------------------------------------------------------------- seguridad web
class _StubAgent:
    cartridge_mgr = None

    def status(self) -> dict:
        return {"ok": True}

    def chat(self, text: str) -> dict:
        return {"answer": "ok"}


def suite_seguridad(ctx: Ctx) -> None:
    R, S = ctx.report, "seguridad"
    try:
        from interfaz import frontend
    except Exception as exc:
        R.add(S, "Se puede importar interfaz.frontend", False, type(exc).__name__, "importable", "alto", "interfaz/", str(exc)[:160])
        return
    srv = frontend.ThreadedHTTPServer(("127.0.0.1", 0), frontend.Handler)
    srv.agent = _StubAgent()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def req(method: str, path: str, host: str | None = None, origin: str | None = None,
            content_length: int | None = None, body: bytes = b"") -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", host or f"127.0.0.1:{port}")
        if origin is not None:
            conn.putheader("Origin", origin)
        if method == "POST":
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(content_length if content_length is not None else len(body)))
        conn.endheaders(body if content_length is None else b"")
        resp = conn.getresponse()
        resp.read()
        out = (resp.status, {k.lower(): v for k, v in resp.getheaders()})
        conn.close()
        return out

    try:
        cases = [
            ("GET con Host/Origin locales → 200", lambda: req("GET", "/api/status")[0], 200),
            ("Host externo (DNS rebinding) → 403", lambda: req("GET", "/api/status", host="evil.com")[0], 403),
            ("Host con sufijo '127.0.0.1.evil.com' → 403", lambda: req("GET", "/api/status", host="127.0.0.1.evil.com")[0], 403),
            ("Origin externo en GET → 403", lambda: req("GET", "/api/status", origin="http://evil.com")[0], 403),
            ("Origin externo en POST (CSRF) → 403", lambda: req("POST", "/api/chat", origin="http://evil.com", body=b"{}")[0], 403),
            ("Origin 'null' → 403", lambda: req("POST", "/api/chat", origin="null", body=b"{}")[0], 403),
            ("POST con Origin local → 200", lambda: req("POST", "/api/chat", origin=f"http://127.0.0.1:{port}", body=b'{"text":"hola"}')[0], 200),
            ("Cuerpo de 30 MB rechazado → 400", lambda: req("POST", "/api/chat", content_length=30 * 1024 * 1024)[0], 400),
        ]
        bad = []
        for name, fn, expect in cases:
            try:
                got = fn()
            except Exception as exc:
                got = f"{type(exc).__name__}"
            ok = got == expect
            if not ok:
                bad.append(f"{name}: obtenido {got}")
            R.add(S, name, ok, str(got), str(expect), "crítico" if expect in (403, 400) else "alto", "interfaz/frontend.py · _origin_ok/_read_json")
        st, hdr = req("GET", "/")
        csp = hdr.get("content-security-policy", "")
        R.add(S, "Cabeceras de seguridad (CSP, X-Frame-Options, nosniff)",
              "default-src 'self'" in csp and hdr.get("x-frame-options") == "DENY" and hdr.get("x-content-type-options") == "nosniff",
              "completas" if csp else "faltan", "presentes", "medio", "interfaz/frontend.py",
              "script-src 'unsafe-inline' debilita la CSP: mueve el JS a un archivo propio con nonce/hash.")
        R.add(S, "CSP sin 'unsafe-inline' en script-src", "script-src 'self' 'unsafe-inline'" not in csp,
              "tiene unsafe-inline" if "unsafe-inline" in csp else "estricta", "estricta", "info", "interfaz/frontend.py + index.html")
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------- ingesta
def suite_ingesta(ctx: Ctx) -> None:
    R, S = ctx.report, "ingesta"
    tree = ctx.new_tree("ingesta")
    tree._ensure_ingest_worker = lambda: None   # solo medimos el filtro, sin procesar la cola
    rej = [not tree.ingest_async(t, "mensaje del usuario") for t in NONFACTUAL_INPUTS]
    acc = [bool(tree.ingest_async(t, "mensaje del usuario")) for t in FACTUAL_INPUTS]
    rr, ar = sum(rej) / len(rej), sum(acc) / len(acc)
    R.add(S, "Los hechos personales pasan el filtro", ar >= 0.95, pct(ar), "≥ 95%", "alto", "ltm/ingestion.py · ingest_async",
          "Perder un hecho aquí es perderlo para siempre.", [f"descartado: «{t}»" for t, a in zip(FACTUAL_INPUTS, acc) if not a])
    R.add(S, "Preguntas, órdenes y saludos se filtran", rr >= 0.70, pct(rr), "≥ 70%", "medio", "ltm/common.py · _ASK_RE/_is_nonfactual",
          "Cada texto no factual que pasa cuesta una llamada LLM; amplía las formas de petición (necesito, podrías, quiero saber…).",
          [f"pasa indebidamente: «{t}»" for t, r in zip(NONFACTUAL_INPUTS, rej) if not r])
    if ctx.live:
        text = "Hola. Mi hija mayor se llama Alicia, vivo en Zamora y soy alérgico a la penicilina. ¿Qué tiempo hará mañana?"
        _src, facts = tree.extract_facts(text, first=False, origin="mensaje del usuario")
        blob = fold(" ".join(f["texto"] for f in (facts or [])))
        found = sum(k in blob for k in ("alicia", "zamora", "penicilina"))
        R.add(S, "Extracción LLM captura los 3 hechos personales", found == 3, f"{found}/3", "3/3", "alto", "ltm/ingestion.py · extract_facts")
        R.add(S, "La extracción no memoriza preguntas", not any("?" in f["texto"] for f in (facts or [])), "limpia", "sin '?'", "medio", "ltm/ingestion.py")
    else:
        R.add(S, "Calidad de extracción LLM", True, "solo --live", "—", "info")


# ---------------------------------------------------------------- síntesis (solo live)
def suite_sintesis(ctx: Ctx) -> None:
    R, S = ctx.report, "sintesis"
    if not ctx.live:
        ctx.report.suite_skipped[S] = "requiere --live (LLM real)"
        return
    corp = ctx.corpus()
    sample = [(q, fi) for q, fi, t, _d in corp.qs if t in ("literal", "paráfrasis")][::3][:24]
    good, fails = 0, []
    for q, fi in sample:
        res = corp.tree.recall(q, reinforce=False, synthesize=True)
        clave = fold(FACTS[fi][2])
        ans = fold(res.get("answer", ""))
        if res.get("found") and clave in ans:
            good += 1
        else:
            fails.append(f"«{q[:60]}» faltó «{FACTS[fi][2]}» en «{res.get('answer', '')[:60]}»")
    acc = good / max(1, len(sample))
    R.add(S, "Respuesta sintetizada contiene el dato clave", acc >= 0.85, pct(acc), "≥ 85%", "alto", "ltm/retrieval.py · prompt de síntesis", evidence=fails)


# ---------------------------------------------------------------- corpus difícil  [PARCHE3]
# Hechos densos del MISMO dominio con vocabulario compartido: reproduce el caso real en el que el hecho
# correcto está entre los candidatos pero pierde el top-1 frente a vecinos casi idénticos.
# grupo -> (dominio, [(hecho, (pregunta parafraseada, pregunta indirecta))])
HARD_CLUSTERS: dict[str, tuple[str, list[tuple[str, tuple[str, str]]]]] = {
    "Riñón": ("Fisiología renal", [
        ("El glomérulo es un ovillo de capilares en el que el plasma se filtra hacia la cápsula de Bowman.",
         ("¿Qué estructura formada por capilares filtra el plasma hacia la cápsula de Bowman?",
          "¿Dónde ocurre el primer paso de la formación de la orina?")),
        ("El túbulo contorneado proximal reabsorbe alrededor del 65 % del sodio y del agua filtrados.",
         ("¿Qué segmento de la nefrona recupera cerca de dos tercios del sodio filtrado?",
          "¿Dónde se reabsorbe la mayor parte del agua del filtrado?")),
        ("El túbulo proximal reabsorbe la totalidad de la glucosa filtrada mediante cotransportadores SGLT2.",
         ("¿Qué transportadores recuperan la glucosa del filtrado renal?",
          "¿Por qué en condiciones normales no aparece azúcar en la orina?")),
        ("La rama descendente del asa de Henle es permeable al agua pero no al sodio.",
         ("¿Qué tramo del asa de Henle deja pasar agua pero no sodio?",
          "¿Dónde pierde agua el filtrado al bajar hacia la médula hipertónica?")),
        ("La rama ascendente gruesa del asa de Henle es impermeable al agua y reabsorbe sodio con el cotransportador NKCC2.",
         ("¿Qué segmento renal es impermeable al agua y reabsorbe sodio con NKCC2?",
          "¿Qué tramo de la nefrona se llama segmento diluyente porque saca sal sin agua?")),
        ("La mácula densa detecta la concentración de cloruro sódico en el túbulo distal y regula la filtración glomerular.",
         ("¿Qué células del túbulo distal miden el cloruro sódico que llega?",
          "¿Qué sensor renal participa en la retroalimentación tubuloglomerular?")),
        ("Las células yuxtaglomerulares de la arteriola aferente secretan renina cuando baja la presión de perfusión.",
         ("¿Qué células secretan renina?",
          "¿Qué enzima libera el riñón cuando cae la presión con la que le llega la sangre?")),
        ("La angiotensina II contrae la arteriola eferente y mantiene la filtración glomerular cuando cae la presión.",
         ("¿Qué hormona contrae la arteriola eferente?",
          "¿Qué mecanismo hormonal sostiene el filtrado cuando la presión de perfusión es baja?")),
        ("La aldosterona actúa en el túbulo colector aumentando la reabsorción de sodio y la secreción de potasio.",
         ("¿Qué hormona hace que el túbulo colector retenga sodio y elimine potasio?",
          "¿Por qué un exceso de mineralocorticoide provoca hipopotasemia?")),
        ("La hormona antidiurética inserta acuaporinas 2 en el túbulo colector y aumenta la reabsorción de agua.",
         ("¿Qué hormona inserta canales de agua en el túbulo colector?",
          "¿Qué hormona falta en la diabetes insípida central?")),
        ("La eritropoyetina se produce en las células intersticiales peritubulares de la corteza renal ante la hipoxia.",
         ("¿Qué hormona fabrica el riñón cuando falta oxígeno?",
          "¿Por qué los pacientes con insuficiencia renal crónica sufren anemia?")),
        ("El riñón convierte el calcidiol en calcitriol mediante la enzima 1-alfa-hidroxilasa.",
         ("¿Qué enzima renal activa la vitamina D?",
          "¿Por qué la enfermedad renal avanzada causa déficit de vitamina D activa?")),
        ("La tasa de filtración glomerular normal de un adulto es de unos 125 mililitros por minuto.",
         ("¿Cuántos mililitros por minuto filtran normalmente los glomérulos de un adulto?",
          "¿Qué valor de filtrado se considera normal en un riñón sano?")),
        ("La creatinina se usa para estimar la filtración glomerular porque se filtra libremente y apenas se secreta.",
         ("¿Qué sustancia de la sangre se usa para estimar el filtrado glomerular?",
          "¿Por qué sube en sangre un producto del músculo cuando fallan los riñones?")),
        ("Los podocitos rodean los capilares glomerulares y sus diafragmas de hendidura impiden el paso de proteínas.",
         ("¿Qué células impiden que las proteínas pasen al filtrado?",
          "¿Qué estructura se daña cuando aparece proteinuria masiva?")),
        ("La autorregulación renal mantiene constante el flujo sanguíneo entre 80 y 180 mmHg de presión arterial.",
         ("¿Entre qué presiones mantiene el riñón constante su flujo sanguíneo?",
          "¿Por qué el filtrado no cambia con subidas moderadas de tensión?")),
        ("El riñón regula el pH secretando hidrogeniones y regenerando bicarbonato en el túbulo.",
         ("¿Cómo compensa el riñón una acidosis?",
          "¿Qué órgano regenera el bicarbonato de la sangre?")),
        ("El amonio excretado por el túbulo proximal permite eliminar ácido sin acidificar en exceso la orina.",
         ("¿Qué compuesto nitrogenado permite al riñón excretar ácido?",
          "¿Qué tampón urinario aumenta más en una acidosis crónica?")),
        ("La urea se recicla en la médula interna y contribuye a la hipertonicidad del intersticio medular.",
         ("¿Qué soluto se recicla en la médula interna?",
          "¿Qué ayuda, además del sodio, a que la médula renal sea muy concentrada?")),
        ("Los vasos rectos son capilares en horquilla que irrigan la médula sin lavar su gradiente osmótico.",
         ("¿Qué capilares irrigan la médula sin disipar el gradiente osmótico?",
          "¿Por qué la sangre de la médula no arrastra la sal acumulada?")),
    ]),
    "Fármacos": ("Farmacología", [
        ("La furosemida es un diurético de asa que inhibe el cotransportador NKCC2.",
         ("¿Qué cotransportador inhibe la furosemida?",
          "¿Qué diurético potente se usa en el edema agudo de pulmón?")),
        ("La hidroclorotiazida inhibe el cotransportador de sodio y cloro del túbulo distal.",
         ("¿Qué diurético actúa sobre el cotransportador sodio-cloro del túbulo distal?",
          "¿Qué diurético tiazídico se usa de primera línea en la hipertensión?")),
        ("La espironolactona es un diurético ahorrador de potasio que bloquea el receptor de aldosterona.",
         ("¿Qué fármaco bloquea el receptor de la aldosterona?",
          "¿Qué diurético no hace perder potasio?")),
        ("El enalapril inhibe la enzima convertidora de angiotensina y puede causar tos seca.",
         ("¿Qué fármaco inhibe la enzima convertidora de angiotensina?",
          "¿Qué antihipertensivo provoca a veces una tos seca persistente?")),
        ("El losartán bloquea el receptor AT1 de la angiotensina II y no produce tos.",
         ("¿Qué fármaco bloquea el receptor AT1?",
          "¿A qué antihipertensivo se cambia a un paciente al que un IECA le da tos?")),
        ("El amlodipino es un calcioantagonista que puede causar edema en los tobillos.",
         ("¿Qué calcioantagonista se asocia a hinchazón de tobillos?",
          "¿Qué antihipertensivo bloquea los canales de calcio vasculares?")),
        ("El bisoprolol es un betabloqueante cardioselectivo que reduce la frecuencia cardíaca.",
         ("¿Qué betabloqueante es cardioselectivo?",
          "¿Qué fármaco baja las pulsaciones bloqueando receptores beta1?")),
        ("La empagliflozina inhibe el SGLT2 y aumenta la eliminación de glucosa por la orina.",
         ("¿Qué antidiabético inhibe el SGLT2?",
          "¿Qué medicamento hace orinar azúcar a propósito para bajar la glucemia?")),
        ("El paracetamol en sobredosis causa necrosis hepática y su antídoto es la N-acetilcisteína.",
         ("¿Cuál es el antídoto de la intoxicación por paracetamol?",
          "¿Qué analgésico común destruye el hígado en sobredosis?")),
        ("La warfarina antagoniza la vitamina K y se controla con el INR.",
         ("¿Qué anticoagulante se controla midiendo el INR?",
          "¿Qué fármaco se contrarresta administrando vitamina K?")),
        ("La heparina potencia la antitrombina y su antídoto es el sulfato de protamina.",
         ("¿Qué anticoagulante se revierte con protamina?",
          "¿Qué fármaco actúa potenciando la antitrombina?")),
        ("La digoxina inhibe la bomba sodio-potasio ATPasa y su toxicidad aumenta con la hipopotasemia.",
         ("¿Qué fármaco cardíaco inhibe la bomba sodio-potasio?",
          "¿Por qué es peligroso un potasio bajo en un paciente que toma un glucósido cardíaco?")),
        ("El omeprazol inhibe la bomba de protones de las células parietales del estómago.",
         ("¿Qué fármaco inhibe la bomba de protones gástrica?",
          "¿Qué medicamento reduce la producción de ácido en el estómago?")),
        ("La levotiroxina es la hormona tiroidea sintética usada en el hipotiroidismo.",
         ("¿Con qué fármaco se trata el hipotiroidismo?",
          "¿Qué pastilla sustituye la hormona que no fabrica una tiroides perezosa?")),
        ("La amoxicilina es una penicilina que inhibe la síntesis de la pared bacteriana.",
         ("¿Cómo actúa la amoxicilina sobre las bacterias?",
          "¿Qué antibiótico betalactámico se usa mucho en la otitis infantil?")),
        ("La azitromicina es un macrólido que inhibe la subunidad 50S del ribosoma bacteriano.",
         ("¿Qué antibiótico macrólido inhibe la subunidad 50S?",
          "¿Qué antibiótico no betalactámico puede tomar alguien alérgico a la penicilina para una infección respiratoria?")),
    ]),
    "Usuario": ("Datos personales", [
        ("La hija mayor del usuario se llama Alicia y nació en Valladolid en 2012.",
         ("¿Cómo se llama la primogénita del usuario?",
          "¿En qué ciudad nació la niña mayor del usuario?")),
        ("El hijo pequeño del usuario se llama Daniel y tiene seis años.",
         ("¿Cómo se llama el hijo menor del usuario?",
          "¿Qué edad tiene el benjamín de la familia del usuario?")),
        ("La esposa del usuario se llama Lucía y es enfermera en el hospital de León.",
         ("¿Cómo se llama la mujer del usuario?",
          "¿Quién en la familia del usuario trabaja en un hospital?")),
        ("El padre del usuario se llama Andrés y fue maquinista de tren.",
         ("¿A qué se dedicaba el padre del usuario?",
          "¿Quién conducía trenes en la familia del usuario?")),
        ("La madre del usuario se llama Carmen y vive en Astorga.",
         ("¿Dónde vive la madre del usuario?",
          "¿Cómo se llama la madre del usuario?")),
        ("El hermano del usuario se llama Rubén y vive en Bilbao.",
         ("¿Cómo se llama el hermano del usuario?",
          "¿Qué familiar del usuario reside en el País Vasco?")),
        ("El usuario es alérgico a la penicilina y a la amoxicilina.",
         ("¿Qué antibióticos no puede tomar el usuario?",
          "¿Qué debe evitar el médico al recetarle un antibiótico al usuario?")),
        ("El usuario es intolerante a la lactosa.",
         ("¿Qué azúcar de la leche no tolera el usuario?",
          "¿Por qué al usuario le sientan mal los lácteos?")),
        ("El usuario tiene hipertensión y toma enalapril cada mañana.",
         ("¿Qué medicación diaria toma el usuario?",
          "¿Qué problema de tensión tiene el usuario?")),
        ("El perro del usuario es un labrador llamado Thor que tiene cuatro años.",
         ("¿Cómo se llama el perro del usuario?",
          "¿De qué raza es la mascota del usuario?")),
        ("El usuario trabaja como ingeniero de datos en una empresa de Zamora.",
         ("¿En qué trabaja el usuario?",
          "¿En qué ciudad está la empresa del usuario?")),
        ("El usuario vive en León, en el barrio de Eras de Renueva.",
         ("¿En qué barrio vive el usuario?",
          "¿Dónde está la casa del usuario?")),
        ("El usuario prefiere recibir las respuestas en español y con ejemplos de código en Python.",
         ("¿Qué lenguaje de programación quiere el usuario en los ejemplos?",
          "¿En qué idioma debe contestarle el asistente al usuario?")),
        ("El usuario corre una media maratón cada primavera.",
         ("¿Qué carrera hace el usuario cada año?",
          "¿Qué deporte practica el usuario?")),
        ("El usuario toca la guitarra clásica desde los quince años.",
         ("¿Qué instrumento toca el usuario?",
          "¿Desde qué edad hace música el usuario?")),
        ("El coche del usuario es un Toyota Corolla híbrido de color gris.",
         ("¿Qué coche tiene el usuario?",
          "¿De qué color es el vehículo del usuario?")),
        ("El cumpleaños del usuario es el 14 de abril.",
         ("¿Cuándo cumple años el usuario?",
          "¿Qué día hay que felicitar al usuario?")),
        ("El usuario estudió Ingeniería Informática en la Universidad de León.",
         ("¿Qué carrera estudió el usuario?",
          "¿En qué universidad se formó el usuario?")),
        ("El usuario tiene una reunión con su jefe todos los lunes a las nueve.",
         ("¿Cuándo se reúne el usuario con su jefe?",
          "¿Qué compromiso fijo tiene el usuario al empezar la semana?")),
        ("Al usuario no le gusta el cilantro.",
         ("¿Qué hierba aromática no le gusta al usuario?",
          "¿Qué ingrediente hay que quitar de las recetas para el usuario?")),
    ]),
}

# Negativas CERCANAS: comparten tema y vocabulario con hechos del corpus, pero no tienen respuesta en él.
HARD_NEGATIVES = [
    "¿Cómo se llama el gato del usuario?",
    "¿Cómo se llama la hija pequeña del usuario?",
    "¿A qué se dedica la madre del usuario?",
    "¿Dónde vive la hermana del usuario?",
    "¿Qué medicamento toma el usuario para el colesterol?",
    "¿A qué es alérgica la esposa del usuario?",
    "¿Qué marca de moto tiene el usuario?",
    "¿Qué instrumento toca la hija del usuario?",
    "¿Cuándo es el cumpleaños de la esposa del usuario?",
    "¿En qué universidad estudió la esposa del usuario?",
    "¿Cómo se llama el abuelo del usuario?",
    "¿Qué hormona secreta el glomérulo?",
    "¿Cuántas nefronas tiene cada riñón?",
    "¿Qué porcentaje del sodio filtrado reabsorbe el túbulo colector?",
    "¿Qué antídoto revierte la intoxicación por digoxina?",
    "¿Qué diurético inhibe la anhidrasa carbónica?",
    "¿Con qué fármaco se trata el hipertiroidismo?",
    "¿Qué antibiótico inhibe la subunidad 30S del ribosoma?",
    "¿Qué efecto adverso tiene la espironolactona en las mamas?",
    "¿Cada cuántas horas se administra la heparina de bajo peso molecular?",
    # [PARCHE5] +40 negativas cercanas (20 personales · 10 riñón · 10 fármacos): con 20, 1 caso = 5 pp
    "¿Cómo se llama la suegra del usuario?",
    "¿Qué edad tiene la esposa del usuario?",
    "¿Dónde trabaja el hermano del usuario?",
    "¿Cuál es el color favorito del usuario?",
    "¿De qué raza es el gato de la madre del usuario?",
    "¿Cómo se llama el segundo perro del usuario?",
    "¿En qué colegio estudia Alicia?",
    "¿Qué deporte practica la esposa del usuario?",
    "¿A qué es alérgico el hijo pequeño del usuario?",
    "¿Qué coche conduce la esposa del usuario?",
    "¿Qué instrumento tocaba el padre del usuario?",
    "¿En qué ciudad nació el usuario?",
    "¿Qué idioma estudia la hija mayor del usuario?",
    "¿Cuántos años lleva el usuario en su empresa actual?",
    "¿Qué medicación toma la madre del usuario?",
    "¿Cómo se llama el jefe del usuario?",
    "¿En qué barrio vive la madre del usuario?",
    "¿Qué marca de portátil usa el usuario?",
    "¿Qué día es el cumpleaños del hijo pequeño del usuario?",
    "¿Qué grupo sanguíneo tiene el usuario?",
    "¿Cuál es el flujo sanguíneo renal normal en mililitros por minuto?",
    "¿Qué porcentaje del calcio filtrado se reabsorbe en el túbulo distal?",
    "¿Qué transportador absorbe la glucosa en el intestino delgado?",
    "¿Qué presión hidrostática hay en los capilares glomerulares?",
    "¿Cuántos litros de orina produce un adulto al día?",
    "¿Qué segmento de la nefrona reabsorbe la mayor parte del magnesio?",
    "¿Cuál es el pH normal de la orina?",
    "¿Cuánto filtra el glomérulo de un recién nacido?",
    "¿Qué arteria lleva la sangre al riñón?",
    "¿Qué hormona aumenta la reabsorción de calcio en el túbulo distal?",
    "¿Cuál es la dosis máxima diaria de paracetamol en un adulto?",
    "¿Qué efecto adverso produce la amiodarona en el tiroides?",
    "¿Qué antibiótico se usa contra Pseudomonas?",
    "¿Qué estatina se usa para bajar el colesterol?",
    "¿Cada cuánto se toma el bisoprolol?",
    "¿Qué diurético se usa en el glaucoma?",
    "¿Qué antagonista de la aldosterona tiene menos efectos hormonales que la espironolactona?",
    "¿Qué antihipertensivo está contraindicado en el embarazo?",
    "¿Qué antibiótico es un aminoglucósido?",
    "¿Qué antidiabético estimula la secreción de insulina por el páncreas?",
]

HARD_CONFIGS = ["actual (.env)", "completo", "solo embeddings", "emb + léxico", "completo − árbol",
                "completo − expansión LLM"]


def suite_dificil(ctx: Ctx) -> None:
    R, S = ctx.report, "dificil"
    tree = ctx.new_tree("dificil")
    id_of: dict[int, str] = {}
    qs: list[tuple[str, int, str, str]] = []
    noise = [(d, t) for d, t, _k, _q in FACTS if d not in ("Fisiología", "Salud", "Datos personales")] + FILLERS
    n_hard = sum(len(items) for _d, items in HARD_CLUSTERS.values())
    print(f"   … cargando corpus denso ({n_hard} hechos en {len(HARD_CLUSTERS)} grupos + {len(noise)} de ruido)",
          end="", flush=True)
    t0 = time.time()
    fi = 0
    for cluster, (dom, items) in HARD_CLUSTERS.items():
        for fact, (qp, qi) in items:
            ctx.register(fact, dom)
            id_of[fi] = tree.store(fact, 7, "dato", meta={"fuente": "benchmark-dificil"})["id"]
            qs += [(qp, fi, "paráfrasis", cluster), (qi, fi, "indirecta", cluster)]
            fi += 1
            print(".", end="", flush=True)
    for dom, text in noise:
        ctx.register(text, dom)
        tree.store(text, 6, "dato", meta={"fuente": "benchmark"})
        print(".", end="", flush=True)
    sync_all(tree)
    print(f" listo ({time.time() - t0:.0f}s)")

    # --- 1) integridad al cargar: ningún hecho de este corpus sustituye a otro
    sup = dbq(tree, "SELECT c.text AS text, n.text AS por FROM cells c LEFT JOIN cells n ON n.id = c.superseded_by "
                    "WHERE c.superseded_by IS NOT NULL")
    R.add(S, "Falsas invalidaciones al cargar el corpus denso", not sup, f"{len(sup)} hecho(s)", "0", "alto",
          "ltm/storage.py · _detect_temporal_supersession",
          "Ningún hecho de este corpus sustituye a otro (la madre vive en Astorga, el hermano en Bilbao): "
          "toda invalidación es pérdida silenciosa de información.",
          [f"«{r['text'][:55]}» invalidado por «{(r['por'] or '?')[:55]}»" for r in sup][:6])
    dup = n_hard - len(set(id_of.values()))
    if dup:
        print(f"   ⚠ {dup} hecho(s) del corpus se fusionaron como duplicados al guardarse")

    # --- 2) ranking en varias configuraciones
    abl = {name: (over, nav) for name, over, nav, _b in ABLATIONS}
    ok1 = lambda r: r["rank"] == 1
    ok3 = lambda r: bool(r["rank"] and r["rank"] <= 3)
    res: dict[str, list[dict[str, Any]]] = {}
    for name in HARD_CONFIGS:
        over, nav = abl[name]
        with _cfg_override(**over):
            res[name] = _evaluate(tree, qs, id_of, navigate=nav)
    n = len(qs)
    groups = list(HARD_CLUSTERS)
    base = res["actual (.env)"]
    print(f"\n   RANKING EN CORPUS DENSO  (n={n}: {n // 2} paráfrasis + {n // 2} indirectas)")
    print(f"   {'configuración':<26}{'Hit@1':>7}{'Hit@3':>7}{'MRR':>7}{'R@18':>7}"
          + "".join(f"{g[:8] + '@1':>12}" for g in groups) + f"{'P95ms':>7}   vs actual (top-1)")
    for name, rows in res.items():
        h1, h3 = sum(map(ok1, rows)) / n, sum(map(ok3, rows)) / n
        r18 = sum(1 for r in rows if r["rank"]) / n
        per = ""
        for g in groups:
            sel = [r for r in rows if r["dom"] == g]
            per += f"{pct(sum(map(ok1, sel)) / max(1, len(sel))):>12}"
        if rows is base:
            cmp = "—"
        else:
            gana = sum(ok1(y) and not ok1(x) for x, y in zip(base, rows))
            pierde = sum(ok1(x) and not ok1(y) for x, y in zip(base, rows))
            cmp = f"+{gana}/−{pierde} p={_sign_p(pierde, gana):.2f}"
        print(f"   {name:<26}{pct(h1):>7}{pct(h3):>7}{_mrr(rows):>7.3f}{pct(r18):>7}{per}"
              f"{percentile([r['ms'] for r in rows], 95):>7.0f}   {cmp}")
        R.add(S, f"Hit@1 · {name}", True, pct(h1), "(observación)", "info", "ltm/retrieval.py · recall")
        R.add(S, f"Hit@3/MRR · {name}", True, f"{pct(h3)} / {_mrr(rows):.3f}", "(observación)", "info",
              "ltm/retrieval.py · recall")
    by_t = {t: [r for r in base if r["tipo"] == t] for t in ("paráfrasis", "indirecta")}
    print("   actual (.env) por tipo: " + " · ".join(
        f"{t} Hit@1 {pct(sum(map(ok1, v)) / max(1, len(v)))} Hit@3 {pct(sum(map(ok3, v)) / max(1, len(v)))}"
        for t, v in by_t.items()))

    bad = [r for r in base if not ok1(r)]
    print(f"\n   FALLOS DE TOP-1 con 'actual (.env)': {len(bad)}/{n}  (esperado → lo que quedó primero)")
    tt = cell_texts(tree, [id_of[r["fi"]] for r in bad] + [r["top1"] for r in bad if r["top1"]])
    for r in bad[:14]:
        print(f"     [{r['dom'][:5]}|{r['tipo'][:4]}] rank={r['rank']} «{r['q'][:62]}»")
        print(f"        esperado: {tt.get(id_of[r['fi']], '')[:72]}")
        print(f"        top-1:    {tt.get(r['top1'], '—')[:72]}")

    # --- 3) compuerta de abstención con la configuración actual
    def gate(q: str) -> str:
        r = tree.recall(q, reinforce=False, synthesize=False)
        return "rechaza" if not r.get("found") else ("tentativa" if r.get("tentative") else "firme")

    pos_g = [(q, fi, gate(q)) for q, fi, _t, _g in qs]
    neg_g = [(q, gate(q)) for q in HARD_NEGATIVES]
    cnt = lambda xs, k: sum(1 for x in xs if x[-1] == k)
    print("\n   COMPUERTA DE ABSTENCIÓN EN CORPUS DENSO  (configuración actual)")
    print(f"   positivas ({len(pos_g)}): firme {cnt(pos_g, 'firme')} · tentativa {cnt(pos_g, 'tentativa')} · "
          f"rechaza {cnt(pos_g, 'rechaza')}")
    print(f"   negativas cercanas ({len(neg_g)}): firme {cnt(neg_g, 'firme')} · tentativa {cnt(neg_g, 'tentativa')} · "
          f"rechaza {cnt(neg_g, 'rechaza')}")
    R.add(S, "Abstención firme en negativas cercanas", True, pct(1 - cnt(neg_g, "firme") / max(1, len(neg_g))),
          "(observación)", "info", "ltm/retrieval.py · _classify_relevance")
    R.add(S, "Positivas que la compuerta rechaza (respuesta perdida)", True,
          f"{cnt(pos_g, 'rechaza')}/{len(pos_g)}", "(observación)", "info", "ltm/retrieval.py · _classify_relevance")
    firm_neg = [q for q, g in neg_g if g == "firme"]
    if firm_neg:
        print("   · negativas cercanas afirmadas con certeza (la compuerta debería dudar):")
        for q in firm_neg[:10]:
            s, cid = _top1_emb(tree, q)
            txt = cell_texts(tree, [cid]).get(cid, "") if cid else ""
            print(f"     sim={(s or 0):.2f} solap={_overlap_terms(tree, q, txt)}  «{q[:46]}» → «{txt[:46]}»")
    lost = [(q, fi) for q, fi, g in pos_g if g == "rechaza"]
    if lost:
        print("   · positivas rechazadas (tenían respuesta):")
        for q, fi in lost[:8]:
            s, _cid = _top1_emb(tree, q)
            gold = cell_texts(tree, [id_of[fi]]).get(id_of[fi], "")
            print(f"     sim={(s or 0):.2f} solap={_overlap_terms(tree, q, gold)}  «{q[:52]}»")
    if not ctx.live:
        print("   ⚠ modo emulado: valida la fontanería; las cifras solo son informativas con --live")


# ---------------------------------------------------------------- reranker + OBJETIVO  [PARCHE4]
# Objetivo del proyecto medido extremo a extremo con el agente real sobre el corpus denso.
NORTH_STAR = {"correctas": 0.95, "inventa": 0.05, "p95_ms": 1500.0}
RR_POS_JUDGE = ("Eres un evaluador estricto. Se hizo una pregunta a un asistente con memoria y existe un DATO "
                "CORRECTO que la responde. Decide si la respuesta del asistente contiene ese dato (aunque use otras "
                "palabras) y no lo contradice. Si dice que no lo sabe o da otro dato, es incorrecta.\n"
                'Devuelve SOLO JSON: {"correcta": true} o {"correcta": false}.')
# [PARCHE5] reglas más precisas: en el parche 4 marcó como invención un «no consta» y un dato médico correcto
RR_NEG_JUDGE = ("Eres un evaluador estricto. Se preguntó a un asistente con memoria algo que NO consta en su memoria. "
                "Decide si la respuesta INVENTA. Inventar es: (a) afirmar como cierto un dato concreto sobre el usuario "
                "o su entorno (familia, salud, gustos, pertenencias, trabajo...) que no consta, o (b) atribuir al sujeto "
                "preguntado un dato que pertenece a otro (otra persona, otro animal, otro objeto).\n"
                "NO es inventar: (c) reconocer que no tiene el dato, aunque después mencione otros datos que sí constan "
                "de esa persona; (d) responder una pregunta de conocimiento general (medicina, ciencia...) con "
                "información correcta y establecida.\n"
                'Devuelve SOLO JSON: {"inventa": true} o {"inventa": false}.')


def _hard_corpus(ctx: Ctx) -> tuple[Any, dict[int, str], list[tuple[str, int, str, str]]]:
    """Corpus denso de la suite 'dificil' (se carga una vez por ejecución)."""
    hc = getattr(ctx, "_hard", None)
    if hc is not None:
        return hc
    tree = ctx.new_tree("dificil_rr")
    id_of: dict[int, str] = {}
    qs: list[tuple[str, int, str, str]] = []
    noise = [(d, t) for d, t, _k, _q in FACTS if d not in ("Fisiología", "Salud", "Datos personales")] + FILLERS
    n_hard = sum(len(items) for _d, items in HARD_CLUSTERS.values())
    print(f"   … cargando corpus denso ({n_hard} hechos + {len(noise)} de ruido)", end="", flush=True)
    t0 = time.time()
    fi = 0
    for cluster, (dom, items) in HARD_CLUSTERS.items():
        for fact, (qp, qi) in items:
            ctx.register(fact, dom)
            id_of[fi] = tree.store(fact, 7, "dato", meta={"fuente": "benchmark-dificil"})["id"]
            qs += [(qp, fi, "paráfrasis", cluster), (qi, fi, "indirecta", cluster)]
            fi += 1
            print(".", end="", flush=True)
    for dom, text in noise:
        ctx.register(text, dom)
        tree.store(text, 6, "dato", meta={"fuente": "benchmark"})
        print(".", end="", flush=True)
    sync_all(tree)
    print(f" listo ({time.time() - t0:.0f}s)")
    ctx._hard = (tree, id_of, qs)
    return ctx._hard


def _cross_ready(tree: Any) -> str | None:
    try:
        import rerank
        rerank.Reranker(tree)._cross_scores("prueba de conexión", ["prueba de conexión"])
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:90]}"


def suite_reranker(ctx: Ctx) -> None:
    # [PARCHE5] configuraciones = modo × política · 60 negativas · juez por mayoría · e2e en paralelo
    R, S = ctx.report, "reranker"
    tree, id_of, qs = _hard_corpus(ctx)
    pmode = str(getattr(C, "RERANK_MODE", "off")).strip().lower()
    ppol = "abstiene" if str(getattr(C, "RERANK_NONE", "tentative")).strip().lower() == "abstain" else "tentativa"
    prod = f"{pmode}·{ppol}" if pmode in ("llm", "cross") else "off"
    configs: list[tuple[str, dict[str, Any]]] = [
        ("off", {"RERANK_MODE": "off"}),
        ("llm·tentativa", {"RERANK_MODE": "llm", "RERANK_NONE": "tentative"}),
        ("llm·abstiene", {"RERANK_MODE": "llm", "RERANK_NONE": "abstain"}),
    ]
    err = _cross_ready(tree)
    if err is None:
        configs += [("cross·tentativa", {"RERANK_MODE": "cross", "RERANK_NONE": "tentative"}),
                    ("cross·abstiene", {"RERANK_MODE": "cross", "RERANK_NONE": "abstain"})]
    print(f"   producción: RERANK_MODE={pmode} · RERANK_NONE={C.RERANK_NONE} → «{prod}» · RERANK_TOP_K={C.RERANK_TOP_K}")
    if err is not None:
        print(f"   · cross-encoder no disponible en {C.RERANK_URL} → no se mide  ({err})")
    gold_txt = cell_texts(tree, list(id_of.values()))
    chat = None
    if ctx.live:
        from llm import LLM
        from agente import Agent
        chat = LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
    workers = max(1, int(os.getenv("HMEM_E2E_WORKERS", "2")))
    neg_group = lambda q: "personal" if ("usuario" in q or "Alicia" in q) else "conocimiento"

    def state(r: dict) -> str:
        return "rechaza" if not r.get("found") else ("tentativa" if r.get("tentative") else "firme")

    def run(label: str, over: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        pos: list[dict] = []
        neg: list[dict] = []
        print(f"   … {label}: recuperando", end="", flush=True)
        with _cfg_override(**over):
            for q, fi, tipo, grp in qs:
                t0 = time.perf_counter()
                r = tree.recall(q, reinforce=False, synthesize=False)
                ms = (time.perf_counter() - t0) * 1000
                cands = list(r.get("candidates") or [])
                gold = id_of[fi]
                pos.append({"q": q, "fi": fi, "tipo": tipo, "grp": grp, "ms": ms, "r": r,
                            "rank": cands.index(gold) + 1 if gold in cands else None})
                print(".", end="", flush=True)
            for q in HARD_NEGATIVES:
                t0 = time.perf_counter()
                r = tree.recall(q, reinforce=False, synthesize=False)
                neg.append({"q": q, "grp": neg_group(q), "ms": (time.perf_counter() - t0) * 1000, "r": r})
                print(".", end="", flush=True)
        print(" listo")
        return pos, neg

    def answer(it: dict) -> str:
        r = it["r"]
        found, tent, ans = bool(r.get("found")), bool(r.get("tentative")), str(r.get("answer") or "")
        ltm = {"found": found, "tentative": tent, "sources": [],
               "answer": "" if tent else (ans if found else ""), "tentative_answer": ans if (found and tent) else ""}
        system = Agent._build_system(None, {}, ltm, None)
        try:
            return chat.chat([{"role": "system", "content": system}, {"role": "user", "content": it["q"]}],
                             max_tokens=350, temperature=0.2)
        except Exception as exc:
            return f"[error LLM: {type(exc).__name__}]"

    def judge(system: str, user: str, key: str) -> bool | None:
        try:
            obj = chat.ask_json(system, user)
            return bool(obj.get(key)) if isinstance(obj, dict) and key in obj else None
        except Exception:
            return None

    def judge_pos(it: dict) -> bool | None:
        return judge(RR_POS_JUDGE, f"Pregunta: {it['q']}\nDato correcto: {gold_txt.get(id_of[it['fi']], '')}\n"
                                   f"Respuesta del asistente: {it['a']}", "correcta")

    def judge_neg(it: dict) -> bool | None:
        # mayoría: 2 votos; si discrepan (o falla uno), un tercero
        u = f"Pregunta: {it['q']}\nRespuesta del asistente: {it['a']}"
        v1, v2 = judge(RR_NEG_JUDGE, u, "inventa"), judge(RR_NEG_JUDGE, u, "inventa")
        if v1 is not None and v1 == v2:
            return v1
        votes = [v for v in (v1, v2, judge(RR_NEG_JUDGE, u, "inventa")) if v is not None]
        return (2 * sum(votes) > len(votes)) if votes else None

    rows: dict[str, tuple[list[dict], list[dict], dict[str, Any] | None]] = {}
    for label, over in configs:
        pos, neg = run(label, over)
        e2e = None
        if ctx.live:
            print(f"   … {label}: respuesta del agente real + juez ({workers} en paralelo)", end="", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for it, a in zip(pos + neg, ex.map(answer, pos + neg)):
                    it["a"] = a
                for it, v in zip(pos, ex.map(judge_pos, pos)):
                    it["ok"] = v
                for it, v in zip(neg, ex.map(judge_neg, neg)):
                    it["inv"] = v
            print(" listo")
            n_inv = sum(1 for it in neg if it["inv"] is not False)          # sin veredicto = inventa
            _lo, hi = _wilson(n_inv, len(neg))
            e2e = {"c": sum(1 for it in pos if it["ok"]) / len(pos), "i": n_inv / len(neg), "i_hi": hi,
                   "n_inv": n_inv,
                   "by_grp": {g: (sum(1 for it in neg if it["grp"] == g and it["inv"] is not False),
                                  sum(1 for it in neg if it["grp"] == g)) for g in ("personal", "conocimiento")}}
        rows[label] = (pos, neg, e2e)

    ok1 = lambda it: it["rank"] == 1
    ok3 = lambda it: bool(it["rank"] and it["rank"] <= 3)
    cnt = lambda xs, k: sum(1 for it in xs if state(it["r"]) == k)
    n, n_neg = len(qs), len(HARD_NEGATIVES)
    print(f"\n   RESULTADOS POR CONFIGURACIÓN  (corpus denso: {n} positivas · {n_neg} negativas cercanas)")
    print(f"   {'configuración':<17}{'Hit@1':>7}{'ind@1':>7}{'MRR':>7}   {'positivas f/t/r':>15}   "
          f"{'negativas f/t/r':>15}{'P95ms':>7}   e2e correctas · inventa (IC95% sup.)")
    p95_of: dict[str, float] = {}
    for label, (pos, neg, e2e) in rows.items():
        h1 = sum(map(ok1, pos)) / n
        mrr = sum(1.0 / it["rank"] for it in pos if it["rank"]) / n
        ind = [it for it in pos if it["tipo"] == "indirecta"]
        ind1 = sum(map(ok1, ind)) / max(1, len(ind))
        p95 = percentile([it["ms"] for it in pos + neg], 95)
        p95_of[label] = p95
        e2e_s = (f"{pct(e2e['c'])} · {e2e['n_inv']}/{n_neg} = {pct(e2e['i'])} (≤{pct(e2e['i_hi'])})"
                 if e2e else "— (requiere --live)")
        print(f"   {label:<17}{pct(h1):>7}{pct(ind1):>7}{mrr:>7.3f}   "
              f"{cnt(pos, 'firme'):>5}/{cnt(pos, 'tentativa'):>3}/{cnt(pos, 'rechaza'):>3}     "
              f"{cnt(neg, 'firme'):>5}/{cnt(neg, 'tentativa'):>3}/{cnt(neg, 'rechaza'):>3}{p95:>7.0f}   {e2e_s}")
        R.add(S, f"Hit@1 · {label}", True, pct(h1), "(observación)", "info", "rerank.py")
        R.add(S, f"Negativas afirmadas con certeza · {label}", True, f"{cnt(neg, 'firme')}/{len(neg)}",
              "(observación)", "info", "rerank.py")
        R.add(S, f"P95 recuperación · {label}", True, f"{p95:.0f} ms", "(observación)", "info", "rerank.py")
        if e2e:
            R.add(S, f"e2e correctas / inventa · {label}", True, f"{pct(e2e['c'])} / {pct(e2e['i'])}",
                  "(observación)", "info", "rerank.py + agente.py")
            bg = e2e["by_grp"]
            print(f"   {'':<17}inventa por grupo: personal {bg['personal'][0]}/{bg['personal'][1]} · "
                  f"conocimiento {bg['conocimiento'][0]}/{bg['conocimiento'][1]}")
    print("   (f/t/r = firme / tentativa (con aviso) / rechaza · e2e = respuesta real del agente, juez por mayoría)")

    for label, (pos, neg, _e) in rows.items():
        if label == "off":
            continue
        vp = sum(1 for it in pos if (it["r"].get("rerank") or {}).get("verdict") is False)
        vn = sum(1 for it in neg if (it["r"].get("rerank") or {}).get("verdict") is False)
        fails = sum(1 for it in pos + neg if it["r"].get("rerank") is None)
        print(f"   [{label}] juez «nada responde»: positivas {vp}/{len(pos)} · negativas {vn}/{len(neg)} · "
              f"sin veredicto (→ compuerta clásica): {fails}")

    ck = next((lb for lb in rows if lb.startswith("cross")), None)
    if ck:
        pos, neg, _e = rows[ck]
        ps = [s for s in ((it["r"].get("rerank") or {}).get("top_score") for it in pos) if s is not None]
        ns = [s for s in ((it["r"].get("rerank") or {}).get("top_score") for it in neg) if s is not None]
        if ps and ns:
            print("   · Calibración cross-encoder (top_score 0-1): umbral → positivas que responden | negativas rechazadas")
            for th in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                print(f"       {th:>4.2f} → {pct(sum(1 for s in ps if s >= th) / len(ps)):>6} | "
                      f"{pct(sum(1 for s in ns if s < th) / len(ns)):>6}")

    ns_ = NORTH_STAR
    print("\n   🎯 OBJETIVO: el agente responde bien lo que sabe y no inventa lo que no sabe, en tiempo acotado")
    print(f"      metas: correctas ≥ {pct(ns_['correctas'])} · inventa ≤ {pct(ns_['inventa'])} · "
          f"P95 recuperación ≤ {ns_['p95_ms']:.0f} ms   (IC95% sup. = peor caso plausible con {n_neg} negativas)")
    flat = lambda s: " ".join(str(s).split())
    for label, (pos, neg, e2e) in rows.items():
        tag = "   ← producción" if label == prod else ""
        if not e2e:
            print(f"      {label:<16} P95 {p95_of[label]:.0f} ms (e2e solo con --live){tag}")
            continue
        c, i = e2e["c"], e2e["i"]
        ok = c >= ns_["correctas"] and i <= ns_["inventa"] and p95_of[label] <= ns_["p95_ms"]
        print(f"      {label:<16} correctas {pct(c)} {'✓' if c >= ns_['correctas'] else '✗'} · inventa {pct(i)} "
              f"{'✓' if i <= ns_['inventa'] else '✗'} (≤{pct(e2e['i_hi'])}) · P95 {p95_of[label]:.0f} ms "
              f"{'✓' if p95_of[label] <= ns_['p95_ms'] else '✗'}   {'CUMPLE' if ok else ''}{tag}")
    for label, (pos, neg, e2e) in rows.items():
        if not e2e:
            continue
        bad_p = [it for it in pos if not it.get("ok")]
        bad_n = [it for it in neg if it.get("inv") is not False]
        print(f"\n   [{label}] positivas mal respondidas: {len(bad_p)}/{len(pos)}")
        for it in bad_p[:8]:
            print(f"     [{it['grp'][:5]}|{it['tipo'][:4]}|{state(it['r'])}] «{it['q'][:52]}» → «{flat(it['a'])[:110]}»")
        print(f"   [{label}] negativas donde el agente inventa (auditar a mano): {len(bad_n)}/{len(neg)}")
        for it in bad_n[:20]:
            print(f"     [{it['grp'][:5]}|{state(it['r'])}] «{it['q'][:52]}» → «{flat(it['a'])[:140]}»")

    if prod in rows and rows[prod][2]:
        pos, neg, e2e = rows[prod]
        c, i = e2e["c"], e2e["i"]
        R.add(S, "OBJETIVO · respuestas correctas del agente (corpus denso)", c >= ns_["correctas"], pct(c),
              f"≥ {pct(ns_['correctas'])}", "alto", "rerank.py · RERANK_MODE / ltm/retrieval.py",
              f"Configuración de producción: {prod}.",
              [f"«{it['q'][:60]}» → «{flat(it['a'])[:60]}»" for it in pos if not it.get("ok")][:4])
        R.add(S, "OBJETIVO · invenciones ante preguntas sin respuesta cercanas", i <= ns_["inventa"],
              f"{pct(i)} (≤{pct(e2e['i_hi'])})", f"≤ {pct(ns_['inventa'])}", "alto",
              "rerank.py · RERANK_NONE / ltm/retrieval.py", f"Configuración de producción: {prod}.",
              [f"«{it['q'][:60]}» → «{flat(it['a'])[:60]}»" for it in neg if it.get("inv") is not False][:4])
        R.add(S, "OBJETIVO · latencia P95 de recuperación", p95_of[prod] <= ns_["p95_ms"], f"{p95_of[prod]:.0f} ms",
              f"≤ {ns_['p95_ms']:.0f} ms", "medio", "rerank.py")


# ---------------------------------------------------------------- escala  [PARCHE6]
# «Memoria infinita»: el objetivo debe mantenerse cuando la memoria crece. Se añaden hechos de RELLENO
# realistas y difíciles: datos de TERCERAS personas (vecinos, compañeros...) con los mismos atributos que el
# usuario (mascota, ciudad, alergias, medicación, jefe, grupo sanguíneo...) e información local inventada.
# Ningún relleno responde a una pregunta del benchmark: nunca hablan del usuario ni de su familia, y los
# fármacos usados no responden a ninguna negativa.
_ESC_M = ["Tomás", "Javier", "Sergio", "Hugo", "Iván", "Óscar", "Raúl", "Marcos", "Adrián", "Gonzalo", "Héctor",
          "Ismael", "Jorge", "Luis", "Mario", "Nicolás", "Pablo", "Quique", "Samuel", "Víctor", "Álvaro", "Bruno",
          "César", "Diego", "Emilio", "Fermín", "Gabriel", "Jaime"]
_ESC_F = ["Marta", "Elena", "Sara", "Paula", "Irene", "Nuria", "Rosa", "Silvia", "Beatriz", "Cristina", "Diana",
          "Eva", "Fátima", "Gloria", "Inés", "Julia", "Laura", "Mónica", "Noelia", "Olga", "Pilar", "Raquel",
          "Sonia", "Teresa", "Verónica", "Ana", "Clara", "Lorena"]
_ESC_REL = [("El vecino", "La vecina"), ("El compañero de trabajo", "La compañera de trabajo"),
            ("El amigo de la infancia", "La amiga de la infancia"), ("El compañero del gimnasio", "La compañera del gimnasio"),
            ("El profesor de yoga", "La profesora de yoga"), ("El socio del club de lectura", "La socia del club de lectura")]
_ESC_CITY = ["Lugo", "Soria", "Teruel", "Cáceres", "Huesca", "Palencia", "Segovia", "Ávila", "Cuenca", "Jaén", "Ourense",
             "Logroño", "Gijón", "Vigo", "Murcia", "Alicante", "Córdoba", "Granada", "Burgos", "Oviedo"]
_ESC_BARRIO = ["San Lázaro", "El Ejido", "La Chantría", "Santa Ana", "El Ensanche", "La Florida", "Los Pinos", "El Puerto"]
_ESC_PET = [("perro", ["Rocky", "Toby", "Lur", "Nala", "Coco", "Simba"]), ("gato", ["Mishi", "Garfield", "Luna", "Pelusa"]),
            ("loro", ["Pepe", "Kiwi"]), ("conejo", ["Tambor", "Copito"])]
_ESC_JOB = [("fontanero", "fontanera"), ("arquitecto", "arquitecta"), ("profesor de historia", "profesora de historia"),
            ("farmacéutico", "farmacéutica"), ("conductor de autobús", "conductora de autobús"), ("abogado", "abogada"),
            ("cocinero", "cocinera"), ("electricista", "electricista"), ("periodista", "periodista"), ("veterinario", "veterinaria")]
_ESC_ALLERG = ["el polen", "los ácaros", "el marisco", "los frutos secos", "el látex", "las picaduras de abeja"]
_ESC_INSTR = ["el piano", "el violín", "la batería", "el saxofón", "la gaita", "el ukelele"]
_ESC_MONTH = ["enero", "febrero", "marzo", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre",
              "diciembre"]
_ESC_CAR = ["Seat Ibiza", "Renault Clio", "Ford Focus", "Kia Ceed", "Peugeot 308", "Dacia Sandero"]
_ESC_COLOR = ["rojo", "azul", "blanco", "negro", "verde"]
_ESC_DEGREE = ["Derecho", "Medicina", "Historia", "Biología", "Arquitectura", "Económicas", "Magisterio"]
_ESC_BLOOD = ["A+", "A−", "B+", "0+", "0−", "AB+"]
_ESC_DRUG = ["omeprazol", "levotiroxina", "paracetamol", "amlodipino", "losartán"]
_ESC_COND = ["asma", "migraña", "artrosis de rodilla", "hipotiroidismo", "rinitis alérgica"]
_ESC_SPORT = ["pádel", "natación", "ciclismo", "escalada", "baloncesto"]
_ESC_FOOD = ["el pepino", "la coliflor", "las aceitunas", "el queso azul", "las alcachofas"]
_ESC_COUNTRY = ["Portugal", "Italia", "Grecia", "Marruecos", "Islandia", "Japón", "Escocia"]
_ESC_SYL = ["val", "mor", "tre", "ana", "cas", "lin", "dor", "ve", "ra", "sel", "bu", "qui", "nal", "to", "ria"]


def _escala_distractors(n: int, seed: int = 7) -> list[str]:
    import random as _random
    rnd = _random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    for _ in range(n * 20):
        if len(out) >= n:
            break
        fem = rnd.random() < 0.5
        name = rnd.choice(_ESC_F if fem else _ESC_M)
        s = f"{rnd.choice(_ESC_REL)[1 if fem else 0]} {name}"
        o = "a" if fem else "o"
        pet, petnames = rnd.choice(_ESC_PET)
        place = "".join(rnd.choice(_ESC_SYL) for _ in range(3)).capitalize()
        k = rnd.randrange(22)
        t = [
            f"{s} tiene un {pet} llamado {rnd.choice(petnames)}.",
            f"{s} vive en {rnd.choice(_ESC_CITY)}, en el barrio de {rnd.choice(_ESC_BARRIO)}.",
            f"{s} nació en {rnd.choice(_ESC_CITY)} en {rnd.randint(1950, 2005)}.",
            f"{s} trabaja como {rnd.choice(_ESC_JOB)[1 if fem else 0]} en {rnd.choice(_ESC_CITY)}.",
            f"{s} es alérgic{o} " + (lambda x: "al " + x[3:] if x.startswith("el ") else "a " + x)(rnd.choice(_ESC_ALLERG)) + ".",
            f"{s} toca {rnd.choice(_ESC_INSTR)} desde hace {rnd.randint(2, 30)} años.",
            f"El cumpleaños de {name} es el {rnd.randint(1, 28)} de {rnd.choice(_ESC_MONTH)}.",
            f"{s} conduce un {rnd.choice(_ESC_CAR)} de color {rnd.choice(_ESC_COLOR)}.",
            f"{s} estudió {rnd.choice(_ESC_DEGREE)} en la Universidad de {rnd.choice(_ESC_CITY)}.",
            f"La hija de {name} se llama {rnd.choice(_ESC_F)}.",
            f"El hijo de {name} se llama {rnd.choice(_ESC_M)}.",
            f"La suegra de {name} se llama {rnd.choice(_ESC_F)}.",
            f"El jefe de {name} se llama {rnd.choice(_ESC_M)}.",
            f"El grupo sanguíneo de {name} es {rnd.choice(_ESC_BLOOD)}.",
            f"{s} toma {rnd.choice(_ESC_DRUG)} desde {rnd.randint(2005, 2025)}.",
            f"{s} tiene {rnd.choice(_ESC_COND)}.",
            f"{s} practica {rnd.choice(_ESC_SPORT)} los fines de semana.",
            f"A {name} no le gusta {rnd.choice(_ESC_FOOD)}.",
            f"{s} viajó a {rnd.choice(_ESC_COUNTRY)} de vacaciones en {rnd.randint(2010, 2025)}.",
            f"El pueblo de {place} celebra su fiesta mayor el {rnd.randint(1, 28)} de {rnd.choice(_ESC_MONTH)}.",
            f"El embalse de {place} tiene una capacidad de {rnd.randint(5, 900)} hectómetros cúbicos.",
            f"La ermita de {place} se construyó en el siglo {rnd.choice(['XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII'])}.",
        ][k]
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _escala_insert(tree: Any, texts: list[str], tag: str) -> float:
    # inserción masiva (sin colocación con LLM: aquí se mide la RECUPERACIÓN a escala, no la ontología)
    t0 = time.time()
    root = tree._make_node(None, {"name": "Terceros y miscelánea", "description": "Relleno del benchmark de escala"}, 0)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = [(f"esc-{tag}-{i:06d}", root, t, 5.0, "dato", "benchmark-escala", now, tree.tick, tree.tick)
            for i, t in enumerate(texts)]
    with tree.lock:
        cur = tree.conn.cursor()
        cur.executemany("INSERT INTO cells (id, node_id, text, importance, kind, fuente, ingest_timestamp, "
                        "created_tick, updated_tick) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        cur.execute("UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells WHERE node_id = ? AND archived = 0) "
                    "WHERE id = ?", (root, root))
        tree.conn.commit()
    emb = getattr(tree, "emb", None)
    if emb is not None:
        print("   … vectorizando", end="", flush=True)
        for i in range(400):
            r = emb.sync(limit=512)
            if "al día" in r or "no disponible" in r or "no respondió" in r:
                break
            if i % 4 == 0:
                print(".", end="", flush=True)
        print(" listo")
    return time.time() - t0


ESC_POS_JUDGE = ("Eres un evaluador. Se hizo una pregunta a un asistente con memoria y existe un DATO CORRECTO que la "
                 "responde. Decide si la respuesta contiene ese dato (aunque use otras palabras o añada información "
                 "adicional correcta) y no lo contradice. Si dice que no lo sabe, o da como respuesta principal un dato "
                 "distinto, es incorrecta.\n"
                 'Devuelve SOLO JSON: {"correcta": true} o {"correcta": false}.')
# [PARCHE7] criterio VERIFICABLE: no «¿inventa?» (difuso) sino «¿afirma un valor concreto para lo preguntado?».
# Como estas preguntas NO tienen respuesta en la memoria, afirmar un valor personal = inventar.
ESC_NEG_JUDGE = ("Eres un evaluador. Se hizo a un asistente una pregunta sobre un dato concreto. Decide si la respuesta "
                 "AFIRMA un valor concreto como respuesta a ESA pregunta (un nombre, un lugar, una cifra, una fecha, un "
                 "objeto, un fármaco...) o si, por el contrario, dice que ese dato no consta o que no lo sabe.\n"
                 "Mencionar otros datos del mismo sujeto, o datos de otras personas, NO es afirmar un valor para lo "
                 "preguntado. Solo importa si responde a la pregunta con un valor.\n"
                 'Devuelve SOLO JSON: {"afirma": true} o {"afirma": false}.')


def suite_escala(ctx: Ctx) -> None:
    # [PARCHE7] métricas que SÍ dependen de la memoria + juez verificable + referencia sin memoria + motor vectorial
    # [PARCHE8] por defecto 10.000 y 100.000 hechos de relleno, e2e en el máximo y consulta tras añadir 1 hecho
    R, S = ctx.report, "escala"
    tree, id_of, qs = _hard_corpus(ctx)
    levels = [int(x) for x in os.getenv("HMEM_ESCALA", "10000,100000").split(",") if x.strip()]
    e2e_at = os.getenv("HMEM_ESCALA_E2E", "max").replace("max", str(max(levels) if levels else 0))
    e2e_levels = {int(x) for x in e2e_at.split(",") if x.strip()}
    pmode = str(getattr(C, "RERANK_MODE", "off")).strip().lower()
    ppol = "abstiene" if str(getattr(C, "RERANK_NONE", "tentative")).strip().lower() == "abstain" else "tentativa"
    prod = f"{pmode}·{ppol}" if pmode in ("llm", "cross") else "off"
    configs: list[tuple[str, dict[str, Any]]] = [("off", {"RERANK_MODE": "off"}),
                                                 ("llm·tentativa", {"RERANK_MODE": "llm", "RERANK_NONE": "tentative"})]
    if prod not in [c[0] for c in configs]:
        configs.append((prod, {"RERANK_MODE": pmode, "RERANK_NONE": "abstain" if ppol == "abstiene" else "tentative"}))
    gold_txt = cell_texts(tree, list(id_of.values()))
    base_n = int(dbq(tree, "SELECT COUNT(*) FROM cells WHERE archived = 0")[0][0])
    backend = "numpy" if getattr(vectores, "np", None) is not None else "Python puro (lento: pip install numpy)"
    print(f"   producción: «{prod}» · relleno: {levels} · e2e en: {sorted(e2e_levels)} · base: {base_n} hechos · "
          f"motor vectorial: {backend}")
    chat = None
    if ctx.live:
        from llm import LLM
        from agente import Agent
        chat = LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
    workers = max(1, int(os.getenv("HMEM_E2E_WORKERS", "2")))
    neg_group = lambda q: "personal" if ("usuario" in q or "Alicia" in q) else "conocimiento"
    state = lambda r: "rechaza" if not r.get("found") else ("tentativa" if r.get("tentative") else "firme")
    flat = lambda s: " ".join(str(s).split())
    n_pers = sum(1 for q in HARD_NEGATIVES if neg_group(q) == "personal")
    n_user = sum(1 for _q, _f, _t, g in qs if g == "Usuario")

    def run(label: str, over: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        pos: list[dict] = []
        neg: list[dict] = []
        print(f"   … {label}: recuperando", end="", flush=True)
        with _cfg_override(**over):
            for q, fi, tipo, grp in qs:
                t0 = time.perf_counter()
                r = tree.recall(q, reinforce=False, synthesize=False)
                cands = list(r.get("candidates") or [])
                gold = id_of[fi]
                pos.append({"q": q, "fi": fi, "tipo": tipo, "grp": grp, "r": r,
                            "ms": (time.perf_counter() - t0) * 1000,
                            "rank": cands.index(gold) + 1 if gold in cands else None})
            for q in HARD_NEGATIVES:
                t0 = time.perf_counter()
                r = tree.recall(q, reinforce=False, synthesize=False)
                neg.append({"q": q, "grp": neg_group(q), "r": r, "ms": (time.perf_counter() - t0) * 1000})
        print(" listo")
        return pos, neg

    def answer(it: dict) -> str:
        r = it["r"]
        found, tent, ans = bool(r.get("found")), bool(r.get("tentative")), str(r.get("answer") or "")
        ltm = {"found": found, "tentative": tent, "sources": [],
               "answer": "" if tent else (ans if found else ""), "tentative_answer": ans if (found and tent) else ""}
        try:
            return chat.chat([{"role": "system", "content": Agent._build_system(None, {}, ltm, None)},
                              {"role": "user", "content": it["q"]}], max_tokens=350, temperature=0.2)
        except Exception as exc:
            return f"[error LLM: {type(exc).__name__}]"

    def judge(system: str, user: str, key: str) -> bool | None:
        try:
            obj = chat.ask_json(system, user)
            return bool(obj.get(key)) if isinstance(obj, dict) and key in obj else None
        except Exception:
            return None

    def vote(system: str, user: str, key: str) -> bool | None:
        v1, v2 = judge(system, user, key), judge(system, user, key)
        if v1 is not None and v1 == v2:
            return v1
        votes = [v for v in (v1, v2, judge(system, user, key)) if v is not None]
        return (2 * sum(votes) > len(votes)) if votes else None

    def judge_pos(it: dict) -> bool | None:
        return vote(ESC_POS_JUDGE, f"Pregunta: {it['q']}\nDato correcto: {gold_txt.get(id_of[it['fi']], '')}\n"
                                   f"Respuesta del asistente: {it['a']}", "correcta")

    def judge_neg(it: dict) -> bool | None:
        return vote(ESC_NEG_JUDGE, f"Pregunta: {it['q']}\nRespuesta del asistente: {it['a']}", "afirma")

    def e2e(pos: list[dict], neg: list[dict], label: str) -> dict[str, Any]:
        print(f"   … {label}: agente real + juez ({workers} en paralelo)", end="", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for it, a in zip(pos + neg, ex.map(answer, pos + neg)):
                it["a"] = a
            for it, v in zip(pos, ex.map(judge_pos, pos)):
                it["ok"] = v
            for it, v in zip(neg, ex.map(judge_neg, neg)):
                it["afirma"] = v
        print(" listo")
        user = [it for it in pos if it["grp"] == "Usuario"]
        pers = [it for it in neg if it["grp"] == "personal"]
        know = [it for it in neg if it["grp"] == "conocimiento"]
        n_inv = sum(1 for it in pers if it["afirma"] is not False)     # sin veredicto = inventa
        return {"c": sum(1 for it in pos if it["ok"]) / len(pos),
                "cu": sum(1 for it in user if it["ok"]) / max(1, len(user)),
                "i": n_inv / max(1, len(pers)), "i_hi": _wilson(n_inv, len(pers))[1], "n_inv": n_inv,
                "kn": sum(1 for it in know if it["afirma"]), "n_kn": len(know)}

    table: list[dict[str, Any]] = []
    detail: dict[tuple[int, str], tuple[list[dict], list[dict]]] = {}
    added = 0
    for lvl in [0] + levels:
        if lvl > added:
            extra = _escala_distractors(lvl, seed=7)[added:lvl]
            secs = _escala_insert(tree, extra, f"{lvl}")
            print(f"   + {len(extra)} hechos de relleno insertados ({secs:.0f}s)")
            added = lvl
        n_mem = int(dbq(tree, "SELECT COUNT(*) FROM cells WHERE archived = 0")[0][0])
        print(f"\n   ── NIVEL +{lvl}: {n_mem} hechos en memoria ──")
        if ctx.live and lvl == 0 and 0 in e2e_levels:
            # referencia SIN memoria: cuánto acierta el LLM solo con su conocimiento (mide qué preguntas dependen de la memoria)
            pos = [{"q": q, "fi": fi, "tipo": t, "grp": g, "r": {"found": False}, "ms": 0.0, "rank": None}
                   for q, fi, t, g in qs]
            neg = [{"q": q, "grp": neg_group(q), "r": {"found": False}, "ms": 0.0} for q in HARD_NEGATIVES]
            table.append({"lvl": 0, "n": 0, "cfg": "sin memoria", "h1": None, "r6": None, "r18": None, "negf": None,
                          "p95": 0.0, "e2e": e2e(pos, neg, "sin memoria")})
            detail[(0, "sin memoria")] = (pos, neg)
        for label, over in configs:
            pos, neg = run(label, over)
            row: dict[str, Any] = {"lvl": lvl, "n": n_mem, "cfg": label,
                                   "h1": sum(1 for it in pos if it["rank"] == 1) / len(pos),
                                   "r6": sum(1 for it in pos if it["rank"] and it["rank"] <= 6) / len(pos),
                                   "r18": sum(1 for it in pos if it["rank"]) / len(pos),
                                   "negf": sum(1 for it in neg if state(it["r"]) == "firme"),
                                   "p95": percentile([it["ms"] for it in pos + neg], 95), "e2e": None}
            if ctx.live and lvl in e2e_levels:
                row["e2e"] = e2e(pos, neg, label)
                detail[(lvl, label)] = (pos, neg)
            table.append(row)

    ns_ = NORTH_STAR
    print(f"\n   ESCALA  ({len(qs)} positivas, {n_user} de ellas del usuario · {len(HARD_NEGATIVES)} negativas: "
          f"{n_pers} personales + {len(HARD_NEGATIVES) - n_pers} de conocimiento general)")
    print(f"   {'relleno':>8}{'hechos':>8}  {'configuración':<15}{'Hit@1':>7}{'R@6':>7}{'R@18':>7}{'P95ms':>7}   "
          f"{'correctas':>9}{'usuario':>9}   inventa (personales, IC95% sup.) · conoc.general respondidas")
    for row in table:
        e = row["e2e"]
        ret = (f"{pct(row['h1']):>7}{pct(row['r6']):>7}{pct(row['r18']):>7}{row['p95']:>7.0f}" if row["h1"] is not None
               else f"{'—':>7}{'—':>7}{'—':>7}{'—':>7}")
        e2e_s = (f"{pct(e['c']):>9}{pct(e['cu']):>9}   {e['n_inv']}/{n_pers} = {pct(e['i'])} (≤{pct(e['i_hi'])}) · "
                 f"{e['kn']}/{e['n_kn']}" if e else f"{'—':>9}{'—':>9}")
        tag = "  ← producción" if row["cfg"] == prod else ""
        print(f"   {'+' + str(row['lvl']):>8}{row['n']:>8}  {row['cfg']:<15}{ret}   {e2e_s}{tag}")
        if row["h1"] is not None:
            R.add(S, f"Hit@1 · +{row['lvl']} · {row['cfg']}", True, pct(row["h1"]), "(observación)", "info", "escala")
            R.add(S, f"P95 · +{row['lvl']} · {row['cfg']}", True, f"{row['p95']:.0f} ms", "(observación)", "info", "escala")
        if e:
            R.add(S, f"e2e correctas / usuario / inventa · +{row['lvl']} · {row['cfg']}", True,
                  f"{pct(e['c'])} / {pct(e['cu'])} / {pct(e['i'])}", "(observación)", "info", "escala")
    print("   (usuario = correctas en las preguntas sobre el usuario, las únicas que exigen memoria · inventa = afirma un "
          "valor para un dato personal que NO consta · conoc.general = negativas de conocimiento general que el LLM "
          "responde con lo que sabe: no es inventar)")

    for (lvl, label), (pos, neg) in sorted(detail.items()):
        bad_p = [it for it in pos if not it.get("ok") and (label != "sin memoria" or it["grp"] == "Usuario")]
        bad_n = [it for it in neg if it["grp"] == "personal" and it.get("afirma") is not False]
        if label == "sin memoria":
            print(f"\n   [sin memoria] preguntas del usuario acertadas sin memoria: "
                  f"{sum(1 for it in pos if it['grp'] == 'Usuario' and it.get('ok'))}/{n_user} (deberían ser ~0)")
            continue
        print(f"\n   [+{lvl} · {label}] positivas mal respondidas: {len(bad_p)}/{len(pos)} · "
              f"afirma un dato personal que no consta: {len(bad_n)}/{n_pers}")
        for it in bad_p[:6]:
            print(f"     ✗pos [{it['grp'][:5]}|{it['tipo'][:4]}|{state(it['r'])}|rank={it['rank']}] «{it['q'][:48]}» → "
                  f"«{flat(it['a'])[:100]}»")
        for it in bad_n[:10]:
            print(f"     ✗neg [{state(it['r'])}] «{it['q'][:48]}» → «{flat(it['a'])[:120]}»")

    lat_rows = [r for r in table if r["cfg"] == prod and r["h1"] is not None]
    if len(lat_rows) >= 2:
        f_l, l_l = lat_rows[0], lat_rows[-1]
        ms_per_k = (l_l["p95"] - f_l["p95"]) / max(1.0, (l_l["n"] - f_l["n"]) / 1000)
        print(f"\n   latencia ({prod}): {f_l['p95']:.0f} ms con {f_l['n']} hechos → {l_l['p95']:.0f} ms con {l_l['n']} "
              f"(+{ms_per_k:.1f} ms por cada 1.000) · proyección a 1.000.000 de hechos ≈ "
              f"{l_l['p95'] + ms_per_k * (1000000 - l_l['n']) / 1000:.0f} ms · motor vectorial: {backend}")

    # aprendizaje continuo: la PRIMERA consulta tras añadir un hecho no debe pagar la reconstrucción de todo el índice
    if getattr(tree, "emb", None) is not None:
        nuevo = "El vecino Telmo tiene un hurón llamado Chispa."
        _escala_insert(tree, [nuevo], "nuevo")
        times, top = [], ""
        with _cfg_override(RERANK_MODE="off"):
            for _ in range(2):
                t0 = time.perf_counter()
                r = tree.recall("¿Cómo se llama el hurón del vecino Telmo?", reinforce=False, synthesize=False)
                times.append((time.perf_counter() - t0) * 1000)
                top = cell_texts(tree, (r.get("candidates") or [])[:1]).get((r.get("candidates") or [""])[0], "")
        n_now = int(dbq(tree, "SELECT COUNT(*) FROM cells WHERE archived = 0")[0][0])
        print(f"   aprendizaje continuo ({n_now} hechos): 1.ª consulta tras añadir 1 hecho {times[0]:.0f} ms · "
              f"2.ª {times[1]:.0f} ms · top-1: «{top[:50]}»")
        R.add(S, f"Consulta tras añadir 1 hecho ({n_now} hechos)", times[0] <= NORTH_STAR["p95_ms"] and nuevo in top,
              f"{times[0]:.0f} ms", f"≤ {NORTH_STAR['p95_ms']:.0f} ms y lo encuentra", "medio",
              "vectores.py · VectorStore._matrix (caché incremental)",
              "Si tarda mucho más que la 2.ª consulta, cada hecho nuevo reconstruye el índice entero.")

    prod_rows = [r for r in table if r["cfg"] == prod and r["e2e"]]
    if prod_rows:
        first, last = prod_rows[0], prod_rows[-1]
        e = last["e2e"]
        print(f"\n   🎯 OBJETIVO A ESCALA ({prod}, {last['n']} hechos = ×{last['n'] / max(1, base_n):.0f} la base): "
              f"correctas {pct(e['c'])} · del usuario {pct(e['cu'])} · inventa {pct(e['i'])} · P95 {last['p95']:.0f} ms")
        if first is not last:
            print(f"      degradación desde el primer nivel medido: usuario {100 * (e['cu'] - first['e2e']['cu']):+.1f} pp · "
                  f"inventa {100 * (e['i'] - first['e2e']['i']):+.1f} pp · Hit@1 {100 * (last['h1'] - first['h1']):+.1f} pp")
        R.add(S, f"OBJETIVO a escala · correctas ({last['n']} hechos)", e["c"] >= ns_["correctas"], pct(e["c"]),
              f"≥ {pct(ns_['correctas'])}", "alto", "rerank.py / ltm/retrieval.py", f"Producción: {prod}.")
        R.add(S, f"OBJETIVO a escala · correctas del usuario ({last['n']} hechos)", e["cu"] >= ns_["correctas"],
              pct(e["cu"]), f"≥ {pct(ns_['correctas'])}", "alto", "rerank.py / ltm/retrieval.py",
              "Las únicas preguntas que el LLM no puede responder sin memoria.")
        R.add(S, f"OBJETIVO a escala · invenciones personales ({last['n']} hechos)", e["i"] <= ns_["inventa"],
              f"{pct(e['i'])} (≤{pct(e['i_hi'])})", f"≤ {pct(ns_['inventa'])}", "alto", "rerank.py / agente.py")
        R.add(S, f"OBJETIVO a escala · latencia P95 ({last['n']} hechos)", last["p95"] <= ns_["p95_ms"],
              f"{last['p95']:.0f} ms", f"≤ {ns_['p95_ms']:.0f} ms", "medio", "vectores.py / rerank.py")
    if not ctx.live:
        print("   ⚠ modo emulado: valida la fontanería; las cifras solo son informativas con --live")


# ---------------------------------------------------------------- LoCoMo  [PARCHE10]
# Benchmark PÚBLICO: ingesta real de conversaciones ajenas y el mismo OBJETIVO (ver benchmarks/locomo.py).
def suite_locomo(ctx: Ctx) -> None:
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import locomo

    def open_tree(path: Path) -> Any:
        tree = MemoryTree(path, ctx.llm)
        if ctx.mock:
            tree.emb = vectores.EmbeddingService(tree, embedder=MockEmbedder())
        ctx.trees.append(tree)
        return tree

    locomo.run(ctx, open_tree=open_tree, sync_all=sync_all, north_star=NORTH_STAR)


SUITES: list[tuple[str, str, Callable[[Ctx], None]]] = [
    ("ablacion", "Ablación de canales con intervalos y split dev/test", suite_ablacion),
    ("diagnostico", "Diagnóstico: candidatos, discordancias y abstención", suite_diagnostico),
    ("identificadores", "Consultas de palabra clave e identificadores", suite_identificadores),
    ("abstencion", "Abstención: tasa de respuesta vs rechazo", suite_abstencion),
    ("e2e_abstencion", "Abstención extremo a extremo con el LLM real", suite_e2e_abstencion),
    ("recuperacion", "Recuperación, fusión RRF y ranking (108 preguntas)", suite_recuperacion),
    ("dificil", "Corpus denso del mismo dominio: riñón, fármacos y datos personales", suite_dificil),  # [PARCHE3]
    ("reranker", "Reranker y OBJETIVO extremo a extremo (corpus denso)", suite_reranker),  # [PARCHE4]
    ("escala", "OBJETIVO a escala: corpus denso + miles de hechos de relleno", suite_escala),  # [PARCHE6]
    ("locomo", "OBJETIVO en LoCoMo: benchmark público con ingesta real", suite_locomo),  # [PARCHE10]
    ("temporal", "Reconciliación temporal (hechos vigentes vs obsoletos)", suite_temporal),
    ("olvido", "Curva de olvido y resurrección", suite_olvido),
    ("sinapsis", "Sinapsis hebbianas y activación propagada", suite_sinapsis),
    ("concurrencia", "Concurrencia (lecturas/escrituras paralelas)", suite_concurrencia),
    ("integridad", "Integridad estructural de la base de datos", suite_integridad),
    ("persistencia", "Persistencia, corte de WAL y backups", suite_persistencia),
    ("stm", "Memoria a corto plazo y compactación", suite_stm),
    ("cartuchos", "Cartuchos: aislamiento, import/export y path traversal", suite_cartuchos),
    ("seguridad", "Consola web: CSRF, DNS-rebinding y límites", suite_seguridad),
    ("ingesta", "Filtro de ingesta y extracción de hechos", suite_ingesta),
    ("sintesis", "Síntesis con LLM real (solo --live)", suite_sintesis),
]


# ==============================================================================
# 6. EJECUCIÓN E INFORME FINAL
# ==============================================================================

def preflight_live() -> bool:
    try:
        import servidor
        ok0 = servidor.is_remote_health_ok(C.SERVER_PORT_GPU0)
        ok1 = servidor.is_remote_health_ok(C.SERVER_PORT_GPU1)
    except Exception:
        ok0 = ok1 = False
    emb = vectores.Embedder().embed(["prueba"])
    print(f"   Servidor LTM  (:{C.SERVER_PORT_GPU0}): {'✓' if ok0 else '✗'}   "
          f"Servidor STM (:{C.SERVER_PORT_GPU1}): {'✓' if ok1 else '✗'}   "
          f"Embeddings (:{C.SERVER_PORT_GPU2}): {'✓' if emb else '✗ (la búsqueda semántica quedará inactiva)'}")
    if not (ok0 and ok1):
        print("❌ Faltan servidores LLM. Arráncalos o ejecuta sin --live.")
        return False
    return True


def final_report(ctx: Ctx, report: Report, args: argparse.Namespace) -> int:
    print("\n" + "=" * 84)
    print(" 📊 RESUMEN")
    print("=" * 84)
    by_suite: dict[str, list[Check]] = {}
    for c in report.checks:
        by_suite.setdefault(c.suite, []).append(c)
    for key, title, _ in SUITES:
        if key in report.suite_skipped:
            print(f"  –  {key:<13} omitida ({report.suite_skipped[key]})")
            continue
        cs = by_suite.get(key)
        if not cs:
            continue
        scored = [c for c in cs if c.severity != "info"]
        ok = sum(1 for c in scored if c.ok)
        mark = "✓" if ok == len(scored) else "✗"
        print(f"  {mark}  {key:<13} {ok}/{len(scored)} comprobaciones puntuables   ({report.suite_time.get(key, 0):.1f}s)")

    sc = report.score()
    if not any(c.severity != "info" for c in report.checks):   # [PARCHE2]
        print("\n  PUNTUACIÓN PONDERADA: —  →  SIN COMPROBACIONES PUNTUABLES (solo observaciones)")
    else:
        verdict = ("SÓLIDO — evidencia suficiente" if sc >= 9 else
                   "ACEPTABLE CON RESERVAS" if sc >= 7.5 else
                   "DÉBIL — hay defectos que corregir" if sc >= 6 else "INSUFICIENTE")
        print(f"\n  PUNTUACIÓN PONDERADA: {sc:.1f} / 10  →  {verdict}")
    print(f"  (pesos: crítico×3, alto×2, medio×1; las 'observaciones' no puntúan)  · modo: "
          f"{'REAL (--live)' if ctx.live else 'EMULADO (valida lógica, no calidad semántica)'}")

    failed = sorted([c for c in report.checks if not c.ok], key=lambda c: SEV_ORDER[c.severity])
    weak = [c for c in failed if c.severity != "info"]
    obs = [c for c in failed if c.severity == "info"]
    print("\n" + "=" * 84)
    print(" 🔎 PUNTOS DÉBILES")
    print("=" * 84)
    if not weak:
        print("  ✓ Ninguna comprobación puntuable ha fallado.")
    for i, c in enumerate(weak, 1):
        print(f"\n  [{i}] {c.severity.upper():<7} {c.suite} · {c.name}")
        print(f"      Medido: {c.value}   (meta {c.target})")
        if c.where:
            print(f"      Dónde mirar: {c.where}")
        if c.hint:
            print(f"      Pista: {c.hint}")
        for e in c.evidence[:5]:
            print(f"      ↳ {e}")
    if obs:
        print("\n  Observaciones (no puntúan en este modo):")
        for c in obs:
            print(f"   • {c.suite} · {c.name}: {c.value} (meta {c.target})")
    if ctx.mock:
        print("\n  ℹ️  Para evidencia de calidad semántica real: python3 benchmarks/prueba.py --live")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "modo": "live" if ctx.live else "emulado",
            "puntuacion": round(sc, 2), "checks": [asdict(c) for c in report.checks],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  Informe JSON guardado en {args.json}")
    bad = report.failures(strict=args.strict)
    print("\n" + ("✅ SIN FALLOS GRAVES" if not bad else f"❌ {len(bad)} FALLO(S) GRAVE(S)") + "\n")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Banco de pruebas con evidencia medible para la memoria jerárquica.")
    ap.add_argument("--live", action="store_true", help="usar los servidores reales configurados en .env")
    ap.add_argument("--only", default="", help="suites separadas por comas (ver --list)")
    ap.add_argument("--list", action="store_true", help="listar suites")
    ap.add_argument("--json", default="", help="guardar resultados en un JSON")
    ap.add_argument("--keep", action="store_true", help="conservar los archivos temporales")
    ap.add_argument("--strict", action="store_true", help="también falla con severidad 'medio'")
    ap.add_argument("--verbose", action="store_true", help="mostrar trazas completas de errores")
    ap.add_argument("--sleep-cycles", type=int, default=18, help="pasos de mantenimiento antes de re-medir (def. 18)")
    args = ap.parse_args()

    if args.list:
        for k, t, _ in SUITES:
            print(f"  {k:<13} {t}")
        return 0
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = wanted - {k for k, _, _ in SUITES}
    if unknown:
        print(f"Suites desconocidas: {', '.join(sorted(unknown))}. Usa --list.")
        return 2

    print("=" * 84)
    print(f" 🧪 BANCO DE PRUEBAS · {'MODO REAL (LIVE)' if args.live else 'MODO EMULADO'}")
    print("=" * 84)
    if args.live and not preflight_live():
        return 2
    if args.live:
        print("   (en modo real cada hecho se coloca con llamadas al LLM: puede tardar varios minutos)")

    report = Report()
    ctx = Ctx(args, report)
    try:
        for key, title, fn in SUITES:
            if wanted and key not in wanted:
                continue
            print(f"\n🧪 [{key}] {title}")
            t0 = time.time()
            try:
                fn(ctx)
            except Exception as exc:
                tb = traceback.format_exc().strip().splitlines()
                report.add(key, "La suite se ejecutó sin excepciones", False, type(exc).__name__, "sin errores", "crítico",
                           "(ver traza)", str(exc)[:200], tb[-6:] if not args.verbose else tb)
            report.suite_time[key] = time.time() - t0
            if key in report.suite_skipped:
                print(f"   – omitida: {report.suite_skipped[key]}")
            report.print_suite(key)
        return final_report(ctx, report, args)
    finally:
        ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())