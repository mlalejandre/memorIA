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
# (nombre, overrides de config, navegar por el árbol)
ABLATIONS: list[tuple[str, dict[str, Any], bool]] = [
    ("actual", {}, True),
    ("solo embeddings", _ONLY_EMB, False),
    ("emb + léxico", {**_ONLY_EMB, "RRF_W_FTS": C.RRF_W_FTS}, False),
    ("emb + exacta + léxico", {**_ONLY_EMB, "RRF_W_FTS": C.RRF_W_FTS, "RRF_W_EXACT": C.RRF_W_EXACT}, False),
    ("sin árbol", {"RRF_W_TREE": 0}, False),
    ("sin sinapsis", {"RRF_W_SYN": 0, "SYN_ENABLED": False}, True),
    ("sin léxico", {"RRF_W_FTS": 0}, True),
    ("sin exacta", {"RRF_W_EXACT": 0}, True),
    ("sin expansión LLM", {"MEM_RECALL_EXPAND": False}, True),
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
    for name, over, nav in ABLATIONS:
        with _cfg_override(**over):
            res[name] = _evaluate(tree, qs, id_of, navigate=nav)

    ok3 = lambda r: bool(r["rank"] and r["rank"] <= 3)
    base, n = res["actual"], len(qs)
    print(f"   {'configuración':<24}{'Hit@3':>7}{'IC95%':>16}{'dev':>7}{'test':>7}{'MRR':>7}{'P95ms':>7}   vs actual")
    for name, rows in res.items():
        k = sum(map(ok3, rows))
        lo, hi = _wilson(k, n)
        ic = f"[{pct(lo)}–{pct(hi)}]"
        dv = sum(ok3(rows[i]) for i in dev_idx) / max(1, len(dev_idx))
        ts = sum(ok3(rows[i]) for i in test_idx) / max(1, len(test_idx))
        if name == "actual":
            cmp = "—"
        else:
            gana = sum(ok3(y) and not ok3(x) for x, y in zip(base, rows))
            pierde = sum(ok3(x) and not ok3(y) for x, y in zip(base, rows))
            cmp = f"+{gana}/−{pierde}  p={_sign_p(pierde, gana):.2f}"
        print(f"   {name:<24}{pct(k / n):>7}{ic:>16}{pct(dv):>7}{pct(ts):>7}"
              f"{_mrr(rows):>7.3f}{percentile([r['ms'] for r in rows], 95):>7.0f}   {cmp}")
        R.add(S, f"Hit@3 · {name}", True, pct(k / n), "(observación)", "info", "config.py · RRF_W_*")
    print(f"   n={n} preguntas · dev={len(dev_idx)} · test={len(test_idx)}")
    print("   (+g/−p: preguntas donde la variante acierta y 'actual' no / al revés; p = test de signos)")
    if not ctx.live:
        print("   ⚠ modo emulado: valida la fontanería, la comparación solo es informativa con --live")


# ---------------------------------------------------------------- diagnóstico
_DEF = {"RRF_W_EXACT": 0.6, "RRF_W_FTS": 0.25, "RRF_W_SEM": 0.3, "RRF_W_TREE": 0.1, "RRF_W_SYN": 0.3, "RRF_W_EMB": 2.5}
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

    # --- abstención
    answered = []
    for q in NEGATIVES:
        r = tree.recall(q, reinforce=False, synthesize=False)
        answered.append(bool(r.get("found")))
    abst = 1 - sum(answered) / len(NEGATIVES)
    R.add(S, "Abstención correcta ante preguntas sin respuesta", abst >= 0.80, pct(abst), "≥ 80%", "medio",
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


SUITES: list[tuple[str, str, Callable[[Ctx], None]]] = [
    ("ablacion", "Ablación de canales con intervalos y split dev/test", suite_ablacion),
    ("diagnostico", "Diagnóstico: candidatos, discordancias y abstención", suite_diagnostico),
    ("identificadores", "Consultas de palabra clave e identificadores", suite_identificadores),
    ("recuperacion", "Recuperación, fusión RRF y ranking (108 preguntas)", suite_recuperacion),
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