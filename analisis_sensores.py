# =============================================================================
#  Autor   : Gabriel Emilio Herrera Balán
# =============================================================================
#  El script esta dividido en dos componentes:
#    COMPONENTE 1 -> Data Cleaning (limpieza + transformacion)
#    COMPONENTE 2 -> EDA (estadistica descriptiva + visualizaciones)
#  Cada problema de calidad detectado se registra en la lista REGISTRO_PROBLEMAS
#  para construir luego la "Tabla de problemas y soluciones" del reporte.
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuracion estetica global de las graficas -------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "axes.titleweight": "bold",
    "font.size": 10,
})
PALETA = "viridis"

# Rutas donde se crearán los .csv y gráficas.
BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_RAW    = os.path.join(BASE, "Production_RawDataSet.csv")
RUTA_LIMPIO = os.path.join(BASE, "sensores_limpios.csv")
DIR_FIG     = os.path.join(BASE, "figuras")
os.makedirs(DIR_FIG, exist_ok=True)   # crea la carpeta de figuras si no existe

# Registro de cada problema encontrado -> alimenta la tabla del reporte
REGISTRO_PROBLEMAS = []
def log_problema(nombre, deteccion, solucion, afectados):
    REGISTRO_PROBLEMAS.append({
        "Problema": nombre,
        "Como se detecto": deteccion,
        "Solucion aplicada": solucion,
        "Registros afectados": afectados,
    })

# =============================================================================
#  CARGA DE DATOS CRUDOS
# =============================================================================
# El archivo usa ';' como separador y trae todo como texto para no perder
# informacion (ej. el string 'null' o minutos invalidos) antes de limpiarlo.
df = pd.read_csv(RUTA_RAW, sep=";", dtype=str)
df.columns = [c.strip() for c in df.columns]
N_INICIAL = len(df)
print(f"Filas crudas: {N_INICIAL} | Columnas: {df.shape[1]}")

# Lista de columnas numericas que representan lecturas de sensores
COLS_SENSOR = ["temperatura_c", "presion_psi", "vibracion_mm_s", "potencia_kw"]

# =============================================================================
#  COMPONENTE 1 - DATA CLEANING
# =============================================================================

# -----------------------------------------------------------------------------
# PROBLEMA 1: Columna basura 'time stamp' (timestamp de ingesta, no de medicion)
# -----------------------------------------------------------------------------
ts_invalidos = df["time stamp"].str.contains(r":[6-9]\d:", regex=True).sum()
df = df.drop(columns=["time stamp"])
log_problema(
    "Columna 'time stamp' irrelevante y con horas invalidas",
    "Valor casi constante (fecha de ingesta) con minutos > 59 (ej. 18:66:05)",
    "Se elimina la columna por no aportar valor analitico",
    f"{ts_invalidos} horas invalidas; columna completa eliminada",
)

# -----------------------------------------------------------------------------
# PROBLEMA 2: Normalizacion de valores faltantes ('null', 'N/A', vacios, ruido)
# -----------------------------------------------------------------------------
TOKENS_NULOS = {"null", "NULL", "N/A", "na", "NA", "nan", "", "zzzzzzzz"}
faltantes_antes = 0
for c in df.columns:
    s = df[c].astype(str).str.strip()
    mask = s.isin(TOKENS_NULOS)
    faltantes_antes += int(mask.sum())
    df[c] = s.where(~mask, np.nan)
log_problema(
    "Placeholders heterogeneos de valores faltantes",
    "Mezcla de 'null', 'N/A', vacios y ruido 'zzzzzzzz' en varias columnas",
    "Unificacion de todos los placeholders a NaN",
    f"{faltantes_antes} celdas normalizadas a NaN",
)

# -----------------------------------------------------------------------------
# PROBLEMA 3: Tipos de dato - conversion de sensores a numerico
# -----------------------------------------------------------------------------
df["id_sensor"] = pd.to_numeric(df["id_sensor"], errors="coerce").astype("Int64")
for c in COLS_SENSOR:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------------------------------------------------------
# PROBLEMA 4: Formato de fecha inconsistente y fechas fuera de rango
# -----------------------------------------------------------------------------
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], dayfirst=True, errors="coerce")
mask_anio = df["fecha_hora"].dt.year > 2030
n_anio = int(mask_anio.sum())
# Restar la decada sobrante (2035->2025, 2045->2025)
df.loc[mask_anio, "fecha_hora"] = df.loc[mask_anio, "fecha_hora"].apply(
    lambda d: d.replace(year=2025)
)
log_problema(
    "Formato de fecha y anios fuera de rango en 'fecha_hora'",
    "Formato D/M/YYYY + anios imposibles 2035 y 2045",
    "Parseo con dayfirst y correccion del anio a 2025 (typo confirmado por secuencia)",
    f"{n_anio} fechas corregidas",
)

# -----------------------------------------------------------------------------
# PROBLEMA 5: Categoria mal escrita en 'maquina_id' (M04D -> M04)
# -----------------------------------------------------------------------------
n_m04d = int((df["maquina_id"] == "M04D").sum())
df["maquina_id"] = df["maquina_id"].str.replace(r"^(M\d{2})\D+$", r"\1", regex=True)
log_problema(
    "Ruido en identificador de maquina",
    "Aparece 'M04D' en lugar de 'M04' (sufijo espurio)",
    "Normalizacion con expresion regular a formato M##",
    f"{n_m04d} registros corregidos",
)

# -----------------------------------------------------------------------------
# PROBLEMA 6: Imputacion de 'maquina_id' faltante por contexto secuencial
# -----------------------------------------------------------------------------
# Si la fila previa y la siguiente (ordenadas por id_sensor) pertenecen a la
# misma maquina, se asume que el hueco corresponde a esa maquina.
df = df.sort_values("id_sensor").reset_index(drop=True)
prev = df["maquina_id"].ffill()
nxt  = df["maquina_id"].bfill()
mask_maq = df["maquina_id"].isna() & (prev == nxt)
n_maq_imp = int(mask_maq.sum())
df.loc[mask_maq, "maquina_id"] = prev[mask_maq]
log_problema(
    "Valores faltantes en 'maquina_id' (columna clave)",
    "Celdas vacias en el identificador de maquina",
    "Imputacion por contexto: si las filas vecinas coinciden, se hereda su maquina",
    f"{n_maq_imp} registros imputados",
)

# -----------------------------------------------------------------------------
# PROBLEMA 7: Valores fuera de rango fisico y outliers extremos en sensores
# -----------------------------------------------------------------------------
# Rangos fisicamente plausibles para este tipo de maquinaria industrial.
# Todo lo que cae fuera (centinelas 999/999.9/99.9, negativos, 1e7, -5e6,
# 0.0001) se considera lectura invalida y pasa a NaN.
RANGOS = {
    "temperatura_c":  (0,   200),   # grados C de operacion industrial
    "presion_psi":    (0,   100),   # psi
    "vibracion_mm_s": (0,   50),    # mm/s (ISO 10816; >50 es fisicamente imposible)
    "potencia_kw":    (0.1, 60),    # kW
}
fuera_rango_total = 0
detalle_rango = {}
for c, (lo, hi) in RANGOS.items():
    mask = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
    detalle_rango[c] = int(mask.sum())
    fuera_rango_total += int(mask.sum())
    df.loc[mask, c] = np.nan
log_problema(
    "Outliers extremos y valores fuera de rango fisico",
    "Centinelas (999, 999.9, 99.9, -99.9), negativos y extremos (1e7, -5.5e6, 0.0001)",
    "Validacion por rango fisico; valores invalidos -> NaN. "
    f"Detalle: {detalle_rango}",
    f"{fuera_rango_total} lecturas invalidadas",
)

# -----------------------------------------------------------------------------
# PROBLEMA 8: Eliminar filas sin NINGUNA lectura valida de sensor
# -----------------------------------------------------------------------------
# Filas donde los 4 sensores quedaron en NaN (apagon total de sensores, ej. la
# fila de '999;999.9;99.9;99.9'). No son recuperables -> se descartan.
mask_vacias = df[COLS_SENSOR].isna().all(axis=1)
n_vacias = int(mask_vacias.sum())
df = df[~mask_vacias].reset_index(drop=True)
log_problema(
    "Filas sin ninguna lectura de sensor valida",
    "Las 4 variables de sensor quedaron en NaN tras la validacion de rango",
    "Eliminacion de la fila (sin senal recuperable)",
    f"{n_vacias} filas eliminadas",
)

# -----------------------------------------------------------------------------
# PROBLEMA 9: Imputacion de sensores faltantes por mediana de cada maquina
# -----------------------------------------------------------------------------
n_imp_sensor = {}
for c in COLS_SENSOR:
    antes = int(df[c].isna().sum())
    df[c] = df.groupby("maquina_id")[c].transform(lambda s: s.fillna(s.median()))
    # red de seguridad: si toda la maquina estuviera vacia, usar mediana global
    df[c] = df[c].fillna(df[c].median())
    n_imp_sensor[c] = antes
log_problema(
    "Valores faltantes en lecturas de sensores",
    "NaN remanentes en temperatura/presion/vibracion/potencia",
    "Imputacion con la MEDIANA POR MAQUINA (respeta la linea base de cada equipo)",
    f"Imputados por columna: {n_imp_sensor}",
)

# -----------------------------------------------------------------------------
# PROBLEMA 10: Formato inconsistente en 'lote_produccion'
# -----------------------------------------------------------------------------
# Aparecen LOTx001, LOT/001 ademas de LOT-001. Se normaliza a 'LOT-###'.
n_lote = int(df["lote_produccion"].notna().sum() -
             df["lote_produccion"].fillna("").str.match(r"^LOT-\d{3}$").sum())
df["lote_produccion"] = (
    df["lote_produccion"].str.replace(r"^LOT[\W_xX]*(\d{3})$", r"LOT-\1", regex=True)
)
df["lote_produccion"] = df["lote_produccion"].fillna("DESCONOCIDO")
log_problema(
    "Formato inconsistente en 'lote_produccion'",
    "Separadores variados: 'LOTx001', 'LOT/001' frente al estandar 'LOT-001'",
    "Normalizacion con regex a 'LOT-###'; faltantes -> 'DESCONOCIDO'",
    f"{n_lote} lotes reformateados",
)

# -----------------------------------------------------------------------------
# PROBLEMA 11: Ruido y faltantes en 'tecnico_responsable'
# -----------------------------------------------------------------------------
n_tec = int(df["tecnico_responsable"].isna().sum())
df["tecnico_responsable"] = df["tecnico_responsable"].fillna("Desconocido").str.strip()
log_problema(
    "Ruido / faltantes en 'tecnico_responsable'",
    "Valores 'zzzzzzzz', 'N/A' y celdas vacias",
    "Etiquetado uniforme como 'Desconocido'",
    f"{n_tec} registros etiquetados",
)

# -----------------------------------------------------------------------------
# PROBLEMA 12: Estandarizacion de variables categoricas (inspectedby, estado)
# -----------------------------------------------------------------------------
df["inspectedby"] = df["inspectedby"].fillna("auto").str.lower().str.strip()
df["estado_operativo"] = df["estado_operativo"].str.upper().str.strip()
log_problema(
    "Categorias inconsistentes en 'inspectedby' / 'estado_operativo'",
    "Mayusculas/minusculas mezcladas y un vacio en 'inspectedby'",
    "Estandarizacion de mayusculas/minusculas; faltante de inspectedby -> 'auto'",
    "Toda la columna normalizada",
)

# -----------------------------------------------------------------------------
# PROBLEMA 13: Verificacion de duplicados (exactos y semanticos)
# -----------------------------------------------------------------------------
n_dup_exactos = int(df.drop(columns=["id_sensor"]).duplicated().sum())
df["_min"] = df["fecha_hora"].dt.floor("min")
n_dup_sem = int(df.duplicated(subset=["maquina_id", "_min"]).sum())
df = df.drop(columns="_min")
df = df.drop_duplicates(subset=[c for c in df.columns if c != "id_sensor"])
log_problema(
    "Duplicados exactos y semanticos",
    "Chequeo de filas identicas y de (maquina_id, fecha_hora al minuto)",
    "Se eliminan duplicados si existen; cadencia de muestreo ~22-25 min",
    f"Exactos: {n_dup_exactos} | Semanticos: {n_dup_sem} (ninguno real)",
)

# -----------------------------------------------------------------------------
#  GUARDAR DATASET LIMPIO
# -----------------------------------------------------------------------------
ORDEN = ["id_sensor", "fecha_hora", "maquina_id", "inspectedby",
         "temperatura_c", "presion_psi", "vibracion_mm_s", "potencia_kw",
         "estado_operativo", "tecnico_responsable", "lote_produccion"]
df = df[ORDEN].sort_values("id_sensor").reset_index(drop=True)
df.to_csv(RUTA_LIMPIO, index=False)
print(f"Filas finales: {len(df)} (eliminadas {N_INICIAL - len(df)})")
print(f"Dataset limpio guardado en: {RUTA_LIMPIO}")

# Guardar la tabla de problemas para el reporte
tabla_prob = pd.DataFrame(REGISTRO_PROBLEMAS)
tabla_prob.to_csv(os.path.join(BASE, "tabla_problemas.csv"), index=False)

# =============================================================================
#  COMPONENTE 2 - EDA (Exploratory Data Analysis)
# =============================================================================
NUM = COLS_SENSOR

# ------- Resumen de valores faltantes ANTES vs DESPUES -----------------------
raw = pd.read_csv(RUTA_RAW, sep=";", dtype=str)
raw.columns = [c.strip() for c in raw.columns]
def contar_invalidos_raw(col):
    s = raw[col].astype(str).str.strip()
    nulos = s.isin(TOKENS_NULOS).sum()
    if col in COLS_SENSOR:
        v = pd.to_numeric(s, errors="coerce")
        lo, hi = RANGOS[col]
        fuera = ((v < lo) | (v > hi)).sum()
        return int(nulos + fuera)
    return int(nulos)

cols_comunes = [c for c in df.columns if c in raw.columns]
antes = {c: contar_invalidos_raw(c) for c in cols_comunes}
despues = {c: int(df[c].isna().sum()) for c in cols_comunes}
comp = pd.DataFrame({"Invalidos_crudo": antes, "Faltantes_limpio": despues})

fig, ax = plt.subplots(figsize=(9, 4.5))
comp.plot(kind="bar", ax=ax, color=["#d1495b", "#2a9d8f"])
ax.set_title("Calidad de datos: valores invalidos antes vs. despues de la limpieza")
ax.set_ylabel("N. de celdas")
ax.set_xlabel("")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(f"{DIR_FIG}/01_missing.png", bbox_inches="tight")
plt.close()

# --- (3) Medidas de tendencia central -----------------------------------------
def moda(s):
    m = s.mode()
    return round(m.iloc[0], 2) if len(m) else np.nan

resumen = pd.DataFrame({
    "Media":   df[NUM].mean().round(2),
    "Mediana": df[NUM].median().round(2),
    "Moda":    {c: moda(df[c]) for c in NUM},
    "Desv_std":df[NUM].std().round(2),
    "Min":     df[NUM].min().round(2),
    "Max":     df[NUM].max().round(2),
    "CV_%":   (df[NUM].std() / df[NUM].mean() * 100).round(1),
})
resumen.to_csv(os.path.join(BASE, "tendencia_central.csv"))
print("\nMedidas de tendencia central:\n", resumen)

# --- (4) Histogramas / distribucion -------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, c in zip(axes.ravel(), NUM):
    sns.histplot(df[c], kde=True, ax=ax, color="#3a86ff", bins=12)
    ax.axvline(df[c].mean(), color="#d1495b", ls="--", lw=1.5, label="Media")
    ax.axvline(df[c].median(), color="#2a9d8f", ls=":", lw=1.5, label="Mediana")
    ax.set_title(f"Distribucion de {c}")
    ax.legend(fontsize=8)
fig.suptitle("Histogramas de las variables de sensor (dataset limpio)", y=1.02,
             fontweight="bold")
plt.tight_layout()
plt.savefig(f"{DIR_FIG}/02_histogramas.png", bbox_inches="tight")
plt.close()

# --- (5) Boxplots / evaluacion de outliers ------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
for ax, c in zip(axes, NUM):
    sns.boxplot(y=df[c], ax=ax, color="#8ecae6")
    sns.stripplot(y=df[c], ax=ax, color="#023047", size=3, alpha=0.5)
    ax.set_title(c, fontsize=10)
    ax.set_xlabel("")
fig.suptitle("Diagramas de caja: deteccion de outliers residuales", y=1.03,
             fontweight="bold")
plt.tight_layout()
plt.savefig(f"{DIR_FIG}/03_boxplots.png", bbox_inches="tight")
plt.close()

# --- (6+7) Matriz y mapa de correlacion ---------------------------------------
corr = df[NUM].corr().round(2)
corr.to_csv(os.path.join(BASE, "correlacion.csv"))
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Mapa de correlacion entre variables de sensor")
plt.tight_layout()
plt.savefig(f"{DIR_FIG}/04_correlacion.png", bbox_inches="tight")
plt.close()
print("\nMatriz de correlacion:\n", corr)

# --- (Negocio) Estado operativo y lecturas por maquina ------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
orden_estado = ["OPERATIVO", "MANTENIMIENTO", "FALLADO"]
cnt = df["estado_operativo"].value_counts().reindex(orden_estado)
sns.barplot(x=cnt.index, y=cnt.values, ax=axes[0],
            palette=["#2a9d8f", "#e9c46a", "#d1495b"], hue=cnt.index, legend=False)
for i, v in enumerate(cnt.values):
    axes[0].text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold")
axes[0].set_title("Distribucion del estado operativo")
axes[0].set_ylabel("N. de lecturas")

sns.boxplot(data=df, x="maquina_id", y="temperatura_c", ax=axes[1], color="#8ecae6")
axes[1].set_title("Temperatura por maquina (lineas base distintas)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{DIR_FIG}/05_negocio.png", bbox_inches="tight")
plt.close()

# --- (Negocio) Vibracion vs estado: senal de mantenimiento predictivo --------
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.boxplot(data=df, x="estado_operativo", y="vibracion_mm_s",
            order=orden_estado, ax=ax,
            palette=["#2a9d8f", "#e9c46a", "#d1495b"], hue="estado_operativo", legend=False)
sns.stripplot(data=df, x="estado_operativo", y="vibracion_mm_s",
              order=orden_estado, ax=ax, color="#023047", size=4, alpha=.6)
ax.set_title("Vibracion segun estado operativo")
plt.tight_layout()
plt.savefig(f"{DIR_FIG}/06_vibracion_estado.png", bbox_inches="tight")
plt.close()

# --- Descripcion general (perfil por maquina) ---------------------------------
perfil = (df.groupby("maquina_id")[NUM]
            .mean().round(1)
            .join(df.groupby("maquina_id").size().rename("n_lecturas")))
perfil.to_csv(os.path.join(BASE, "perfil_maquinas.csv"))
print("\nPerfil por maquina:\n", perfil)

print("\n=== EDA COMPLETADO: figuras en", DIR_FIG, "===")
