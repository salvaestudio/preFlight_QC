

# qc_gui_v1.py
# GUI (Tkinter) + motor QC parametrizable
# Requisitos: Python 3.x + opencv-python + numpy
# pip install opencv-python numpy

import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable

import sys

# ---------------- DEFAULTS ----------------
# Determinar la carpeta base para que sea portable
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

DEFAULT_OUT_DIR = BASE_DIR / "output"

def default_params() -> Dict[str, Any]:
    return {
        # I/O
        "OUT_DIR": str(DEFAULT_OUT_DIR),
        "PROXY_LONG_SIDE": 4000,
        "ROI_TOP_N": 40,

        # tiling
        "TILE_SIZE": 320,
        "SUB_GRAF": 160,
        "SUB_FOTO": 160,
        "DEDUPE_DIST_FACTOR": 0.55,

        # classify graf/foto
        "GRAF_GRAD_RATIO_THR": 0.035,
        "GRAF_GRAD_THR": 12,

        # FOTO outliers (baseline por blur)
        "FOTO_LOW_PCTL": 12,
        "W_FOTO_OUTLIER": 0.90,
        "FOTO_MIN_MEAN_GRAD": 9.5,
        "FOTO_MIN_EDGE_RATIO": 0.020,

        # blur
        "USE_NORM_BLUR": 1,
        "NORM_EPS": 1e-6,

        # graf weights
        "W_GRAF_STAIR": 1.6,
        "W_GRAF_BLUR": 0.18,
        "THRESH_GRAF_BLUR": 240,

        # banding
        "W_BANDING": 2.4,
        "BAND_MAX_EDGE_RATIO": 0.02,
        "BAND_MIN_MEAN_GRAD": 0.65,
        "BAND_MAX_MEAN_GRAD": 3.0,
        "BAND_LEVELS_MAX": 22,
        "BAND_FLAT_RUN_MIN": 12,
        "BAND_TOP_PCTL": 92,

        # staircase selection thresholds
        "STAIR_TOP_PCTL": 95,
        "STAIR_MIN_ABS": 0.10,

        # ink gate
        "INK_MIN": 0.01,
        "INK_MAX": 0.28,
        "INK_OTSU_MIN_DELTA": 22,
        "INK_MAX_BBOX_FRAC": 0.45,

        # staircase runs
        "RUN_MIN": 3,
        "RUN_RATIO_MIN": 0.10,

        # heat clamp
        "HEAT_CLAMP_PCTL": 95,
    }

# ---------------- FACTORY PRESETS ----------------
def factory_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "Mixto (recomendado)": {
            "PROXY_LONG_SIDE": 6000,
            "TILE_SIZE": 320,
            "SUB_GRAF": 160,
            "SUB_FOTO": 160,
            "HEAT_CLAMP_PCTL": 95,

            "W_GRAF_STAIR": 1.6,
            "STAIR_TOP_PCTL": 95,
            "STAIR_MIN_ABS": 0.10,

            "W_BANDING": 2.4,
            "BAND_TOP_PCTL": 92,

            "W_FOTO_OUTLIER": 0.90,
            "FOTO_LOW_PCTL": 12,
            "FOTO_MIN_MEAN_GRAD": 9.5,
            "FOTO_MIN_EDGE_RATIO": 0.020,
        },
        "Foto (blur/ampliación)": {
            "PROXY_LONG_SIDE": 7000,
            "TILE_SIZE": 240,
            "SUB_FOTO": 120,
            "SUB_GRAF": 160,
            "HEAT_CLAMP_PCTL": 95,

            "W_FOTO_OUTLIER": 1.6,
            "FOTO_LOW_PCTL": 18,
            "FOTO_MIN_MEAN_GRAD": 7.5,
            "FOTO_MIN_EDGE_RATIO": 0.014,

            "W_GRAF_STAIR": 1.0,
            "W_BANDING": 1.6,
        },
        "Gráfico (logos/texto/aliasing)": {
            "PROXY_LONG_SIDE": 6000,
            "TILE_SIZE": 320,
            "SUB_GRAF": 120,
            "SUB_FOTO": 160,
            "HEAT_CLAMP_PCTL": 95,

            "W_GRAF_STAIR": 2.2,
            "STAIR_TOP_PCTL": 93,
            "STAIR_MIN_ABS": 0.08,
            "RUN_MIN": 3,
            "RUN_RATIO_MIN": 0.10,

            "W_BANDING": 2.2,
            "BAND_TOP_PCTL": 92,

            "W_FOTO_OUTLIER": 0.7,
            "FOTO_LOW_PCTL": 12,
        },
    }

# ---------------- ENGINE ----------------
def resize_long_side(img: np.ndarray, target_long: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = target_long / max(h, w)
    if scale >= 1.0:
        return img, 1.0
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale

def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def get_grad_mag(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def get_edge_map(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 80, 160)

def classify_tile(tile_mag: np.ndarray, P: Dict[str, Any]) -> str:
    ratio = float(np.mean(tile_mag > P["GRAF_GRAD_THR"]))
    return "GRAFICO" if ratio > P["GRAF_GRAD_RATIO_THR"] else "FOTO"

def blur_score(tile_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(tile_gray, cv2.CV_64F).var())

def norm_blur(tile_gray: np.ndarray, tile_mag: np.ndarray, tile_edges: np.ndarray, P: Dict[str, Any]) -> float:
    b = blur_score(tile_gray)
    er = float(np.mean(tile_edges > 0))
    mg = float(np.mean(tile_mag))
    eps = float(P["NORM_EPS"])
    eps = eps if eps > 0 else 1e-6
    denom = (0.6 * er + 0.4 * (mg / 20.0)) + eps
    return float(b / denom)

def ink_mask_and_stats(tile_gray: np.ndarray) -> Tuple[np.ndarray, float, float]:
    g = cv2.GaussianBlur(tile_gray, (3, 3), 0)
    thr, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.mean(bw == 255)
    ink = bw if white_ratio < 0.5 else (255 - bw)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    ink_ratio = float(np.mean(ink > 0))
    m0 = float(np.mean(g[ink == 0])) if np.any(ink == 0) else 0.0
    m1 = float(np.mean(g[ink > 0])) if np.any(ink > 0) else 255.0
    delta = abs(m1 - m0)
    return ink, ink_ratio, delta

def staircase_score(tile_gray: np.ndarray, P: Dict[str, Any]) -> float:
    ink, ink_ratio, delta = ink_mask_and_stats(tile_gray)

    if ink_ratio < P["INK_MIN"] or ink_ratio > P["INK_MAX"]:
        return 0.0
    if delta < P["INK_OTSU_MIN_DELTA"]:
        return 0.0

    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0

    c0 = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c0)
    bbox_frac = (w * h) / float(tile_gray.shape[0] * tile_gray.shape[1])
    if bbox_frac > P["INK_MAX_BBOX_FRAC"]:
        return 0.0

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
    long_run_px = 0
    total_px = 0
    run_min = int(P["RUN_MIN"])

    for c in cnts:
        pts = c[:, 0, :]
        if len(pts) < 120:
            continue

        d = np.diff(pts, axis=0)
        dx, dy = d[:, 0], d[:, 1]

        run = 0
        run_axis = None

        for i in range(len(dx)):
            is_h = (dy[i] == 0 and dx[i] != 0)
            is_v = (dx[i] == 0 and dy[i] != 0)

            if is_h:
                if run_axis == 'h':
                    run += 1
                else:
                    if run_axis is not None and run >= run_min:
                        long_run_px += run
                    run_axis, run = 'h', 1
            elif is_v:
                if run_axis == 'v':
                    run += 1
                else:
                    if run_axis is not None and run >= run_min:
                        long_run_px += run
                    run_axis, run = 'v', 1
            else:
                if run_axis is not None and run >= run_min:
                    long_run_px += run
                run_axis, run = None, 0

        if run_axis is not None and run >= run_min:
            long_run_px += run
        total_px += len(dx)

    if total_px == 0:
        return 0.0
    ratio = float(long_run_px / total_px)
    return ratio if ratio >= P["RUN_RATIO_MIN"] else 0.0

def banding_score(tile_gray: np.ndarray, tile_mag: np.ndarray, tile_edges: np.ndarray, P: Dict[str, Any]) -> float:
    er = float(np.mean(tile_edges > 0))
    if er > P["BAND_MAX_EDGE_RATIO"]:
        return 0.0

    gm = float(np.mean(tile_mag))
    if gm < P["BAND_MIN_MEAN_GRAD"] or gm > P["BAND_MAX_MEAN_GRAD"]:
        return 0.0

    s = cv2.GaussianBlur(tile_gray, (5, 5), 0)
    q = (s // 4).astype(np.uint8)
    levels = int(np.unique(q).size)
    level_pen = max(0.0, (P["BAND_LEVELS_MAX"] - levels) / P["BAND_LEVELS_MAX"])

    h, w = s.shape
    row = s[h // 2, :].astype(np.int16)
    col = s[:, w // 2].astype(np.int16)

    def max_flat_run(arr):
        d = np.abs(np.diff(arr))
        flat = d <= 1
        maxrun = 0
        run = 0
        for v in flat:
            if v:
                run += 1
                maxrun = max(maxrun, run)
            else:
                run = 0
        return maxrun

    run = max(max_flat_run(row), max_flat_run(col))
    run_pen = max(0.0, (run - P["BAND_FLAT_RUN_MIN"]) / (max(1, w // 6)))
    return max(0.0, (level_pen * 1.2) + (run_pen * 1.0))

def iterate_tiles(h, w, size):
    """Yield tile top-left coords; covers image borders even when size doesn't divide W/H."""
    if size <= 0:
        return
    last_y = max(0, h - size)
    last_x = max(0, w - size)

    for y in range(0, h, size):
        y0 = min(y, last_y)
        for x in range(0, w, size):
            x0 = min(x, last_x)
            yield x0, y0

def clamp_and_colormap(hm, P):
    hm = hm.copy()
    vals = hm[hm > 0]
    if vals.size == 0:
        u8 = np.zeros_like(hm, dtype=np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    cap = float(np.percentile(vals, P["HEAT_CLAMP_PCTL"]))
    cap = max(1e-6, cap)
    hm = np.clip(hm, 0, cap) / cap
    u8 = (hm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)

def run_qc(img_path, P, status_cb=None):
    out_dir = Path(P["OUT_DIR"])
    roi_dir = out_dir / "rois"
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    if status_cb: status_cb("Cargando imagen...")
    
    # --- CARGA ROBUSTA MULTINIVEL ---
    try:
        # 1. Intentamos carga completa (respetando 16-bit y canales originales)
        img_full_src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        load_mode = "UNCHANGED"

        # 2. Reintento con flags combinados (a veces ayuda con ciertos TIFFs)
        if img_full_src is None:
            img_full_src = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            load_mode = "ANY_STRICT"

        # 3. Fallback a 8-bit COLOR (modo estándar)
        if img_full_src is None:
            img_full_src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            load_mode = "8BIT_COLOR"

        # 4. Fallback a 8-bit GRAYSCALE (el modo más compatible de OpenCV para TIFFs complejos)
        if img_full_src is None:
            img_full_src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            load_mode = "8BIT_GRAY"

    except Exception:
        # Algunos TIFFs corruptos o con demasiados canales lanzan excepción de C++ en imread
        img_full_src = None
        if status_cb: status_cb("Error interno del decodificador...")

    if img_full_src is None:
        raise ValueError(
            f"OpenCV no puede leer este TIFF específico.\n\n"
            f"Sugerencia: Este archivo contiene metadatos o canales extra (5+) "
            f"que el motor de OpenCV no soporta.\n"
            f"Por favor, conviértelo a PNG de 16-bits o TIFF estándar (sin canales extra)."
        )

    if status_cb:
        msg = f"Imagen cargada (Modo {load_mode})."
        if load_mode == "8BIT_GRAY": msg += " Usada escala de grises para compatibilidad."
        status_cb(msg)

    # Normaliza a BGR para análisis; preserva original (p.ej. 16-bit) para recortar ROIs.
    if img_full_src.dtype == np.uint16:
        if status_cb: status_cb("Imagen 16-bit detectada. Análisis en 8-bit (modo conservador).")
        img_full = (img_full_src / 257).astype(np.uint8)
    else:
        img_full = img_full_src.copy()

    # Aseguramos que img_full sea BGR (3 canales) para el análisis y el overlay final
    if img_full.ndim == 2:
        img_full = cv2.cvtColor(img_full, cv2.COLOR_GRAY2BGR)
    elif img_full.shape[2] == 4:
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGRA2BGR)
    elif img_full.shape[2] > 4:
        # Caso raro de imágenes con más canales (p.ej. CMYK cargado como tal), forzamos a 3
        img_full = img_full[:, :, :3]

    img_proxy, scale = resize_long_side(img_full, int(P["PROXY_LONG_SIDE"]))
    gray = to_gray(img_proxy)
    H, W = gray.shape

    if status_cb: status_cb("Pre-computando mapas de métricas...")
    mag = get_grad_mag(gray)
    edges = get_edge_map(gray)

    TILE = int(P["TILE_SIZE"])
    SUB_G = int(P["SUB_GRAF"])
    SUB_F = int(P["SUB_FOTO"])
    TOP_N = int(P["ROI_TOP_N"])
    DEDUPE_DIST = max(1, int(TILE * float(P["DEDUPE_DIST_FACTOR"])))

    heat_total = np.zeros((H, W), dtype=np.float32)
    heat_stair = np.zeros((H, W), dtype=np.float32)
    heat_foto  = np.zeros((H, W), dtype=np.float32)
    heat_band  = np.zeros((H, W), dtype=np.float32)

    meta = []
    foto_blurs = []
    band_vals = []
    stair_vals = []

    if status_cb: status_cb("Primera pasada: analizando distribuciones...")

    # PRE-PASS (baseline de foto SIN depender del clasificador)
    for x, y in iterate_tiles(H, W, TILE):
        t_gray = gray[y:y+TILE, x:x+TILE]
        t_mag = mag[y:y+TILE, x:x+TILE]
        t_edges = edges[y:y+TILE, x:x+TILE]

        ttype = classify_tile(t_mag, P)
        b = norm_blur(t_gray, t_mag, t_edges, P) if int(P["USE_NORM_BLUR"]) else blur_score(t_gray)
        bd = banding_score(t_gray, t_mag, t_edges, P)
        st = staircase_score(t_gray, P)

        band_vals.append(bd)
        stair_vals.append(st)
        meta.append((x, y, ttype, b, bd, st))

        mg_val = float(np.mean(t_mag))
        er_val = float(np.mean(t_edges > 0))

        if mg_val >= float(P["FOTO_MIN_MEAN_GRAD"]) and er_val >= float(P["FOTO_MIN_EDGE_RATIO"]):
            foto_blurs.append(b)

            for sx, sy in iterate_tiles(TILE, TILE, SUB_F):
                s_mag = t_mag[sy:sy+SUB_F, sx:sx+SUB_F]
                s_edges = t_edges[sy:sy+SUB_F, sx:sx+SUB_F]
                if float(np.mean(s_mag)) >= float(P["FOTO_MIN_MEAN_GRAD"]) and float(np.mean(s_edges > 0)) >= float(P["FOTO_MIN_EDGE_RATIO"]):
                    s_gray = t_gray[sy:sy+SUB_F, sx:sx+SUB_F]
                    sb = norm_blur(s_gray, s_mag, s_edges, P) if int(P["USE_NORM_BLUR"]) else blur_score(s_gray)
                    foto_blurs.append(sb)

    foto_cut  = float(np.percentile(np.array(foto_blurs, dtype=np.float32), int(P["FOTO_LOW_PCTL"]))) if foto_blurs else 0.0
    band_cut  = float(np.percentile(np.array(band_vals, dtype=np.float32), int(P["BAND_TOP_PCTL"]))) if band_vals else 0.0

    if stair_vals:
        stair_p = float(np.percentile(np.array(stair_vals, dtype=np.float32), int(P["STAIR_TOP_PCTL"])))
    else:
        stair_p = 0.0
    stair_cut = max(stair_p, float(P["STAIR_MIN_ABS"]))

    if status_cb: status_cb("Segunda pasada: generando mapas de calor...")

    records = []
    for x, y, ttype, b_tile, bd_tile, st_tile in meta:
        t_gray = gray[y:y+TILE, x:x+TILE]
        t_mag = mag[y:y+TILE, x:x+TILE]
        t_edges = edges[y:y+TILE, x:x+TILE]

        # banding
        bd = bd_tile if bd_tile >= band_cut else 0.0
        if bd > 0:
            heat_band[y:y+TILE, x:x+TILE] = np.maximum(heat_band[y:y+TILE, x:x+TILE], bd)
        comp_band = float(P["W_BANDING"]) * bd

        # foto outliers (SIN depender del clasificador)
        comp_foto_tile = 0.0
        mg_tile = float(np.mean(t_mag))
        er_tile = float(np.mean(t_edges > 0))

        if foto_cut > 0 and mg_tile >= float(P["FOTO_MIN_MEAN_GRAD"]) and er_tile >= float(P["FOTO_MIN_EDGE_RATIO"]):
            r = max(0.0, foto_cut - b_tile) * float(P["W_FOTO_OUTLIER"])
            if r > 0:
                heat_foto[y:y+TILE, x:x+TILE] = np.maximum(heat_foto[y:y+TILE, x:x+TILE], r)
                comp_foto_tile = r
                records.append((x, y, TILE, float(max(r, comp_band)), "FOTO"))

            for sx, sy in iterate_tiles(TILE, TILE, SUB_F):
                s_mag = t_mag[sy:sy+SUB_F, sx:sx+SUB_F]
                s_edges = t_edges[sy:sy+SUB_F, sx:sx+SUB_F]
                if float(np.mean(s_mag)) < float(P["FOTO_MIN_MEAN_GRAD"]) or float(np.mean(s_edges > 0)) < float(P["FOTO_MIN_EDGE_RATIO"]):
                    continue

                s_gray = t_gray[sy:sy+SUB_F, sx:sx+SUB_F]
                sb = norm_blur(s_gray, s_mag, s_edges, P) if int(P["USE_NORM_BLUR"]) else blur_score(s_gray)
                sr = max(0.0, foto_cut - sb) * float(P["W_FOTO_OUTLIER"])
                if sr > 0:
                    gx, gy = x + sx, y + sy
                    heat_foto[gy:gy+SUB_F, gx:gx+SUB_F] = np.maximum(heat_foto[gy:gy+SUB_F, gx:gx+SUB_F], sr)
                    comp = max(sr, comp_band)
                    heat_total[gy:gy+SUB_F, gx:gx+SUB_F] = np.maximum(heat_total[gy:gy+SUB_F, gx:gx+SUB_F], comp)
                    records.append((gx, gy, SUB_F, float(comp), "FOTO"))

        if ttype == "GRAFICO":
            st = st_tile if st_tile >= stair_cut else 0.0
            comp_stair = float(P["W_GRAF_STAIR"]) * st

            r_blur = max(0.0, float(P["THRESH_GRAF_BLUR"]) - blur_score(t_gray))
            comp_blur = float(P["W_GRAF_BLUR"]) * r_blur

            comp_total = max(comp_band, comp_stair, comp_blur, comp_foto_tile)
            if comp_total > 0:
                if st > 0:
                    heat_stair[y:y+TILE, x:x+TILE] = np.maximum(heat_stair[y:y+TILE, x:x+TILE], st)
                heat_total[y:y+TILE, x:x+TILE] = np.maximum(heat_total[y:y+TILE, x:x+TILE], comp_total)
                records.append((x, y, TILE, float(comp_total), "GRAF"))

            for sx, sy in iterate_tiles(TILE, TILE, SUB_G):
                s_gray = t_gray[sy:sy+SUB_G, sx:sx+SUB_G]
                s_mag = t_mag[sy:sy+SUB_G, sx:sx+SUB_G]
                s_edges = t_edges[sy:sy+SUB_G, sx:sx+SUB_G]

                sb = blur_score(s_gray)
                sbd = banding_score(s_gray, s_mag, s_edges, P)
                sbd = sbd if sbd >= band_cut else 0.0
                comp_b = float(P["W_BANDING"]) * sbd

                sst = staircase_score(s_gray, P)
                sst = sst if sst >= stair_cut else 0.0
                comp_s = float(P["W_GRAF_STAIR"]) * sst

                sr_blur = max(0.0, float(P["THRESH_GRAF_BLUR"]) - sb)
                comp_l = float(P["W_GRAF_BLUR"]) * sr_blur

                stotal = max(comp_b, comp_s, comp_l)
                if stotal > 0:
                    gx, gy = x + sx, y + sy
                    if sst > 0:
                        heat_stair[gy:gy+SUB_G, gx:gx+SUB_G] = np.maximum(heat_stair[gy:gy+SUB_G, gx:gx+SUB_G], sst)
                    if sbd > 0:
                        heat_band[gy:gy+SUB_G, gx:gx+SUB_G] = np.maximum(heat_band[gy:gy+SUB_G, gx:gx+SUB_G], sbd)
                    heat_total[gy:gy+SUB_G, gx:gx+SUB_G] = np.maximum(heat_total[gy:gy+SUB_G, gx:gx+SUB_G], stotal)
                    records.append((gx, gy, SUB_G, float(stotal), "GRAF"))
        else:
            comp_total = max(comp_band, comp_foto_tile)
            if comp_total > 0:
                heat_total[y:y+TILE, x:x+TILE] = np.maximum(heat_total[y:y+TILE, x:x+TILE], comp_total)

    if status_cb: status_cb("Guardando resultados...")

    total_c = clamp_and_colormap(heat_total, P)
    stair_c = clamp_and_colormap(heat_stair, P)
    foto_c  = clamp_and_colormap(heat_foto, P)
    band_c  = clamp_and_colormap(heat_band, P)
    overlay = cv2.addWeighted(img_proxy, 0.62, total_c, 0.38, 0)

    cv2.imwrite(str(out_dir / "heat_total.png"), total_c)
    cv2.imwrite(str(out_dir / "heat_staircase.png"), stair_c)
    cv2.imwrite(str(out_dir / "heat_foto_outliers.png"), foto_c)
    cv2.imwrite(str(out_dir / "heat_banding.png"), band_c)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    records.sort(key=lambda r: r[3], reverse=True)
    picked = []
    for r in records:
        if len(picked) >= TOP_N:
            break
        if all(abs(r[0] - p[0]) >= DEDUPE_DIST or abs(r[1] - p[1]) >= DEDUPE_DIST for p in picked):
            picked.append(r)

    for i, (x, y, size, risk, kind) in enumerate(picked, start=1):
        fx, fy, fs = int(round(x/scale)), int(round(y/scale)), int(round(size/scale))
        fx = max(0, min(fx, img_full_src.shape[1]-1))
        fy = max(0, min(fy, img_full_src.shape[0]-1))
        fs = max(1, min(fs, min(img_full_src.shape[1]-fx, img_full_src.shape[0]-fy)))
        crop = img_full_src[fy:fy+fs, fx:fx+fs]
        cv2.imwrite(str(roi_dir / f"roi_{i:02d}_{kind}_R{risk:.2f}.png"), crop)

    return {
        "proxy": (W, H),
        "out_dir": str(out_dir),
        "foto_cut": float(foto_cut),
        "band_cut": float(band_cut),
        "stair_cut": float(stair_cut),
        "rois": len(picked),
    }

# ---------------- GUI ----------------
class Slider:
    def __init__(self, parent, label, key, vmin, vmax, step, var_type=float):
        self.key = key
        self.var_type = var_type
        self.var = tk.DoubleVar() if var_type is float else tk.IntVar()

        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)

        ttk.Label(row, text=label, width=22).pack(side="left")

        self.scale = tk.Scale(
            row, from_=vmin, to=vmax, resolution=step,
            orient="horizontal", showvalue=False, length=330,
            variable=self.var, command=self._sync_label
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=6)

        self.value_lbl = ttk.Label(row, width=10, anchor="e")
        self.value_lbl.pack(side="right")
        self._sync_label(self.var.get())

    def _sync_label(self, val):
        v = float(val)
        if self.var_type is int:
            v = int(round(v))
            self.value_lbl.config(text=str(v))
        else:
            self.value_lbl.config(text=f"{v:.3f}")

    def set(self, v):
        self.var.set(v)
        self._sync_label(v)

    def get(self):
        v = self.var.get()
        return int(round(v)) if self.var_type is int else float(v)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("preFlight QC")
        self.geometry("860x680")

        self.params = default_params()
        self.img_path = tk.StringVar(value="")

        self.factory = factory_presets()
        self.factory_name = tk.StringVar(value="Mixto (recomendado)")

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Button(top, text="Elegir imagen", command=self.pick_image).pack(side="left")
        ttk.Label(top, textvariable=self.img_path).pack(side="left", padx=10)

        outrow = ttk.Frame(self)
        outrow.pack(fill="x", padx=10, pady=4)
        ttk.Label(outrow, text="Output").pack(side="left")
        self.out_var = tk.StringVar(value=self.params["OUT_DIR"])
        ttk.Entry(outrow, textvariable=self.out_var, width=65).pack(side="left", padx=8)
        ttk.Button(outrow, text="Cambiar", command=self.pick_out).pack(side="left")

        btnrow = ttk.Frame(self)
        btnrow.pack(fill="x", padx=10, pady=6)

        ttk.Button(btnrow, text="Guardar preset", command=self.save_preset).pack(side="left")
        ttk.Button(btnrow, text="Cargar preset", command=self.load_preset).pack(side="left", padx=8)

        ttk.Label(btnrow, text="Preset fábrica").pack(side="left", padx=(14, 6))
        self.factory_combo = ttk.Combobox(
            btnrow,
            textvariable=self.factory_name,
            values=list(self.factory.keys()),
            state="readonly",
            width=24
        )
        self.factory_combo.pack(side="left")
        ttk.Button(btnrow, text="Aplicar", command=self.apply_factory_preset).pack(side="left", padx=6)

        ttk.Button(btnrow, text="Cerrar preFlight QC", command=self.destroy).pack(side="right")
        ttk.Button(btnrow, text="RUN", command=self.run).pack(side="right", padx=8)

        self.status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status).pack(fill="x", padx=10, pady=6)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        tab_global = ttk.Frame(nb)
        tab_band = ttk.Frame(nb)
        tab_stair = ttk.Frame(nb)
        tab_foto = ttk.Frame(nb)
        nb.add(tab_global, text="Global")
        nb.add(tab_band, text="Banding")
        nb.add(tab_stair, text="Staircase")
        nb.add(tab_foto, text="Foto")

        self.sliders = {}

        # Global
        self.add_slider(tab_global, "Proxy long side", "PROXY_LONG_SIDE", 2000, 8000, 100, int)
        self.add_slider(tab_global, "Tile size", "TILE_SIZE", 160, 640, 16, int)
        self.add_slider(tab_global, "Sub GRAF", "SUB_GRAF", 80, 320, 8, int)
        self.add_slider(tab_global, "Sub FOTO", "SUB_FOTO", 32, 320, 8, int)
        self.add_slider(tab_global, "Heat clamp pctl", "HEAT_CLAMP_PCTL", 85, 99, 1, int)

        # Banding
        self.add_slider(tab_band, "W banding", "W_BANDING", 0.5, 5.0, 0.05, float)
        self.add_slider(tab_band, "Band top pctl", "BAND_TOP_PCTL", 85, 99, 1, int)
        self.add_slider(tab_band, "Max edge ratio", "BAND_MAX_EDGE_RATIO", 0.005, 0.20, 0.001, float)
        self.add_slider(tab_band, "Min mean grad", "BAND_MIN_MEAN_GRAD", 0.10, 3.0, 0.05, float)
        self.add_slider(tab_band, "Max mean grad", "BAND_MAX_MEAN_GRAD", 1.0, 12.0, 0.1, float)
        self.add_slider(tab_band, "Levels max", "BAND_LEVELS_MAX", 8, 80, 1, int)
        self.add_slider(tab_band, "Flat run min", "BAND_FLAT_RUN_MIN", 4, 40, 1, int)

        # Staircase
        self.add_slider(tab_stair, "W graf stair", "W_GRAF_STAIR", 0.2, 4.0, 0.05, float)
        self.add_slider(tab_stair, "Stair top pctl", "STAIR_TOP_PCTL", 85, 99, 1, int)
        self.add_slider(tab_stair, "Stair min abs", "STAIR_MIN_ABS", 0.01, 0.40, 0.01, float)
        self.add_slider(tab_stair, "Run min", "RUN_MIN", 2, 8, 1, int)
        self.add_slider(tab_stair, "Run ratio min", "RUN_RATIO_MIN", 0.02, 0.30, 0.01, float)
        self.add_slider(tab_stair, "Ink min", "INK_MIN", 0.001, 0.08, 0.001, float)
        self.add_slider(tab_stair, "Ink max", "INK_MAX", 0.10, 0.55, 0.01, float)
        self.add_slider(tab_stair, "Otsu delta min", "INK_OTSU_MIN_DELTA", 8, 60, 1, int)
        self.add_slider(tab_stair, "Max bbox frac", "INK_MAX_BBOX_FRAC", 0.10, 0.80, 0.01, float)

        # Foto
        self.add_slider(tab_foto, "Foto low pctl", "FOTO_LOW_PCTL", 1, 70, 1, int)
        self.add_slider(tab_foto, "W foto outlier", "W_FOTO_OUTLIER", 0.1, 5.0, 0.05, float)
        self.add_slider(tab_foto, "Min mean grad", "FOTO_MIN_MEAN_GRAD", 0.5, 25.0, 0.5, float)
        self.add_slider(tab_foto, "Min edge ratio", "FOTO_MIN_EDGE_RATIO", 0.001, 0.20, 0.001, float)

        self.apply_params_to_ui()

    def add_slider(self, parent, label, key, vmin, vmax, step, var_type):
        self.sliders[key] = Slider(parent, label, key, vmin, vmax, step, var_type)

    def apply_params_to_ui(self):
        for k, s in self.sliders.items():
            if k in self.params:
                s.set(self.params[k])

    def ui_to_params(self):
        P = dict(self.params)
        P["OUT_DIR"] = self.out_var.get().strip() or str(DEFAULT_OUT_DIR)
        for k, s in self.sliders.items():
            P[k] = s.get()
        P["USE_NORM_BLUR"] = int(P.get("USE_NORM_BLUR", 1))
        return P

    def apply_factory_preset(self):
        name = self.factory_name.get()
        overrides = self.factory.get(name, {})
        base = default_params()
        base.update(overrides)
        base["OUT_DIR"] = self.out_var.get().strip() or base["OUT_DIR"]

        self.params = base
        self.out_var.set(self.params.get("OUT_DIR", str(DEFAULT_OUT_DIR)))
        self.apply_params_to_ui()
        self.status.set(f"Preset fábrica aplicado: {name}")

    def pick_image(self):
        fp = filedialog.askopenfilename(
            title="Selecciona imagen",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"), ("All", "*.*")]
        )
        if fp:
            self.img_path.set(fp)

    def pick_out(self):
        d = filedialog.askdirectory(title="Selecciona carpeta de output")
        if d:
            self.out_var.set(d)

    def save_preset(self):
        P = self.ui_to_params()
        fp = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not fp:
            return
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(P, f, indent=2)
        self.status.set(f"Preset guardado: {fp}")

    def load_preset(self):
        fp = filedialog.askopenfilename(title="Cargar preset", filetypes=[("JSON", "*.json")])
        if not fp:
            return
        with open(fp, "r", encoding="utf-8") as f:
            P = json.load(f)
        self.params.update(P)
        self.out_var.set(self.params.get("OUT_DIR", str(DEFAULT_OUT_DIR)))
        self.apply_params_to_ui()
        self.status.set(f"Preset cargado: {fp}")

    def run(self):
        img_path = self.img_path.get().strip()
        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("Error", "Selecciona una imagen válida.")
            return

        P = self.ui_to_params()

        def update_status(text: str):
            self.status.set(text)
            self.update_idletasks()

        try:
            info = run_qc(img_path, P, status_cb=update_status)
            self.status.set(
                f"OK | {info['proxy'][0]}x{info['proxy'][1]} | "
                f"foto_cut={info['foto_cut']:.2f} | b_cut={info['band_cut']:.2f} | "
                f"s_cut={info['stair_cut']:.2f} | rois={info['rois']}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))
            self.status.set(f"Error: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
