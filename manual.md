

# Manual de uso — PREFLIGHT QC


## Flujo de trabajo recomendado

1. Seleccionar imagen original (no reducida).
    
2. Ajustar **Proxy long side** según el nivel de detalle necesario.
    
3. Ejecutar con valores base.
    
4. Ajustar **sliders por módulo**, no todos a la vez.
    
5. Interpretar `heat_total` y, si hace falta, revisar mapas individuales.
    
6. Guardar preset cuando el comportamiento sea correcto para ese tipo de pieza.
    

---

## Cómo interpretar los resultados

### `overlay.png`

Mapa de riesgo global superpuesto a la imagen.  
Zonas más intensas = **mayor probabilidad de defecto visual**.

### `heat_total.png`

Mapa combinado.  
Es el **máximo** de los detectores activos (no una suma ciega).

### `heat_foto_outliers.png`

Zonas con **déficit de nitidez relativo** → ampliaciones, fotos “blandas”, pérdida de detalle.

### `heat_staircase.png`

Zonas con **aliasing / dientes de sierra** en contornos.

### `heat_banding.png`

Zonas con **banding** en degradados.

---

## Pestaña GLOBAL

### Proxy long side

Tamaño máximo del proxy de análisis.

- 4000 → iteración rápida.
    
- 6000–8000 → detección fina (textos pequeños, objetos pequeños, globos).
    

> Para validación final, siempre subir a 6000+.

---

### Tile size

Tamaño base de análisis.

- Más grande → más estable, menos ruido, menos detalle.
    
- Más pequeño → más sensible a defectos locales.
    

Valores típicos:

- 320 → uso general.
    
- 160 → detectar objetos pequeños / caras pequeñas.
    

---

### Sub GRAF / Sub FOTO

Subdivisiones internas para análisis fino.

- **Sub FOTO es crítico** para detectar ampliaciones pequeñas.
    
- Para objetos pequeños (globos, logos pequeños, caras lejanas):
    
    - Sub FOTO: **64–80**
        
- Para carteles más limpios:
    
    - Sub FOTO: **120–160**
        

---

### Heat clamp pctl

Controla la **compresión visual** del heatmap.

- Sube si todo se ve rojo.
    
- Baja si todo se ve azul y nada destaca.
    

---

## Pestaña FOTO (Blur / ampliación)

> Este módulo detecta **déficit de nitidez relativo**, no DPI ni resolución real.  
> Compara zonas entre sí.

### W foto

Peso del módulo Foto en el resultado final.

- 0 → desactivado.
    
- 1–2 → uso normal.
    
- 3–5 → análisis forzado / test.
    

---

### Foto k (sensibilidad)

Parámetro más importante del módulo Foto.

- Más bajo → **más sensible** (marca más ampliaciones).
    
- Más alto → más estricto.
    

Valores orientativos:

- 0.10–0.15 → detectar ampliaciones claras.
    
- 0.20–0.30 → solo ampliaciones muy evidentes.
    

---

### Foto pctl low / Foto pctl high

Rango de normalización interna del detector Foto.

- Controla qué parte de la distribución de nitidez se considera “normal”.
    
- Afecta directamente a la **amplitud del mapa**.
    

Valores recomendados:

- Low: **5–10**
    
- High: **80–90**
    

Si no detecta nada → baja Low o High.  
Si detecta demasiado → sube Low.

---

### Foto tex min / Foto tex max

Máscara de textura (protección contra fondos planos).

- Evita marcar fondos lisos, cielos, papeles uniformes.
    
- Permite marcar objetos con algo de estructura.
    

Valores típicos:

- tex min: **0.005–0.010**
    
- tex max: **0.030–0.060**
    

Si no detecta caras u objetos pequeños → baja tex min.  
Si ensucia fondos → sube tex min.

---

## Pestaña STAIRCASE (aliasing / dientes de sierra)

### W graf stair

Peso del detector de aliasing.

- 0.5–1.5 → uso normal.
    
- Subir solo si el texto o logos malos no aparecen.
    

---

### Stair top pctl

Percentil relativo: solo se marcan los peores contornos.

- 95–97 → muy limpio (pocos falsos positivos).
    
- 90–94 → más sensible.
    

Si marca contornos “buenos” → subir.

---

### Stair min abs

Umbral mínimo absoluto del detector.

- Subir para eliminar ruido residual.
    
- Muy útil para limpiar falsos positivos en ropa o siluetas.
    

---

### Run min / Run ratio min

Definen qué se considera “escalón real”.

- Run min: longitud mínima del escalón.
    
- Run ratio min: proporción mínima de escalones en el contorno.
    

Para limpiar falsos:

- Run min → **4–5**
    
- Run ratio min → **0.12–0.18**
    

---

### Ink min / Ink max / Otsu delta / Max bbox frac

Filtros para asegurar que el staircase actúa solo sobre:

- texto,
    
- logos,
    
- gráficos con tinta clara.
    

Si marca caras o masas grandes → bajar Ink max o Max bbox frac.  
Si no marca texto → subir Ink max o bajar Otsu delta.

---

## Pestaña BANDING (degradados)

### W banding

Peso del banding en el resultado final.

---

### Band top pctl

Solo se marcan los degradados más problemáticos.

- 90–93 → sensible.
    
- 95–98 → muy estricto.
    

---

### Max edge ratio

Evita banding en zonas con textura o detalle.

- Si el fondo tiene grano → subir.
    
- Si no detecta banding → bajar.
    

---

### Min / Max mean grad

Rango de gradiente válido para buscar banding.

- Ajustar si marca cosas que no son degradados.
    
- Ajustar si se le escapa un degradado claro.
    

---

### Levels max / Flat run min

Sensibilidad interna al escalonado tonal.

- Levels max bajo → más sensible.
    
- Flat run min bajo → más sensible.
    

---

## Lectura del log de estado (muy importante)

Ejemplo:

```
OK | 6000x3428 | foto_cut=123.4 | b_cut=0.45 | s_cut=0.82 | rois=40
```

- **foto_cut > 0** → módulo Foto activo (bien).
    
- **b_cut > 0** → banding activo.
    
- **s_cut > 0** → staircase activo.
    

Si alguno es **0.00**, ese módulo está muerto y no influye.

---

## Recetas rápidas

### Detectar ampliaciones pequeñas (caras lejanas, globos)

- Proxy ≥ 6000
    
- Tile size = 160
    
- Sub FOTO = 64
    
- Foto k = 0.10
    
- Foto tex min = 0.005
    

---

### Limpiar falsos positivos en fondos

- Subir Foto tex min
    
- Subir Stair top pctl
    
- Subir Run min
    

---

### Cartel con mucho texto y logos

- Subir W graf stair
    
- Bajar Stair top pctl
    
- Subir Otsu delta
    

---

## Filosofía del sistema

PREFLIGHT QC **no decide si una imagen es válida**.  
Señala **zonas con riesgo visual** para acelerar la revisión humana.

Cuanto más específico sea el preset para el tipo de pieza, mejor funciona.

---
