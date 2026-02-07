

## PREFLIGHT QC (Quality Control) — detector de riesgos visuales antes de imprenta


Requisitos: Python 3.x + opencv-python + numpy
 pip install opencv-python numpy

Testado con Windows 11
 
### Qué hace

Este programa analiza una imagen final (keyart/cartel/poster) y genera mapas de calor (heatmaps) que señalan **zonas con alta probabilidad de problemas de calidad**, típicos cuando hay:

- elementos **ampliados** por encima de su resolución real,
    
- bordes con **aliasing** (“dientes de sierra”),
    
- **banding** (escalonado) en degradados,
    
- áreas fotográficas con pérdida de detalle relativa respecto a otras zonas.
    

El objetivo no es “aprobar o suspender”, sino **dirigir la revisión humana** a los puntos con más riesgo.

---

## Cómo funciona (idea general)

1. **Proxy de análisis**  
    La imagen se reescala a un tamaño manejable (por defecto 4000 px lado largo).  
    Esto permite iterar rápido. Si hace falta ver defectos finos, se sube el proxy.
    
2. **Análisis por bloques (tiles)**  
    La imagen se divide en “cuadros” (tiles).  
    En cada tile se calculan métricas y se clasifican patrones.
    
3. **Detectores especializados**
    
    - Banding (degradados)
        
    - Staircase (aliasing/contornos)
        
    - Foto outliers (zonas fotográficas “blandas”)
        
4. **Fusión y visualización**
    
    - `heat_total` combina los detectores con pesos.
        
    - `overlay` superpone el heatmap sobre la imagen.
        
    - Se exportan también mapas separados y recortes ROI.
        

---

## Qué defectos intenta detectar y con qué pistas

### A) Banding en degradados (`heat_banding`)

Problema: un degradado “debería” ser suave, pero aparecen escalones.

Pistas que se usan:

- La zona debe tener **pocos bordes** (si hay muchos bordes suele ser textura/foto).
    
- Debe existir gradiente “moderado” (ni plano ni con mucho detalle).
    
- Se mide:
    
    - **pocos niveles tonales efectivos** (cuantización),
        
    - **tramos largos casi constantes** a lo largo de líneas (flat runs),
        
    - se filtra por percentil para quedarnos con lo peor.
        

Parámetros clave:

- `Band top pctl`, `Max edge ratio`, `Min/Max mean grad`, `Levels max`, `Flat run min`, `W banding`.
    

---

### B) Dientes de sierra / aliasing (`heat_staircase`)

Problema: bordes oblicuos o curvos con escalones (típico en texto/logos ampliados o rasterizados mal).

Pistas que se usan:

- Se crea una máscara de “tinta” (ink) por umbral (Otsu) para aislar contornos.
    
- Se analizan contornos y se buscan **secuencias largas** de pasos horizontales/verticales (runs), típicas del aliasing.
    
- Se filtra para evitar marcar:
    
    - fondos vacíos,
        
    - fotos con textura,
        
    - grandes masas de tinta no informativas.
        

Parámetros clave:

- `Stair top pctl`, `Stair min abs`, `Run min`, `Run ratio min`,  
    `Ink min/max`, `Otsu delta min`, `Max bbox frac`, `W graf stair`.
    

---

### C) Foto ampliada / pérdida de detalle relativa (`heat_foto_outliers`)

Problema: una parte fotográfica está “blanda” (poca micro-textura), comparada con otras fotos de la pieza.

Pistas que se usan:

- Se calcula una medida de “nitidez” (variancia del Laplaciano).
    
- Se compara cada zona de foto contra el conjunto (percentil bajo):
    
    - si una zona está por debajo del corte → outlier.
        
- Hay filtros para no analizar zonas sin información:
    
    - mínimo gradiente,
        
    - mínimo ratio de bordes.
        

Parámetros clave:

- `Foto low pctl`, `Min mean grad`, `Min edge ratio`, `W foto outlier`.
    

---

## Qué NO hace (limitaciones actuales)

- No “entiende capas” (PSD) ni sabe qué elemento es logo/texto/foto: trabaja sobre el raster final.
    
- No mide DPI real del recurso original: detecta **síntomas visibles** de ampliación (blur relativo, aliasing, banding).
    
- Puede dar falsos positivos en:
    
    - piel muy suave, fondos muy limpios,
        
    - texturas finas que parezcan banding,
        
    - contornos muy contrastados que parezcan staircase.
        

Por eso hay sliders: para ajustar según el tipo de arte final.

---

## Outputs

- `overlay.png`: imagen + heat total.
    
- `heat_total.png`: mapa combinado.
    
- `heat_banding.png`: solo banding.
    
- `heat_staircase.png`: solo aliasing.
    
- `heat_foto_outliers.png`: solo foto outliers.
    
- `rois/roi_*.png`: recortes de las zonas con mayor score para revisión rápida.
    

---

## Recomendación de uso en producción

- Tener **2–3 presets**:
    
    1. Keyart foto con poco texto
        
    2. Cartel con mucho texto/logos
        
    3. Piezas con degradados y fondos limpios
        
- Analizar en proxy 4000 para iterar. Subir a 6000–8000 solo si buscas fallos finos.
    
- Usar el heatmap como **“radar”**: prioriza revisión humana en esas zonas.
    

---
