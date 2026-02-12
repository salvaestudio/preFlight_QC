

#work #LabCorporativa 

# README / Memoria explicativa


 
## PREFLIGHT QC (Quality Control) — detector de riesgos visuales antes de imprenta


Requisitos: Python 3.x + opencv-python + numpy
 pip install opencv-python numpy


---

  

## 🛠 Instalación y Requisitos

*   Python 3.x

*   `pip install opencv-python numpy`

*   No requiere librerías adicionales (como Pillow), lo que facilita compartirlo entre compañeros.

---
 
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

## **Soporte de imágenes 16-bit (modo conservador)**  

El sistema detecta automáticamente imágenes de 16-bit (por ejemplo TIFF) y las convierte internamente a 8-bit **solo para el análisis de calidad**. Este comportamiento es intencionado y responde a un enfoque de _preflight conservador_: la reducción de profundidad tonal actúa como un test de estrés que puede hacer aflorar banding, micro-blur o problemas latentes que, aunque poco visibles en 16-bit ideal, podrían manifestarse tras conversiones de color, reprocesos o flujos de imprenta agresivos. La imagen original se conserva sin degradación para el recorte y guardado de ROIs.

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

Esta versión v24 incluye mejoras críticas para el manejo de imágenes de **16 bits** y una gestión robusta de archivos complejos.

  

## 🚀 Mejoras de la v24 vs v23

  

1.  **Preservación de Calidad**: A diferencia de la v23, esta versión mantiene la profundidad de **16 bits** original para el recorte de ROIs (siempre que el formato lo soporte).

2.  **Análisis de Estrés**: El análisis de calidad se realiza internamente a **8 bits**. Esto es intencionado: al reducir la profundidad tonal, afloran más fácilmente problemas de **banding, micro-blur y aliasing** que podrían pasar desapercibidos en 16 bits pero causar problemas en imprenta.

3.  **Cargador Multinivel**: Si un archivo TIFF complejo falla al cargar, el script intenta automáticamente varios modos de compatibilidad (Color 8-bit, Grayscale) para evitar errores.

  

## 📊 Matriz de Compatibilidad

| Formato  | Profundidad | Espacio Color | Estado                                |
| :------- | :---------- | :------------ | :------------------------------------ |
| **PNG**  | 8 / 16 bit  | RGB / CMYK    | **OK** (Recomendado para 16-bit)      |
| **TIFF** | 8 bit       | RGB / CMYK    | **OK**                                |
| **TIFF** | 16 bit      | RGB           | **OK**                                |
| **TIFF** | 16 bit      | CMYK          | **Limitado** (Fallo nativo de OpenCV) |

  

> [!TIP]

> **¿Qué hacer si un TIFF de 16-bit CMYK da error?**

> Conviértelo a **PNG de 16-bit**. El script lo leerá con máxima calidad y detectará todas las "impurezas" sin problemas.


