# 🚢 AeroShip AI - Sistema de Detección de Barcos

Este proyecto implementa la solución completa solicitada para la detección de barcos en imágenes aéreas.

## 🚀 Guía Rápida de Uso

1. **Instalar Dependencias**:
   Abre una terminal en esta carpeta (`c:\Users\USER\Desktop\Inteligencia Artificial\1\ship_detection`) y ejecuta:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Preparar el Dataset (Entrenamiento)**:
   - Descarga el dataset de Kaggle: [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery).
   - Extrae el archivo `.zip`.
   - Coloca la carpeta que contiene todas las imágenes `png` (generalmente llamada `shipsnet`) de tal forma que la ruta final quede como `ship_detection/dataset/shipsnet/shipsnet`.

3. **Entrenar el Modelo (>98% Accuracy)**:
   - Ejecuta el script de entrenamiento:
   ```bash
   python backend/train.py
   ```
   - Esto guardará el archivo `ship_model_final.keras`.

4. **Lanzar la Interfaz (UI en Vivo)**:
   - Ejecuta el servidor FastAPI (desde la carpeta `ship_detection/backend/`):
   ```bash
   python main.py
   ```
   - Abre tu navegador en **http://127.0.0.1:8000**.

## 🧪 Pruebas con la UI (Evidencias E1, E3, E4)
1. **Cargar carpeta ciega (E1)**: En la interfaz, usa el botón "Select Image Folder" y selecciona una carpeta con imágenes de prueba (puede ser un subconjunto que hayas separado, o el dataset completo).
2. **Etiquetar imágenes**: Haz clic en **"Auto-Extract Labels"** para usar el formato del dataset de Kaggle (`1_xxx.png` / `0_xxx.png`) para extraer las verdaderas etiquetas automáticamente y poder comparar métricas. (También puedes marcarlas manualmente).
3. **Ejecutar Inferencia (E3)**: Pulsa **"Start Live Inference"**.
4. **Métricas en Tiempo Real (E4)**: El Dashboard se actualizará imagen por imagen (en paralelo), mostrando el *Accuracy*, *Precision*, *Recall*, la Matriz de Confusión y resaltando los aciertos y fallos visualmente.

## 📄 Sustento Técnico (E2)
El sustento de las técnicas y parámetros empleados (Arquitectura CNN, normalización y abandono clásico de descriptores HOG manuales) está disponible en tu panel de artefactos como `Technical_Report.md`.
