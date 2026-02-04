# Lucid Scanner
### Archivo de La Unión

## Resumen

Lucid Scanner es un software especializado en la digitalización de películas, diseñado para la preservación y el escaneo en alta resolución de material fílmico. Desarrollado para el **Archivo de La Unión**, este sistema une la tecnología moderna de visión artificial con hardware vintage modificado para crear un flujo de trabajo de archivo de grado profesional.

El núcleo del sistema funciona como una interfaz de alta velocidad para la cámara de visión artificial **Lucid Vision Phoenix PHX081s**, integrada con un mecanismo de transporte **Retroscan Mk1** modificado. Está diseñado para priorizar la integridad absoluta del cuadro, asegurando que cada fotograma del medio físico sea capturado, almacenado en buffer y guardado sin pérdidas, incluso a altas tasas de transferencia.

## Arquitectura del Sistema

El software está construido sobre una arquitectura modular que separa las operaciones críticas de hardware de la interfaz de usuario para garantizar estabilidad y rendimiento.

### 1. Motor de Captura de Alto Rendimiento
*   **Integración Directa de Hardware**: Utiliza el SDK Arena para comunicarse directamente con la cámara Lucid Phoenix, evitando las capas estándar de controladores para minimizar la latencia.
*   **Buffer "Zero-Drop" (Sin Pérdida)**: Implementa una robusta tubería multihilo. El hilo `CameraWorker` gestiona el acceso directo a memoria (DMA) desde la cámara, empujando inmediatamente los datos crudos a un buffer de escritura de 64MB. Esto desacopla la captura de la escritura en disco, permitiendo al sistema absorber los picos de latencia del sistema operativo sin perder cuadros.
*   **Telemetría en Tiempo Real**: Proporciona monitoreo granular de la salud del sistema, incluyendo temperatura del sensor, ancho de banda de la interfaz (calculado y reportado por el driver) y saturación del buffer de escritura.

### 2. Pipeline de Imagen RAW de 12-Bit
A diferencia del software de captura de video estándar, este sistema trata cada cuadro como un negativo digital.
*   **Soporte de Formato**: Captura en formatos nativos **BayerRG12p** (RAW empaquetado de 12 bits) o **RGB8**.
*   **Resolución**: Captura estandarizada de alta resolución a **2840 x 2200**.
*   **Integridad de Datos**: Los cuadros se almacenan en un formato contenedor binario crudo con metadatos JSON asociados, preservando todo el rango dinámico del sensor para la postproducción.

### 3. Visualización y Monitoreo Avanzado
La interfaz gráfica de usuario ofrece herramientas profesionales para verificación de foco y exposición durante la captura:
*   **Focus Peaking**: Superposición de detección de bordes en tiempo real para asistir con el foco óptico preciso.
*   **Zoom 1:1**: Modo de inspección píxel a píxel.
*   **Previsualización No Bloqueante**: Un hilo dedicado `PreviewWorker` maneja el debayering y la conversión de espacio de color para la pantalla, utilizando técnicas de subsampling para mantener una interfaz fluida sin robar ciclos de CPU al hilo crítico de captura.

## Revisión y Exportación

Una vez digitalizado el material, el sistema ofrece herramientas dedicadas para la gestión, revisión y conversión del material.

### Visor de Reproducción y Ajuste
La pestaña de "Visor" permite una revisión detallada del material capturado sin salir de la aplicación:
*   **Control de Velocidad de Cuadros (FPS)**: Permite ajustar y corregir la velocidad de reproducción de la captura en los metadatos. Si una película fue escaneada a una velocidad incorrecta, se puede redefinir aquí (ej. 18fps, 24fps) antes de la exportación final.
*   **Navegación Precisa**: Barra de desplazamiento y controles de reproducción para examinar la secuencia cuadro por cuadro.
*   **Inspección de Detalle**: Visualización directa de la data RAW interpretada para verificar la calidad de la imagen antes del procesado.

### Exportación por Lotes (Batch Export)
El módulo de exportación (`raw2video.py`) convierte los datos crudos escaneados en formatos de video estándar de la industria, permitiendo procesar múltiples rollos o secuencias de forma desatendida.

**Interfaz de Exportación:**
*   **Miniaturas Visuales**: El diálogo de exportación genera miniaturas reales desde el archivo RAW para identificar visualmente cada toma.
*   **Configuración de Revelado**: Selección de perfiles de nitidez y procesado (Suave, Medio, Grueso/DCB).
*   **Formatos de Salida**: Soporte para codecs de alta fidelidad:
    *   **ProRes 4444 (12-bit)**: Máxima calidad con información de color completa, ideal para etalonaje.
    *   **GoPro CineForm (12-bit)**: Codec intermedio de alto rendimiento.
    *   **ProRes 422 HQ (10-bit)**: Calidad estándar broadcast.
    *   **HEVC / H.265 (10-bit)**: Alta eficiencia para almacenamiento.
    *   **FFV1**: Formato de archivo sin pérdidas (Lossless).

### Cadena de Procesamiento Interno
Durante la exportación, cada cuadro pasa por un riguroso proceso de revelado:
1.  **Linealización**: Desempaqueta los datos Bayer de 12 bits y normaliza los valores a un rango flotante de 0.0–1.0.
2.  **Corrección Gamma**: Aplica una curva gamma configurable (default 1.4) para mapear correctamente los datos lineales del sensor al espacio visual, realzando el detalle en las sombras.
3.  **Reducción de Ruido Chroma**: Convierte al espacio de color YCrCb y aplica un desenfoque selectivo a los canales de croma. Esto elimina el ruido de color inherente a los sensores Bayer de un solo chip sin afectar el detalle de luminancia (grano).
4.  **Enfoque de Luminancia**: Aplica una máscara de enfoque (Unsharp Masking) específicamente al canal Luma para recuperar la nitidez percibida perdida durante la transferencia óptica.

## Requisitos

*   **Sistema Operativo**: Windows 10/11 (Requerido para optimización y soporte del SDK Arena).
*   **Hardware**: 
    *   Cámara: Lucid Vision Phoenix PHX081s.
    *   Transporte: Retroscan Mk1 (Modificado).
*   **Dependencias**: 
    *   Python 3.9+
    *   Arena SDK (Lucid Vision Labs)
    *   FFmpeg (accesible vía path del sistema)
    *   PyQt6, NumPy, OpenCV, psutil.

## Flujo de Trabajo Operativo

1.  **Gestión de Colecciones**: Crear una nueva colección para el rollo asignado.
2.  **Calibración**: Usar el histograma en vivo y las herramientas de peaking para establecer foco y exposición (Gain/Shutter).
3.  **Captura**: Iniciar el proceso de escaneo. El sistema escribe un flujo continuo de archivos binarios crudos.
4.  **Revisión y Ajuste**: Verificar el material en el Visor, ajustando la tasa de cuadros (FPS) correcta en los metadatos si es necesario.
5.  **Exportación**: Seleccionar las secuencias deseadas en el menú de Exportación por Lotes, elegir el codec (ej. ProRes 4444) y procesar para postproducción.

---
*Software desarrollado para uso interno en el Archivo de La Unión.*
