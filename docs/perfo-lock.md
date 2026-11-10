# Perfo lock: cómo replicarlo

Este documento describe el método **Perfo** de AvyScan Lab 4.2.5 (clase `PerfoLockService`, algoritmo versión 2) con el detalle necesario para reimplementarlo. No es el plugin [PerfPan](https://github.com/arnean/PerfPan). PerfPan binariza un recorte y busca el mejor encaje entero, de a un píxel. AvyScan aprieta la zona al agujero con Otsu y después registra cada fotograma en subpíxel con `findTransformECC`, solo traslación, sobre la **magnitud del gradiente**.

El método Edge (línea o patrón) y el método Imagen (MVTools + DePan) son otros caminos. Acá está solo Perfo.

Lo que hay que obtener, por cada fotograma, es un par `(tx, ty)` en píxeles del escaneo original y una confianza `cc`. Aplicar esa traslación a todo el fotograma deja la perforación quieta. No hay rotación ni zoom.

## Constantes

Estos son los valores con los que corre el programa si nadie los cambia.

| Nombre | Valor | Para qué |
| --- | --- | --- |
| `MAX_DIM` | 1200 | El lado largo de la imagen de registro no pasa de esto |
| `ECC_ITERS` | 50 | Iteraciones internas de `findTransformECC` |
| `ECC_EPS` | 0.0001 | Criterio de parada de ECC (`COUNT` o `EPS`) |
| `REFINE_PASSES` | 2 | Pasadas externas que acumulan la traslación |
| `GAUSS_FILT` | 5 | Tamaño del filtro gaussiano de ECC |
| `MIN_CC` | 0.5 | Confianza mínima. Bajo esto la medida se considera mala |
| `MIN_STD_VISTO` | 5 | Desviación estándar mínima del fotograma que el usuario está mirando |
| `MIN_STD_CANDIDATO` | 10 | Desviación estándar mínima de un fotograma de reemplazo |
| `MAX_FRAC_MALOS` | 0.30 | Si más de este porcentaje queda bajo `MIN_CC`, el análisis se rechaza |
| `ROI_MIN` | 8 | Ancho y alto mínimos de la zona, en píxeles |
| `MARGEN_FRAC` | 0.05 | Margen automático de la zona de búsqueda, fracción del lado largo del fotograma |
| `MARGEN_MIN` | 40 | Tope inferior de ese margen, en píxeles |
| `MARGEN_MAX` | 400 | Tope superior |
| `PAD_EXTRA` | 8 | Píxeles extra alrededor del mayor desplazamiento, para el recorte posterior |

OpenCV debe ver imágenes en gris de 8 bits para el análisis. El programa fuerza un solo hilo (`setNumThreads(1)`) para que el resultado no dependa del orden de los núcleos. Conviene hacer lo mismo.

## Coordenadas y signo

Todas las zonas están en píxeles del fotograma completo, origen arriba a la izquierda, `x` hacia la derecha, `y` hacia abajo.

`tx` y `ty` son la traslación que `findTransformECC` encontró para **llevar el fotograma actual encima de la referencia**. Se aplican con el mismo convenio que `warpAffine` de OpenCV con la bandera `WARP_INVERSE_MAP`:

```text
destino(x, y) = origen(x + tx, y + ty)
```

El contenido se ve moverse en la dirección contraria: `(-tx, -ty)`. Si en tu script un desplazamiento positivo mueve la imagen hacia la derecha y hacia abajo, hay que aplicar `(-tx, -ty)`. Si usas `warpAffine` con `WARP_INVERSE_MAP` y la matriz

```text
| 1  0  tx |
| 0  1  ty |
```

el signo ya es el correcto y no hay que invertirlo.

La interpolación del análisis interno es lineal. Para el fotograma final se puede usar lineal, cúbica o Lanczos; el programa no mete el kernel de salida dentro de este servicio. Lo que tiene que coincidir es el vector `(tx, ty)`, no el filtro con el que después se remuestrea.

Los bordes que entran al cuadro quedan vacíos. El programa los muestra negros. El relleno recomendado para poder recortar después es:

```text
pad = techo( max sobre fotogramas de max(|tx|, |ty|) ) + 8
```

Ese `pad` no se usa durante la medición. Sirve para saber cuánto margen negro va a aparecer y para recortar después del nodo, nunca antes.

## 1. Zonas

Hay dos rectángulos.

**Zona de identificación.** La dibuja el usuario alrededor de la perforación. Puede ser holgada. Si el agujero está cortado por el borde del escaneo, se traza solo la parte nítida. Mínimo 8×8. Esta zona es la plantilla.

**Zona de búsqueda.** Es donde se permite que el agujero se mueva de un fotograma a otro. Si el usuario no la dibuja, se calcula así, con `W` y `H` el tamaño del fotograma:

```text
margen = clamp( round(0.05 * max(W, H)), 40, 400 )
búsqueda = identificación expandida `margen` píxeles hacia los cuatro lados
búsqueda = intersección de eso con el fotograma
```

Si el usuario dibuja la zona cian, se usa esa, pero solo si **contiene por completo** a la zona de identificación. Si le falta un pedazo, se ignora y se rastrea solo dentro de la identificación.

La zona de búsqueda no cambia la plantilla. La plantilla sigue siendo el agujero apretado. La búsqueda solo le da espacio al agujero para aparecer en otra posición.

## 2. Leer los fotogramas

Cada medición se hace sobre un recorte en gris de 8 bits, un byte por píxel, sin marco de color. El recorte es:

- la zona de búsqueda, si existe y es más grande que la identificación;
- si no, la zona de identificación ya apretada (sección 4).

El orden de los fotogramas es el orden del plano, de 0 a N−1. No se salta fotogramas en el rastreo. Un fotograma negro o blanco no se omite: queda con mala confianza y después se interpola.

Para elegir la referencia sí se descartan los planos uniformes. Ver la sección siguiente.

## 3. Elegir el fotograma de referencia

La plantilla sale de un solo fotograma.

1. Se toma el fotograma que el usuario está mirando al trazar la zona. Se calcula la desviación estándar del recorte de la identificación. Si es **≥ 5**, ese fotograma es la referencia.
2. Si es más plano (cabecera, fundido, flash), se buscan reemplazos. Siempre se incluye el fotograma pedido. Si se conoce el largo `N > 1`, también se agregan estas posiciones, con `último = N - 1`:

```text
fracciones = 0, 0.05, 0.15, 0.35, 0.60, 0.85
fotograma   = clamp( round(fracción * último), 0, último )
```

Si no se conoce el largo, los reemplazos son el fotograma pedido más 0, 12, 30, 60 y 120, sin bajar de 0.

3. Se recorre esa lista en orden de número de fotograma y se queda el **primero** cuya desviación estándar en la zona sea **≥ 10**. Si ninguno sirve, el análisis termina en «Sin perforación».

La desviación estándar es la de OpenCV `meanStdDev` sobre el recorte en gris, canal único.

## 4. Apretar la zona al agujero

Esto se hace una sola vez, sobre el recorte de la identificación en el fotograma de referencia. La imagen de entrada es gris de 8 bits.

1. CLAHE con `clipLimit = 2` y grilla de 8×8.
2. Desenfoque gaussiano 7×7, sigma 0 (OpenCV lo calcula), borde `BORDER_REFLECT_101`.
3. Dos umbrales de Otsu sobre esa imagen suavizada:
   - `THRESH_BINARY | THRESH_OTSU`, umbral 0, máximo 255. Deja blanco lo claro. Sirve para un agujero luminoso.
   - `THRESH_BINARY_INV | THRESH_OTSU`. Deja blanco lo oscuro. Sirve para un agujero negro.
4. A cada máscara: cierre morfológico (`MORPH_CLOSE`) con rectángulo 15×15, una pasada, borde constante.
5. Contornos externos (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`).
6. Se mide el área de cada contorno. Se descartan los que tienen menos del **15 %** del área del contorno mayor. De los que quedan se arma el rectángulo que envuelve a todos, y se suma el área de esos contornos.
7. Si el rectángulo envolvente ocupa más del **88 %** del área de la zona, esa polaridad se tira: es el fondo, no el agujero.
8. Entre la polaridad clara y la oscura gana la que haya sumado más área de contorno.

Con el rectángulo ganador:

- Si su área es **≥ 50 %** de la zona dibujada, no se aprieta. La plantilla es toda la zona del usuario. Igual se considera que el agujero existe.
- Si es menor, se agranda un **8 %** del ancho y del alto, con un mínimo de **6** píxeles por lado. El porcentaje se trunca, no se redondea:

```text
mx = max(6, (int)(ancho * 0.08))
my = max(6, (int)(alto  * 0.08))
apretado = (x - mx, y - my, ancho + 2*mx, alto + 2*my)
apretado = intersección con la zona dibujada
```

Si después de recortar queda de menos de 8×8, se vuelve a la zona original.

Si ninguna polaridad dejó un contorno usable, **no** se declara «Sin perforación» todavía. Se sigue con la zona original y se activa el plan B de la sección 9.

## 5. La imagen que realmente se registra

Esto es lo más fácil de implementar mal.

El ECC **no** compara el gris. Compara la magnitud del gradiente de Sobel. La vista grisácea que el programa muestra dentro del rectángulo es exactamente esa imagen.

```text
función bordes(gris8):
    suave = GaussianBlur(gris8, ksize 3x3, sigma 0, BORDER_REFLECT_101)
    gx = Sobel(suave, profundidad CV_32F, dx=1, dy=0, ksize 3)
    gy = Sobel(suave, profundidad CV_32F, dx=0, dy=1, ksize 3)
    devolver magnitude(gx, gy)    # flotante, mismo tamaño
```

Por eso da igual que el agujero sea claro o oscuro: lo que se alinea es el borde del orificio, no su tono.

Escala, para no rastrear un recorte enorme:

```text
escala = min(1, 1200 / max(ancho, alto))
```

El `ancho` y `alto` son los de la **plantilla apretada** en el camino normal. Si hay zona de búsqueda y se arma el buscador grueso de la sección 8, esa escala se calcula con el tamaño de la **búsqueda**, no de la plantilla. Pueden ser dos escalas distintas.

Si `escala < 1`, se reduce con `INTER_AREA`. Si no, se copia. Después se calculan los bordes. La plantilla de referencia es `bordes(reducir(recorte_apretado))`.

Los desplazamientos que devuelve ECC están en píxeles de la imagen reducida. Hay que dividirlos por `escala` para volver al fotograma original. Como `escala` puede ser 1, la división siempre es válida.

## 6. ECC de un fotograma

`findTransformECC(plantilla, actual, warp, MOTION_TRANSLATION, criterio, máscara, gaussFiltSize=5)`.

- `criterio` = `COUNT + EPS`, 50 iteraciones, épsilon 0.0001.
- El movimiento es solo traslación. La matriz es 2×3, identidad, y los únicos números que importan son `warp[0,2]` y `warp[1,2]`.
- El valor de retorno es la correlación `cc`. Cerca de 1 el encaje es bueno. El umbral del programa es 0.5.
- `gaussFiltSize` va en 5. No es el default de todas las versiones de OpenCV; hay que pasarlo.

Se hacen **dos pasadas externas** (`REFINE_PASSES = 2`). En cada una:

1. Si el desplazamiento acumulado todavía es (0, 0), el fotograma actual entra tal cual.
2. Si no, se pre-deforma el fotograma actual con

```text
warpAffine(actual, tamaño del actual,
           matriz = identidad con tx = acumulado_x * escala,
                                ty = acumulado_y * escala,
           INTER_LINEAR + WARP_INVERSE_MAP,
           BORDER_REPLICATE)
```

3. ECC alinea esa imagen ya movida contra la plantilla. El residuo se suma:

```text
acumulado_x += warp[0,2] / escala
acumulado_y += warp[1,2] / escala
```

4. `cc` es el de la **primera** pasada, no el de la segunda. Si la primera pasada lanza excepción, el fotograma queda en (0, 0) con confianza 0. Si falla una pasada posterior, se conserva lo que ya se había acumulado.

`acumulado_x` y `acumulado_y` son el `tx` y `ty` del fotograma, en píxeles del escaneo.

La máscara de ECC es nula en el camino normal. Solo el plan B (sección 9) pasa una máscara.

## 7. Si la primera correlación no alcanza

Si `cc >= 0.5`, ese resultado se acepta y no se busca más en este recorte.

Si no:

1. `phaseCorrelate` entre plantilla y fotograma actual, las dos ya en magnitud de Sobel, con una ventana de Hanning del mismo tamaño (`createHanningWindow`, tipo `CV_32F`).
2. El resultado `(dx, dy)` está en píxeles de la imagen reducida. El arranque para la segunda ECC es `(dx / escala, dy / escala)`, en las mismas unidades que el acumulado.
3. Se vuelve a correr el ECC de dos pasadas, pero partiendo de ese arranque en vez de (0, 0).
4. Si la nueva `cc` no supera a la primera, se queda la primera.
5. Si la supera, se queda la nueva. El fotograma queda marcado como «rescatado por fase» cuando esa `cc` también llega a 0.5. Esa marca protege la medida en la limpieza final.

Si `phaseCorrelate` falla, se deja el primer ECC aunque esté bajo 0.5.

## 8. Zona de búsqueda y matchTemplate

Esto solo corre si la zona de búsqueda es más ancha o más alta que la plantilla apretada.

Antes del lazo se guarda:

- el rectángulo de búsqueda, en coordenadas del fotograma;
- la posición de la plantilla **dentro** de ese rectángulo (`tight.x - búsqueda.x`, igual en `y`);
- `escala_b = min(1, 1200 / max(ancho_búsqueda, alto_búsqueda))`;
- la plantilla de bordes ya reducida con `escala_b`.

Por cada fotograma se decodifica el recorte de búsqueda, en gris. Sobre ese recorte:

1. Se corre el ECC de las secciones 6 y 7, pero solo sobre la ventana de la plantilla en su posición de referencia (el sub-rectángulo de arriba). Ese `(tx0, ty0, cc0)` es el candidato fino. Está expresado respecto de esa posición nominal.
2. En paralelo, `matchTemplate` de la plantilla de bordes contra `bordes(reducir(recorte_búsqueda, escala_b))`, modo `TM_CCOEFF_NORMED`.
3. `minMaxLoc` da el máximo. Si ese máximo no es finito o es **< 0.12**, el template matching se ignora y se conserva el ECC.
4. El pico se pasa a píxeles del recorte:

```text
px = clamp( round(pico_x / escala_b), 0, ancho_búsqueda - ancho_plantilla )
py = clamp( round(pico_y / escala_b), 0, alto_búsqueda  - alto_plantilla )
sx = px - origen_x_de_la_plantilla_en_la_búsqueda
sy = py - origen_y_de_la_plantilla_en_la_búsqueda
```

`sx, sy` es cuánto se corrió el agujero según el template matching, en píxeles del fotograma.

5. También se mira la vecindad de 5×5 (el píxel predicho por el ECC y dos a cada lado) en el mapa de `matchTemplate`, y se guarda el máximo de esa vecindad. Eso dice si el ECC ya estaba encima de un buen pico.

Se conserva el ECC, sin recentrar, cuando ocurre cualquiera de estas:

- `cc0 >= 0.5` y `|sx - tx0| <= 8` y `|sy - ty0| <= 8`. Los dos métodos coinciden.
- `cc0` ya es alto, concretamente `cc0 >= max(0.85, MIN_CC + 0.2)`, y el pico global no le gana a la vecindad del ECC por al menos 0.001. El template matching no aporta nada.

Si no se conserva, se recorta una ventana nueva del tamaño de la plantilla, con origen `(búsqueda.x + px, búsqueda.y + py)`, y se vuelve a correr el ECC ahí.

- El desplazamiento de esa segunda medición es relativo a la ventana nueva, así que se le suma `(sx, sy)`.
- Se adopta solo si su `cc >= 0.5`.
- Si el primer ECC también tenía `cc >= 0.5`, además hace falta que el pico global le gane a la vecindad por al menos **0.02**, y que la nueva `cc` supere a la primera en **0.08**. Si no, se deja la primera.
- Si se adopta, el fotograma queda marcado como recentrado. Esa marca también lo protege en la limpieza.

## 9. Plan B: Otsu no encontró el agujero

Si la sección 4 no dejó ningún contorno, la plantilla no es «el rectángulo apretado». Se busca una mancha lisa —el interior del agujero— y se registra el **anillo** de su borde.

Sobre el recorte de referencia, en gris:

1. Convertir a flotante (`CV_32F`, escala 1, así los valores siguen en 0…255).
2. Varianza local con un box de 9×9, normalizado, borde `BORDER_REFLECT_101`:

```text
media  = boxFilter(imagen, 9x9)
media2 = boxFilter(imagen * imagen, 9x9)
var    = media2 - media * media
lisa   = var < 64
```

3. Apertura morfológica (`MORPH_OPEN`) con elipse 7×7, una pasada. Borra manchas lisas chicas.
4. Componentes conexas de 8 vecinos, con estadísticas.
5. Se salta el fondo (etiqueta 0). Se descarta un componente si su área es menor al **3 %** de la imagen, o si el área de su caja es mayor al **88 %**.
6. Se ordenan por área de mayor a menor y se dejan como máximo **4**.

Cada candidato es una máscara binaria del componente. Con la plantilla de bordes ya escalada (`P`) y la máscara llevada a ese mismo tamaño (vecino más cercano si hubo que reducir):

```text
dilatada = dilate(máscara, elipse 29x29)
erosionada = erode(máscara, elipse 21x21)
anillo = dilatada AND NOT erosionada
plantilla_enmascarada = P copiada solo donde anillo es distinto de cero; el resto queda en 0
```

Además se guarda una copia de `P` al 25 % (`INTER_AREA`) y el anillo dilatado con una elipse de 5×5 a esa misma escala. El rectángulo envolvente de la máscara es la zona que tiene que seguir viéndose: si un desplazamiento candidato la saca del recorte y deja menos del **60 %** de su área adentro, ese candidato se descarta.

Para elegir entre los hasta 4 candidatos se usan 10 fotogramas espaciados parejo en los primeros 600, o en todo el plano si es más corto:

```text
n = min(N - 1, 600)
fotograma_i = round( i * n / 9 )    # i = 0..9
```

En cada uno se mide la `cc` de ese candidato (ECC con la máscara del anillo, más el intento al 25 % que se describe abajo). Un candidato se acepta si:

- la mediana de sus `cc` es **≥ 0.6**, y
- acierta (`cc >= 0.5`) en al menos el **80 %** de las muestras, o en al menos el **60 %** si la mediana es **≥ 0.8**.

Entre los aceptados gana el que tenga más aciertos. Si empatan en aciertos, gana el de mayor mediana, con una histéresis de 0.02: el que tiene menos aciertos solo le gana si su mediana supera a la otra por 0.02 o más. Si nadie se acepta, el plan B no se usa y se sigue con ECC sin máscara sobre la zona completa. Si además la mayoría de los fotogramas queda bajo 0.5, el resultado final será «Sin perforación».

Cuando el plan B sí elige un candidato, el ECC de cada fotograma del plano usa esa máscara de anillo. El arranque, si la primera `cc` no llega a 0.5, prueba en este orden y se queda con la mejor `cc`:

1. ECC desde (0, 0), con la máscara.
2. ECC desde el arranque de `phaseCorrelate`, con la máscara.
3. ECC al 25 % del tamaño, una sola pasada, con la máscara chica. Si dio alguna correlación, ese desplazamiento se divide por `0.25 * escala` y se usa como tercer arranque de un ECC a tamaño completo.

Un arranque se tira si, al sumarlo, el agujero se sale del recorte y queda con menos del 60 % de su área visible.

Si este plan B llegó a usarse, en la limpieza de la sección 10 los fotogramas con `cc >= 0.85` no se tocan. Si no se usó, esa protección por confianza no existe: solo protegen las marcas de la sección 7 y de la sección 8.

## 10. Limpiar la trayectoria

Después de medir todos los fotogramas se ordenan por número. Si es el camino Perfo (no el de línea) y más del **30 %** tiene `cc < 0.5`, se tira todo el análisis y se responde «Sin perforación». No se entrega una trayectoria a medias.

Si se salva, se aplican tres reparaciones, en este orden. Ninguna modifica un fotograma protegido. Un fotograma está protegido si:

- lo rescató la correlación de fase y su `cc` quedó ≥ 0.5, o
- lo recentró el template matching, o
- el plan B está activo y su `cc` es ≥ 0.85.

### 10.1 Confianza baja

Para cada fotograma con `cc < 0.5` se buscan el vecino bueno anterior y el siguiente.

- Si solo existe el siguiente, se copia su `(tx, ty)`.
- Si solo existe el anterior, se copia el suyo.
- Si existen los dos, se interpola lineal por número de fotograma:

```text
t = (f - f_anterior) / (f_siguiente - f_anterior)
tx = tx_anterior + (tx_siguiente - tx_anterior) * t
ty = ty_anterior + (ty_siguiente - ty_anterior) * t
```

La `cc` no se reescribe. Los fotogramas buenos de los extremos anclan a los malos que quedan pegados al inicio o al final.

### 10.2 Picos de un fotograma

Solo si hay al menos 3 fotogramas. Se mira la serie de `tx` y la de `ty` por separado.

La escala del ruido es la mediana de las diferencias entre fotogramas consecutivos, llevada a desviación con 1.4826:

```text
d[i] = v[i+1] - v[i]
umbral = max( 4 px,  4 * 1.4826 * MAD(d) )
```

`MAD` es la mediana de `|d - mediana(d)|`.

El fotograma `i` (ni el primero ni el último) es un pico en un eje si las dos diferencias contra sus vecinos superan el umbral **y tienen el mismo signo**: subió respecto de los dos, o bajó respecto de los dos. Un escalón, que sube y se queda arriba, no es pico.

Si es pico en X o en Y, y no está protegido, se marca. Si los marcados son más del **20 %** del plano, no se corrige ninguno. Si no, cada marcado se reemplaza por la interpolación lineal entre el ancla no marcada anterior y la siguiente, igual que en 10.1. Las anclas se leen de la serie **original**, no de valores ya reescritos en este mismo paso.

### 10.3 Apartados de la trayectoria

Solo si hay al menos 5 fotogramas. Ventana de mediana de radio 3 (la ventana pedida es 7, el radio es `7 // 2`).

```text
tendencia[i] = mediana( v[i-3 .. i+3] )   # recortada en los bordes
residuo[i]   = v[i] - tendencia[i]
umbral       = max( 8 px,  4 * 1.4826 * MAD(residuo) )
```

Se marca quien, sin estar protegido, tenga `|residuo|` sobre el umbral en X o en Y. Si son más del **25 %**, no se toca nada. Si no, se interpolan igual que los picos, entre anclas no marcadas. En los extremos, si falta un lado, se copia el ancla que sí existe.

## 11. Aplicar la corrección

Para el fotograma `f`:

```text
destino(x, y) = origen_f( x + tx[f],  y + ty[f] )
```

con muestra fuera de rango igual a negro, o al color de borde que uses. Subpíxel: el `tx` no es entero. Hay que remuestrear, no basta con un desplazamiento de píxeles enteros.

La perforación queda fija. La imagen se mueve. El negro que aparece en los bordes se recorta **después**, con el `pad` de más arriba, o con el recorte que ya tuviera el proyecto. Recortar antes de medir le quita el agujero al rastreador.

El programa guarda la serie como texto, tres decimales, cultura invariante:

- una lista `tx` separada por comas, en orden de fotograma;
- otra lista `ty` igual;
- y una forma compacta `fotograma:tx:ty` separada por punto y coma.

Esas listas son el contrato. El reproductor no vuelve a medir.

## Pseudocódigo del camino normal

Este es el caso habitual: el agujero se ve, Otsu lo encuentra, y hay zona de búsqueda automática.

```text
margen = clamp(round(0.05 * max(W, H)), 40, 400)
id = rectángulo del usuario
busqueda = expandir(id, margen) ∩ fotograma

ref = fotograma_visto si std(gris(ref, id)) >= 5
      si no, el primer candidato con std >= 10
      si no hay, abortar "sin perforación"

tight, hubo_contorno = apretar_otsu(gris(ref, id))
plantilla_rect = tight si hubo_contorno si no id

escala = min(1, 1200 / max(plantilla_rect.w, plantilla_rect.h))
plantilla = bordes(reducir(gris(ref, plantilla_rect), escala))

si busqueda es más grande que plantilla_rect:
    preparar matchTemplate como en la sección 8

para cada fotograma f:
    recorte = gris(f, busqueda) si hay buscador si no gris(f, plantilla_rect)
    (tx, ty, cc, marcas) = registrar(plantilla, recorte)   # secciones 6, 7 y 8

si fracción con cc < 0.5  >  0.30:
    abortar "sin perforación"

reparar confianza baja
reparar picos
reparar apartados de la trayectoria

para cada f:
    deformar el fotograma f con (tx, ty) según la sección 11
```

`bordes()` es la magnitud de Sobel de la sección 5. `registrar()` devuelve píxeles del fotograma original, no de la imagen reducida.

## Qué no hay que copiar de PerfPan

| | PerfPan | Este método |
| --- | --- | --- |
| Imagen que se compara | Un clip ya binarizado a mano | Magnitud de Sobel, calculada sola |
| Búsqueda | Vecinos a ±1 píxel, entero, hasta un cuarto del recorte | ECC subpíxel, más fase y template matching como arranque |
| Agujero claro y oscuro | Hay que elegir el umbral | Otsu prueba las dos polaridades |
| Fotograma malo | Puede saltar al límite del recorte | Se interpola, o se rechaza el plano entero si fallan más del 30 % |
| Transformación | Traslación entera | Traslación subpíxel, sin rotación ni zoom |

## Comprobación

Sobre un plano corto, con la perforación visible:

1. El fotograma de referencia tiene que ser uno donde el agujero se vea nítido, no una cabecera.
2. El rectángulo apretado tiene que caer sobre el agujero, con un pequeño margen, no sobre el fotograma entero. Si ocupa casi toda la zona dibujada, la zona estaba demasiado justa o el Otsu se quedó con el fondo: conviene achicar el trazo para que el agujero no sea el 88 % de la zona.
3. `cc` de los fotogramas buenos queda sobre 0.5, en general bastante más alto. Una mayoría bajo 0.5 es el síntoma de «Sin perforación».
4. `tx` y `ty` del fotograma de referencia tienen que salir cerca de cero. Si salen grandes, el signo o el recorte de la plantilla están mal.
5. Al aplicar `destino(x, y) = origen(x + tx, y + ty)`, el agujero no se mueve y la imagen sí. Si se mueve el agujero y se queda quieta la imagen, el signo está al revés.
6. Un fotograma negro en el medio no tiene que disparar un salto: su `(tx, ty)` queda interpolado entre los vecinos.
