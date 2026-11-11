# Perfo: eje horizontal opcional

Este documento completa [perfo-lock.md](perfo-lock.md). Describe el interruptor **Eje horizontal** del método Perfo en AvyScan Lab 4.2.6. El análisis sigue siendo el de ese documento: `PerfoLockService`, algoritmo versión 2, traslación subpíxel `(tx, ty)`. Lo nuevo no mide de nuevo. Decide, al aplicar, si el `tx` ya medido entra en la corrección.

El interruptor sale activado. Desactivado, el fotograma solo se mueve en vertical. Sirve cuando el registro mete un temblor lateral que la película no tenía: el orificio queda fijo en Y y el costado no se corrige.

El método Edge no usa este interruptor. Aunque el mismo aplicador sirve para los dos, Edge sigue corrigiendo los dos ejes.

## Qué no cambia

Desactivar el eje no invalida el análisis. El botón «Analizar» sigue verde. No se vuelven a correr Otsu, ECC, fase ni template matching, ni la limpieza de la trayectoria.

Siguen guardadas las dos listas, con tres decimales:

- `tx` de cada fotograma;
- `ty` de cada fotograma.

El `pad` que calculó el análisis tampoco se rehace. Sigue siendo el de la sección 11 de perfo-lock.md:

```text
pad = techo( max sobre fotogramas de max(|tx|, |ty|) ) + 8
pad = pad + (pad % 2)          # par superior, por el YUV 4:2:0 y 4:2:2
```

Ese `pad` puede haber quedado grande por un `tx` que ahora no se va a usar. Igual se conserva. Al desactivar el eje solo se deja de agregar margen a izquierda y derecha. El margen de arriba y de abajo sigue siendo ese `pad` completo, no uno recalculado solo con `ty`.

## El interruptor

Es una casilla del nodo Estabilización, visible solo con el método Perfo activo. El valor guardado es el texto `true` o `false`.

| Estado | Valor | Qué se corrige |
| --- | --- | --- |
| Activado (defecto) | `true` | `tx` y `ty` |
| Desactivado | `false` | solo `ty` |
| Sin valor guardado | se toma como activado | `tx` y `ty` |

Activado es cualquier valor distinto de `false`. Un proyecto de antes de 4.2.6, que no tiene la clave, se comporta como antes: los dos ejes.

Al cambiar el interruptor se reescribe el script de aplicación. Los archivos de `tx` y `ty` no se tocan.

## Cómo se aplica

La llamada, con el método Perfo o Edge ya analizado, es:

```text
PerfoLockApply(clip, archivo_tx, archivo_ty, pad, imagenes_de_relleno, horizontal)
```

`horizontal` no es el interruptor pelado. Es:

```text
horizontal = (eje_horizontal == true)  o  (método Perfo apagado)
```

Con Perfo activo y el interruptor apagado, `horizontal` sale falso. Con Edge, el método Perfo está apagado, el segundo término es verdadero y `horizontal` queda verdadero aunque el interruptor exista en la configuración. Edge no pierde el eje lateral.

Dentro del aplicador:

```text
horizontal = verdadero si no vino el argumento
padx = pad    si horizontal    si no  0
kx   = 1      si horizontal    si no  0

origen_con_margen = AddBorders(fotograma, padx, pad, padx, pad)

destino = BicubicResize(origen_con_margen,
                        ancho, alto,
                        src_left = padx + tx * kx,
                        src_top  = pad  + ty,
                        src_width = ancho,
                        src_height = alto)
```

Con el eje activado, `kx` es 1 y `padx` es `pad`. Queda el mismo convenio que la sección 11 de perfo-lock.md:

```text
destino(x, y) = origen(x + tx, y + ty)
```

Con el eje desactivado, `kx` es 0 y `padx` es 0:

```text
src_left = 0
src_top  = pad + ty
destino(x, y) = origen(x, y + ty)
```

`tx` se lee igual, fotograma a fotograma, y se multiplica por cero. No hace falta reescribir la lista ni ponerla en ceros. `src_left` queda en 0 y la ventana horizontal es el ancho entero del fotograma, así que el remuestreo no desplaza en X. En Y el desplazamiento sigue siendo subpíxel.

El negro de los bordes, con el eje apagado, aparece solo arriba y abajo. Izquierda y derecha no ganan margen. El relleno temporal de cabecera y cola (`skip`, las imágenes que el script maestro agrega y después recorta) no se estabiliza, igual que antes.

## Dónde encaja en el otro documento

perfo-lock.md termina el camino de medición en la sección 10 y aplica las dos componentes en la sección 11. Este interruptor va después de esa medición, en el momento de aplicar:

```text
medir (tx, ty, cc) como en perfo-lock.md, secciones 1 a 10
guardar las dos listas

al aplicar cada fotograma:
    si eje horizontal activado:
        destino(x, y) = origen(x + tx, y + ty)
        margen en los cuatro lados = pad
    si no:
        destino(x, y) = origen(x, y + ty)
        margen solo arriba y abajo = pad
        izquierda y derecha = 0
```

Nada de lo anterior a la sección 11 cambia. Un `tx` grande sigue contando para el `pad`, para la limpieza de picos y para la confianza. Solo deja de mover la imagen.

## Comprobación

Sobre un plano ya analizado, sin volver a analizar:

1. Con el interruptor activado, el orificio no se mueve ni en X ni en Y. Es el resultado de perfo-lock.md.
2. Al desactivarlo, el botón «Analizar» sigue verde. Las listas `tx` y `ty` quedan iguales, número por número.
3. El orificio puede volver a temblar de lado a lado. En vertical tiene que seguir quieto.
4. Si se mueve también en vertical al desactivar el eje, se está anulando `ty`, o se están intercambiando los ejes.
5. Los bordes negros nuevos quedan arriba y abajo. A los lados no tiene que aparecer una franja negra que antes no estaba por culpa de este interruptor.
6. Volver a activar el eje recupera el `tx` original. No hay que analizar otra vez.
7. En el método Edge el interruptor no se muestra, y la corrección lateral sigue aplicada.
