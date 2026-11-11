# Cómo se lee el binario de AvyScan Lab

Notas de lectura de **AvyScan Lab 4.2.6** (`AvyscanLab.exe`, `FileVersion` 4.2.6.0), hechas para seguir en otra sesión sin volver a reconstruir el camino. El algoritmo de medición está en [perfo-lock.md](perfo-lock.md). El interruptor del eje está en [perfo-eje-horizontal.md](perfo-eje-horizontal.md). Acá está solo cómo se abre el programa.

Los números de offset, de fila de metadatos y de corte del blob son de **esta** build. Una actualización los mueve. El orden de los pasos no cambia. Al final hay una lista de lo que hay que rehacer cuando suba la versión.

La carpeta de instalación no deja escribir sin elevación. El ejecutable es un single-file de .NET: el código está en `AvyscanLab.dll`, embebida. No se descompila el `.exe`.

## Orden de lectura

1. Sacar `AvyscanLab.dll` del bundle.
2. Confirmar la versión y `CurrentAlgorithmVersion`.
3. Buscar el nombre en el heap de cadenas de usuario y ver qué método lo carga.
4. Descompilar solo ese tipo con ilspycmd.
5. Si el texto no está en claro (el script AviSynth no está), descifrar el canon.

Lo que sí está en claro en la DLL: nombres de tipo, placeholders (`perfo_horizontal`, `perfo_dx`, `perfo_dy`, `perfo_pad`) y la lógica de C# (`PerfoLockService`, `RecommendedPad`, el interruptor). Lo que está cifrado: el catálogo de filtros y `PerfoLock.avsi`.

## 1. Sacar la DLL

Firma del bundle, buscada en el `.exe`:

```text
8B 12 02 B9 6A 61 20 38
```

En 4.2.6 está en el offset `8358264`. Ocho bytes antes hay un `int64` con la posición del header: `198477904`. El header es versión mayor 6, menor 0, `208` archivos.

A partir del header:

```text
uint32 major, uint32 minor, uint32 cantidad
byte longitud + id del bundle          # en 4.2.6 el id es tAd4jys4UYkL
int64 depsOffset, int64 depsSize
int64 runtimeConfigOffset, int64 runtimeConfigSize
uint64 flags
luego, por cada archivo:
    int64 offset, int64 size, int64 compressedSize, byte tipo, string nombre
```

El nombre es un byte de longitud si ese byte es menor que 128, y después los caracteres. `compressedSize == 0` significa que el archivo no está comprimido: se copia el rango `[offset, offset+size)`.

En 4.2.6:

| Archivo | Offset | Tamaño | Comprimido |
| --- | --- | --- | --- |
| `AvyscanLab.dll` | 10215424 | 7014400 | no |

La 4.2.5 medía 6959616 bytes y salía del mismo offset. El offset se mantuvo en este salto; no hay que darlo por fijo en el siguiente. Se vuelve a parsear el header y se busca el nombre `AvyscanLab.dll`.

Comprobar que lo extraído empieza con `MZ`.

## 2. Versión y algoritmo

```text
[System.Diagnostics.FileVersionInfo]::GetVersionInfo("...\AvyscanLab.exe")
```

`FileVersion` 4.2.6.0, `ProductVersion` 4.2.6.

En `AvyScanLab.Models.PerfoLockData`, `CurrentAlgorithmVersion` sigue en **2**. `RecommendedPad()` no cambió con el eje horizontal:

```text
techo( max de los fotogramas de max(|tx|, |ty|) ) + 8
```

Si esa constante o esa función cambian, [perfo-lock.md](perfo-lock.md) quedó viejo.

## 3. Cadenas y quién las usa

Herramienta: `dnfile` sobre la DLL. El heap `#US` se recorre con la longitud comprimida de .NET (un byte si es `< 0x80`). Cada entrada es UTF-16 y un byte de cola. El token `ldstr` es `0x70000000 | offset`, con el offset de la entrada dentro del heap, no el offset en el archivo.

Cadenas útiles de 4.2.6 y el offset de su entrada en `#US`:

| Cadena | Offset `#US` |
| --- | --- |
| `perfo_horizontal` | 19592 |
| `perfo_on` | 18730 |
| `perfo_mode` | 74581 |
| `.perfo_dx.txt` | 39732 |
| `.perfo_dy.txt` | 39760 |
| `perfo_dx` | 107868 |
| `perfo_dy` | 107886 |
| `perfo_pad` | 107904 |

Esos offsets cambian en cada build. Se buscan de nuevo por el texto.

Para ver quién las usa: se recorren los cuerpos de `MethodDef` y se busca el opcode `0x72` seguido del token en little-endian. En 4.2.6 eso dio:

| Cadena | Método |
| --- | --- |
| `perfo_horizontal` | `AvyScanLab.Views.CustomFilterPresenter.BuildStabilizeControls` |
| `perfo_dx`, `perfo_dy`, `perfo_pad` | `AvyScanLab.Models.ClipState.InjectPerfoLockInto` |
| `.perfo_dx.txt`, `.perfo_dy.txt` | `AvyScanLab.Views.MainWindow.CaptureScriptSnapshot` |

`InjectPerfoLockInto` copia las dos listas tal cual cuando hay datos (`perfo_mode = on`). No pone `tx` en cero. El eje se anula después, en el script.

El interruptor, dentro de `BuildStabilizeControls`, solo se arma si `perfo_on` está activo. Encendido significa «el valor guardado no es `false`». Un proyecto anterior, sin la clave, queda encendido. El clic solo hace `CommitValue`: reescribe la configuración y el script. No llama al análisis.

Nombres estables, aunque el resto del tipo esté ofuscado:

```text
AvyScanLab.Services.CustomFilterService.PerfoHorizontalPlaceholder = "perfo_horizontal"
AvyScanLab.Services.CustomFilterService.PerfoLockLibraryId = "perfolock"
```

`dnfile` nombra así los campos del header CLR: `ResourcesRva` y `ResourcesSize`, y la tabla `FieldRva`. No `Resources` ni `FieldRVA`.

## 4. Descompilar

`ilspycmd` 9.1.0.7988, instalado como herramienta local:

```text
dotnet tool install ilspycmd --version 9.1.0.7988 --tool-path %TEMP%\ilspy
```

En esta máquina el ejecutable quedó en `C:\Users\juan\AppData\Local\Temp\ilspy\ilspycmd.exe`.

Se descompila un tipo, no el ensamblado entero:

```text
ilspycmd AvyscanLab.dll -t AvyScanLab.Services.PerfoLockService -o salida
ilspycmd AvyscanLab.dll -t AvyScanLab.Models.PerfoLockData -o salida
ilspycmd AvyscanLab.dll -t AvyScanLab.Models.ClipState -o salida
ilspycmd AvyscanLab.dll -t AvyScanLab.Views.CustomFilterPresenter -o salida
ilspycmd AvyscanLab.dll -t AvyScanLab.Services.CustomFilterService -o salida
```

`CustomFilterPresenter` sale grande (cientos de KB). Se busca `perfo_horizontal` adentro.

El cargador del canon es un tipo de nombre ofuscado: espacio de nombres U+1680, clase U+2057 (`⁗`). ilspycmd lo acepta si se le pasa ese nombre. Ahí están `GetManifestResourceStream` de los tres binarios y el descifrado.

Tipos que responden cada pregunta:

| Pregunta | Dónde |
| --- | --- |
| Cómo se mide `(tx, ty)` | `PerfoLockService` |
| Versión del algoritmo, listas, `pad` | `PerfoLockData` |
| Qué se inyecta al script (`perfo_dx`, `perfo_dy`, `perfo_pad`) | `ClipState.InjectPerfoLockInto` |
| El interruptor en la interfaz | `CustomFilterPresenter.BuildStabilizeControls` |
| El texto del botón y de la guía | recursos `AvyScanLab.Lang.ui.*.json` y `whatsnew.*.json` |
| El script que aplica el eje | canon, sección siguiente |

La guía de usuario en `Users Guide\AvyScanLab_Guide_es.txt` describe el interruptor, no la fórmula. El texto de la 4.2.6 está en el recurso `AvyScanLab.Lang.whatsnew.es.json`, clave `"4.2.6"`.

No sirve el JSON de filtros que la aplicación deja en `%AppData%\Roaming\AvyScanLab\Filters\`. Al guardar un filtro de fábrica le vacía `Code`. `stabilize.json` de esa carpeta no tiene el script. Tampoco sirve `ScriptUser.avs` si es anterior al ejecutable: el del 10 de noviembre de 2026 todavía llama a `PerfoLockApply` con cinco argumentos. El de la 4.2.6 tiene seis y está en el canon.

## 5. Descifrado del canon

Tres recursos del manifiesto, cada uno con un `uint32` de longitud y después el payload:

| Recurso | Para qué | En 4.2.6, payload |
| --- | --- | --- |
| `AvyScanLab.FilterCanon.bin` | lista JSON de filtros, con el `Code` de Estabilización | 55496 bytes cifrados |
| `AvyScanLab.ScriptCanon.bin` | `PerfoLock.avsi` y `SpliceDetect.avsi` | 5896 |
| `AvyScanLab.MasterCanon.bin` | `ScriptMaster.en.avs` | 19400 |

`ScriptMaster.en.avs` no contiene `PerfoLockApply`. El aplicador es `PerfoLock.avsi`. La llamada está en el `Code` del filtro `stabilize`.

Formato del payload, comprobado contra esta build:

```text
[0:8]     magic ASCII "AVYFCN01"
[8:24]    IV de AES, 16 bytes
[24:N-32] ciphertext AES-256-CBC, PKCS7
[N-32:N]  HMAC-SHA256 de los bytes [0:N-32]
```

Las dos claves de 32 bytes salen del constructor estático de la clase del canon: dos `newarr byte` de longitud 32, `ldtoken` a un campo con RVA, `RuntimeHelpers.InitializeArray`. El primero es `k1`, el segundo es `k2`. En 4.2.6 esos campos son las filas 9934 y 9924 de `Field`, RVA en la tabla `FieldRva` (esa tabla tiene 25 filas en esta build):

```text
k1 = 3d9b3e6e473f5279f03804cc20f7c68b59644428377d687254d133ab3cdb31c1
k2 = 197064ac159b3ea966218bd6bd854b2c4be23c66438d14cfb9df7cd4db155bb7
```

```text
clave_aes  = SHA256( k1 + "canon-data-v1" + k2 )
clave_hmac = SHA256( k2 + "canon-auth-v1" + k1 )
```

Esas dos etiquetas y el magic no están en el heap `#US`. Están en un blob de bytes del mismo tipo ofuscado (`<PrivateImplementationDetails>{64A6ECF5-4606-4BC4-954C-686E1DF05309}.C075EAD6-C65F-437E-92D7-33DF1EA38889`). Su constructor estático hace `InitializeArray` de **100869** bytes (`0x018A05`) sobre el campo fila 9977 y después recorre el arreglo:

```text
blob[i] = blob[i] XOR (i & 0xFF) XOR 0xAA
```

En 4.2.6, ya decodificado, los cortes son:

| Qué | Inicio | Largo | Texto |
| --- | --- | --- | --- |
| magic | `0xBB18` | 8 | `AVYFCN01` |
| etiqueta HMAC | `0xBB20` | 13 | `canon-auth-v1` |
| etiqueta AES | `0xBB2D` | 13 | `canon-data-v1` |

En otra build no se reutilizan el largo ni los cortes. Se decodifica el blob y se busca `AVYFCN01`. Las etiquetas van pegadas a ese magic. Si el XOR ya no es `0xAA`, el blob decodificado no contiene ese ASCII: hay que volver a leer el bucle del constructor (el `xor` con el índice y la constante).

Después del HMAC (tiene que coincidir; si no, la clave o el formato cambiaron):

```text
AES-256-CBC con clave_aes y el IV
quitar el PKCS7
inflate crudo, wbits = -15
```

El resultado es JSON. `ScriptCanon` es un diccionario nombre → texto. `FilterCanon` es una lista de filtros. El filtro `Id == "stabilize"` trae el control y la llamada.

Control, en esta build:

```text
Placeholder = perfo_horizontal
Type        = checkbox
Default     = true
OnValue     = true
OffValue    = false
```

Llamada, en el `Code` de `stabilize`:

```text
PerfoLockApply(c, dxfile, dyfile, pad, n_img_pad, (perfo_horizontal || !perfo_on))
```

Con Perfo activo y el interruptor en `false`, el sexto argumento es falso. Con Edge, `perfo_on` es falso y el sexto argumento queda verdadero: Edge no pierde el eje lateral. La fórmula que sale de eso está en [perfo-eje-horizontal.md](perfo-eje-horizontal.md). No hay que volver a deducirla del IL si `PerfoLock.avsi` sigue igual.

`PerfoLock.avsi` de 4.2.6, en corto:

```text
pad  = pad + (pad % 2)                 # par superior
padx = horizontal ? pad : 0
kx   = horizontal ? 1.0 : 0.0
AddBorders(fotograma, padx, pad, padx, pad)
BicubicResize(..., src_left = padx + tx * kx, src_top = pad + ty,
              src_width = ancho, src_height = alto)
```

`horizontal` por defecto es verdadero si el argumento no viene.

## 6. Cuando salga otra versión

No reutilizar offset de DLL, filas 9934 / 9924 / 9977, ni los cortes `0xBB18`. Rehacer esto y comparar:

1. `FileVersion` del `.exe`.
2. Parsear el bundle y extraer la DLL nueva. Tiene que empezar con `MZ`.
3. `CurrentAlgorithmVersion` y `RecommendedPad` en `PerfoLockData`. Si cambian, se actualiza [perfo-lock.md](perfo-lock.md).
4. Buscar otra vez `perfo_horizontal` en `#US` y el método que hace `ldstr`.
5. Volver a ubicar `k1` y `k2` por el constructor de la clase del canon (dos arreglos de 32 bytes), no por el número de fila.
6. Decodificar el blob, buscar `AVYFCN01`, leer las dos etiquetas, descifrar los tres recursos. El HMAC tiene que cuadrar.
7. Diff de `PerfoLock.avsi` y del `Code` del filtro `stabilize`. Si la llamada o `kx` / `padx` cambian, se actualiza [perfo-eje-horizontal.md](perfo-eje-horizontal.md).
8. Anotar en este archivo los offsets nuevos y tachar los de 4.2.6.

Mientras el HMAC verifique y `PerfoLock.avsi` tenga `bool "horizontal"` con `kx = 0` al desactivarlo, la fórmula del otro documento sigue siendo la de esta versión.
