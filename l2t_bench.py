import numpy as np
import cv2
import sys
import os
import io
import time
import subprocess
import tifffile
import imageio
from pathlib import Path

# --- INTENTO DE IMPORTAR LIBRERÍAS EXTERNAS ---
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("AVISO: 'rawpy' no está instalado. Saltando pruebas DCB/AHD.")

try:
    import colour_demosaicing
    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False
    print("AVISO: 'colour-demosaicing' no instalado. Saltando prueba Menon.")

# --- CONFIGURACIÓN ---
WIDTH = 2840
HEIGHT = 2200
FRAME_SIZE_PACKED = int(WIDTH * HEIGHT * 1.5)

# !!! AJUSTA ESTA RUTA SI RAWTHERAPEE ESTÁ EN OTRO LADO !!!
# Ruta típica en Windows:
RT_CLI_PATH = r"C:\Program Files\RawTherapee\5.12\rawtherapee-cli.exe"

# Si usas la versión 5.8 o portable, ajusta la ruta.
# Si no encuentras el .exe, el script saltará este paso.

# --- FUNCIONES DE UTILIDAD ---

def unpack_12bit_little_endian(raw_bytes, width, height):
    """ Desempaquetado Little Endian (Fix H) """
    expected_size = int(width * height * 1.5)
    if len(raw_bytes) > expected_size: raw_bytes = raw_bytes[:expected_size]
    elif len(raw_bytes) < expected_size: raw_bytes += b'\x00' * (expected_size - len(raw_bytes))

    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    b0 = data[:, 0].astype(np.uint16)
    b1 = data[:, 1].astype(np.uint16)
    b2 = data[:, 2].astype(np.uint16)
    
    p0 = ((b1 & 0x0F) << 8) | b0
    p1 = (b2 << 4) | ((b1 & 0xF0) >> 4)
    
    img_flat = np.empty(width * height, dtype=np.uint16)
    img_flat[0::2] = p0
    img_flat[1::2] = p1
    return img_flat.reshape(height, width)

def create_dng(image_array, output_path):
    """ Crea un DNG real en disco para RawTherapee """
    extratags = [
        (50706, 'B', 4, [1, 4, 0, 0], True),      # DNGVersion
        (50717, 'I', 1, 4095, True),              # WhiteLevel
        (50718, 'I', 1, 0, True),                 # BlackLevel
        (33422, 'B', 4, [0, 1, 1, 2], True),      # CFAPattern (RGGB)
        (50710, 'B', 3, [0, 1, 2], True),         # CFAPlaneColor
        (50708, 's', 0, "Lucid Phoenix IMX566", True),
    ]
    tifffile.imwrite(output_path, image_array, photometric='cfa', planarconfig=1, extratags=extratags)

def create_dng_memory(image_array):
    """ Crea objeto DNG en RAM para Rawpy """
    buffer = io.BytesIO()
    extratags = [
        (50706, 'B', 4, [1, 4, 0, 0], True),
        (50717, 'I', 1, 4095, True),
        (50718, 'I', 1, 0, True),
        (33422, 'B', 4, [0, 1, 1, 2], True),
        (50710, 'B', 3, [0, 1, 2], True),
        (50708, 's', 0, "Lucid Phoenix", True),
    ]
    tifffile.imwrite(buffer, image_array, photometric='cfa', planarconfig=1, extratags=extratags)
    buffer.seek(0)
    return buffer

def apply_sharpening(img, label):
    """ Aplica un poco de sharpening para simular escaneo final """
    # Convertir a float
    img_f = img.astype(np.float32)
    # Unsharp Mask
    gaussian = cv2.GaussianBlur(img_f, (0, 0), 1.5)
    sharp = cv2.addWeighted(img_f, 2.5, gaussian, -1.5, 0)
    return np.clip(sharp, 0, 65535).astype(np.uint16)

# --- MAIN ---

def main():
    print("=== BENCHMARK GLOBAL DE ALGORITMOS DE DEBAYERING ===")
    
    # 1. Buscar archivo
    files = list(Path(".").glob("*.raw"))
    if not files:
        print("Error: No encontré archivos .raw en esta carpeta.")
        return
    
    target_path = files[0]
    print(f"-> Analizando: {target_path.name}")
    
    out_dir = target_path.parent / "_benchmark_global"
    out_dir.mkdir(exist_ok=True)
    
    # 2. Leer y Desempaquetar
    with open(target_path, 'rb') as f:
        raw_data = f.read(FRAME_SIZE_PACKED)
    
    print("-> Desempaquetando 12-bit Little Endian...")
    img_16_raw = unpack_12bit_little_endian(raw_data, WIDTH, HEIGHT)
    
    # --- RONDA 1: OPENCV (Rápido) ---
    print("\n[1/4] OpenCV Algorithms...")
    
    # 1.1 Bilineal
    img_bi = cv2.cvtColor(img_16_raw, cv2.COLOR_BayerRG2RGB) * 16
    cv2.imwrite(str(out_dir / "01_OpenCV_Bilinear.tif"), img_bi)
    
    # 1.2 Edge Aware (Tu favorito actual)
    img_ea = cv2.cvtColor(img_16_raw, cv2.COLOR_BayerRG2RGB_EA) * 16
    cv2.imwrite(str(out_dir / "02_OpenCV_EdgeAware.tif"), img_ea)
    
    # 1.3 VNG (Limitado a 8-bit)
    img_8bit = (img_16_raw >> 4).astype(np.uint8)
    try:
        img_vng = cv2.cvtColor(img_8bit, cv2.COLOR_BayerRG2RGB_VNG)
        cv2.imwrite(str(out_dir / "03_OpenCV_VNG_8bit.tif"), img_vng)
    except: pass

    # --- RONDA 2: RAWPY / LIBRAW (Pro) ---
    if HAS_RAWPY:
        print("\n[2/4] LibRaw (Rawpy)...")
        dng_mem = create_dng_memory(img_16_raw)
        
        with rawpy.imread(dng_mem) as raw:
            # 2.1 AHD
            rgb_ahd = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,
                no_auto_bright=True
            )
            imageio.imwrite(out_dir / "04_LibRaw_AHD.tif", rgb_ahd)
            
            # 2.2 DCB (Competencia directa de AMaZE)
            rgb_dcb = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.DCB,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,
                no_auto_bright=True
            )
            imageio.imwrite(out_dir / "05_LibRaw_DCB.tif", rgb_dcb)
            
            # 2.3 DCB + Sharpening (Para comparar con EA Strong)
            dcb_sharp = apply_sharpening(rgb_dcb, "DCB")
            imageio.imwrite(out_dir / "06_LibRaw_DCB_Sharp.tif", dcb_sharp)
    else:
        print("\n[2/4] Saltando LibRaw (instala rawpy)")

    # --- RONDA 3: PYTHON CIENTÍFICO ---
    if HAS_COLOUR:
        print("\n[3/4] Scientific (Menon 2007)... (Esto es lento)")
        # Normalizar 0-1
        img_norm = img_16_raw.astype(np.float32) / 4095.0
        # Menon 2007 (DDFAPD)
        img_menon = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(img_norm, pattern='RGGB')
        img_menon = np.clip(img_menon * 65535, 0, 65535).astype(np.uint16)
        
        # Convertir RGB a BGR para OpenCV Save (colour output is RGB)
        img_menon_bgr = cv2.cvtColor(img_menon, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "07_Scientific_Menon.tif"), img_menon_bgr)
    else:
        print("\n[3/4] Saltando Menon (instala colour-demosaicing)")

    # --- RONDA 4: RAWTHERAPEE CLI (AMaZE) ---
    print("\n[4/4] RawTherapee CLI (AMaZE)...")
    
    # TRUCO 1: Usar rutas absolutas basadas en la ubicación del script
    # Esto arregla el error "amaze.pp3 not found"
    script_dir = Path(__file__).parent.resolve()
    rt_exe = Path(RT_CLI_PATH)
    pp3_file = script_dir / "amaze.pp3"
    
    # Verificación estricta
    if not rt_exe.exists():
        print(f"   -> ERROR CRÍTICO: No encuentro RawTherapee en: {rt_exe}")
        print("      Edita la variable RT_CLI_PATH al inicio del script.")
    elif not pp3_file.exists():
        print(f"   -> ERROR CRÍTICO: No encuentro el perfil en: {pp3_file}")
        print("      Asegúrate de haber creado el archivo amaze.pp3.")
    else:
        # 1. Crear DNG temporal en disco
        dng_temp = target_path.with_suffix(".dng")
        tif_out = out_dir / "08_RawTherapee_AMaZE.tif"
        
        print("   -> Creando DNG intermedio...")
        create_dng(img_16_raw, dng_temp)
        
        print("   -> Ejecutando motor AMaZE...")
        
        # Comando CLI con rutas absolutas
        cmd = [
            str(rt_exe),
            "-o", str(tif_out),  # Salida
            "-p", str(pp3_file), # Perfil
            "-Y",                # Sobrescribir
            "-t",                # TIFF
            "-b16",              # 16 bits
            "-c", str(dng_temp)  # Entrada
        ]
        
        try:
            # TRUCO 2: encoding='utf-8' y errors='replace'
            # Esto arregla el UnicodeDecodeError 0x9d
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                encoding='utf-8',     # Forzar UTF-8
                errors='replace'      # Si hay un caracter raro, pon un "?" en vez de crashear
            )
            
            if result.returncode == 0:
                print("   -> Éxito: AMaZE generado.")
                
                # Crear versión Sharp también
                if tif_out.exists():
                    img_amaze = cv2.imread(str(tif_out), cv2.IMREAD_UNCHANGED)
                    if img_amaze is not None:
                        img_amaze_sharp = apply_sharpening(img_amaze, "AMaZE")
                        cv2.imwrite(str(out_dir / "09_RawTherapee_AMaZE_Sharp.tif"), img_amaze_sharp)
            else:
                print(f"   -> Error RT (Exit Code {result.returncode}):")
                print(f"      STDERR: {result.stderr}")
                print(f"      STDOUT: {result.stdout}")

        except Exception as e:
            print(f"   -> Fallo al ejecutar subprocess: {e}")
            
        # Limpieza (Opcional)
        # try: os.remove(dng_temp)
        # except: pass
    
    print("\n=== BENCHMARK FINALIZADO ===")
    print(f"Carpeta de resultados: {out_dir}")
    print("Comparativa sugerida:")
    print("1. Nitidez Pura: Compara 02_OpenCV_EdgeAware vs 05_LibRaw_DCB vs 08_RawTherapee_AMaZE")
    print("2. Look Final: Compara 02 (tu actual) vs 06 (DCB Sharp) vs 09 (AMaZE Sharp)")

if __name__ == "__main__":
    main()