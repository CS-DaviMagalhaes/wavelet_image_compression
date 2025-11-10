import math
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pywt

#calaverita
def wdr_encoder_MEJORADO(coeff_list, T_min=1):
    coeff_list = np.asarray(coeff_list).ravel()
    n_coeffs = len(coeff_list)
    ics = set(range(n_coeffs))
    scs = set()
    max_coef = np.max(np.abs(coeff_list))
    if max_coef == 0: 
        return "", (0, []), n_coeffs
    t_k = 2**math.floor(math.log2(max_coef))
    t_max = t_k
    bitstream_lista = []
    thresholds = []
    reconstruction = np.zeros(n_coeffs)
    while t_k >= T_min:
        thresholds.append(t_k)
        nuevos_significantes = set() 
        for idx in sorted(ics):
            coef = coeff_list[idx]       
            if abs(coef) >= t_k:
                bitstream_lista.append('1')
                ics.remove(idx)
                scs.add(idx)
                nuevos_significantes.add(idx)

                signo = '1' if coef >= 0 else '0' 
                bitstream_lista.append(signo)
                
                sign_val = 1 if coef >= 0 else -1
                reconstruction[idx] = sign_val * 1.5 * t_k
            else:
                bitstream_lista.append('0')

        # --- PASE DE REFINAMIENTO CORREGIDO ---
        for idx in sorted(scs.difference(nuevos_significantes)):
            coef_abs = abs(coeff_list[idx])
            current_recon_abs = abs(reconstruction[idx])
            
            # Determinar si está en mitad superior del intervalo
            if coef_abs >= current_recon_abs:
                bitstream_lista.append('1')
                reconstruction[idx] += np.sign(reconstruction[idx]) * (t_k / 2)
            else:
                bitstream_lista.append('0')
                reconstruction[idx] -= np.sign(reconstruction[idx]) * (t_k / 2)

        t_k = t_k / 2
        
    return "".join(bitstream_lista), (t_max, thresholds), n_coeffs


def wdr_decoder_MEJORADO(bitstream, num_coeffs, T_info):
    """
    Decoder WDR corregido
    """
    t_max, thresholds = T_info
    
    rec_coeffs = np.zeros(num_coeffs)
    ics = set(range(num_coeffs))
    scs = set()
    
    bit_pos = 0
    
    for t_k in thresholds:
        nuevos_significantes = set()
        
        # --- PASE DE SIGNIFICACIÓN ---
        for idx in sorted(ics):
            if bit_pos >= len(bitstream): 
                break
            
            bit_sig = bitstream[bit_pos]
            bit_pos += 1
            
            if bit_sig == '1':
                ics.remove(idx)
                scs.add(idx)
                nuevos_significantes.add(idx)
                
                if bit_pos >= len(bitstream): 
                    break
                signo_bit = bitstream[bit_pos]
                bit_pos += 1
                signo = 1 if signo_bit == '1' else -1
                
                rec_coeffs[idx] = signo * 1.5 * t_k

        # --- PASE DE REFINAMIENTO ---
        for idx in sorted(scs.difference(nuevos_significantes)):
            if bit_pos >= len(bitstream): 
                break
            
            ref_bit = bitstream[bit_pos]
            bit_pos += 1
            
            signo = np.sign(rec_coeffs[idx])
            if ref_bit == '1':
                rec_coeffs[idx] += signo * (t_k / 2)
            else:
                rec_coeffs[idx] -= signo * (t_k / 2)
            
    return rec_coeffs


def main_cwdr_MEJORADO(image_path, niveles_dwt=3, T_min_Y=4, T_min_UV=8):
    """
    Pipeline MEJORADO con submuestreo 4:2:0 y DWT multinivel
    """
    # 1. Cargar imagen
    img_rgb_orig = cv2.imread(image_path)
    if img_rgb_orig is None:
        print(f"Error: No se pudo cargar la imagen de {image_path}")
        return

    h, w, _ = img_rgb_orig.shape
    img_rgb = img_rgb_orig[0:h-(h%2), 0:w-(w%2)]
    
    # 2. Convertir a YUV
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    
    # 3. SUBMUESTREO 4:2:0 para U y V (CLAVE PARA COMPRESIÓN)
    u_sub = cv2.resize(u, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    v_sub = cv2.resize(v, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    
    print(f"Imagen: {w}x{h}")
    print(f"Y: {y.shape}, U: {u_sub.shape}, V: {v_sub.shape} (submuestreo 4:2:0)")

    bitstreams = {}
    T_infos = {}
    shapes = {}
    num_coeffs = {}
    original_shapes = {}
    
    # --- ENCODER ---
    print(f"\n--- CODIFICACIÓN (DWT nivel={niveles_dwt}) ---")
    
    for canal_data, canal_nombre, T_min in [(y, "Y", T_min_Y), 
                                              (u_sub, "U", T_min_UV), 
                                              (v_sub, "V", T_min_UV)]:
        
        # DWT MULTINIVEL (crítico para SPIHT)
        coeffs = pywt.wavedec2(canal_data, 'haar', level=niveles_dwt) #filtro haar
        
        # Aplanar
        coeff_list, c_shapes = pywt.coeffs_to_array(coeffs)
        
        original_shapes[canal_nombre] = coeff_list.shape
        shapes[canal_nombre] = c_shapes

        # Encoder
        bs, T_info, n_coeffs = wdr_encoder_MEJORADO(coeff_list, T_min=T_min)
        bitstreams[canal_nombre] = bs
        T_infos[canal_nombre] = T_info
        num_coeffs[canal_nombre] = n_coeffs
        
        print(f"{canal_nombre}: {n_coeffs} coef → {len(bs)} bits (T_min={T_min})")

    bits_total = sum(len(bitstreams[c]) for c in ['Y', 'U', 'V'])
    bpp = bits_total / (w * h)
    print(f"\nTOTAL: {bits_total} bits | BPP: {bpp:.4f} | Ratio: {(w*h*24)/bits_total:.2f}:1")

    # --- DECODER ---
    print("\n--- DECODIFICACIÓN ---")
    
    # Y
    rec_y = wdr_decoder_MEJORADO(bitstreams["Y"], num_coeffs["Y"], T_infos["Y"])
    rec_y_2d = rec_y.reshape(original_shapes["Y"])
    rec_coeffs_y = pywt.array_to_coeffs(rec_y_2d, shapes["Y"], output_format='wavedec2')
    y_rec = pywt.waverec2(rec_coeffs_y, 'haar')
    y_rec = np.clip(y_rec[:h, :w], 0, 255).astype(np.uint8)
    
    # U
    rec_u = wdr_decoder_MEJORADO(bitstreams["U"], num_coeffs["U"], T_infos["U"])
    rec_u_2d = rec_u.reshape(original_shapes["U"])
    rec_coeffs_u = pywt.array_to_coeffs(rec_u_2d, shapes["U"], output_format='wavedec2')
    u_rec_sub = pywt.waverec2(rec_coeffs_u, 'haar')
    u_rec_sub = np.clip(u_rec_sub[:h//2, :w//2], 0, 255).astype(np.uint8)
    u_rec = cv2.resize(u_rec_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # V
    rec_v = wdr_decoder_MEJORADO(bitstreams["V"], num_coeffs["V"], T_infos["V"])
    rec_v_2d = rec_v.reshape(original_shapes["V"])
    rec_coeffs_v = pywt.array_to_coeffs(rec_v_2d, shapes["V"], output_format='wavedec2')
    v_rec_sub = pywt.waverec2(rec_coeffs_v, 'haar')
    v_rec_sub = np.clip(v_rec_sub[:h//2, :w//2], 0, 255).astype(np.uint8)
    v_rec = cv2.resize(v_rec_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Reconstruir
    img_yuv_rec = cv2.merge([y_rec, u_rec, v_rec])
    img_rgb_rec = cv2.cvtColor(img_yuv_rec, cv2.COLOR_YUV2BGR)
    
    # Métricas
    psnr_val = psnr(img_rgb, img_rgb_rec, data_range=255)
    ssim_val = ssim(img_rgb, img_rgb_rec, data_range=255, channel_axis=2)
    
    print(f"\n{'='*50}")
    print(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
    print(f"{'='*50}")
    
    cv2.imwrite("imagen_reconstruida_mejorada.png", img_rgb_rec)
    
    return {'bpp': bpp, 'psnr': psnr_val, 'ssim': ssim_val, 
            'compression_ratio': (w*h*24)/bits_total}


if __name__ == "__main__":
    print("="*60)
    print("WDR/SPIHT MEJORADO")
    print("="*60)
    
    # Prueba con parámetros razonables
    results = main_cwdr_MEJORADO(
        image_path="lenna.png", 
        niveles_dwt=10,      # 3-5 niveles típicos
        T_min_Y=16,          # Mayor umbral = má0 compresión
        T_min_UV=10          # Cromático menos sensible
    )