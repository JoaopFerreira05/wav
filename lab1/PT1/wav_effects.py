#!/usr/bin/env python3
import sys
import wave
import numpy as np


# Funções Auxiliares

def print_help():
    print("Usage: python wav_effects.py <input.wav> <output.wav> <effect>\n")
    print("Available effects:")
    print("se - Single echo")
    print("me - Multiple echoes")
    print("am - Amplitude modulation")
    print("vd - Time-varying delays")
    sys.exit(1)

def read_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()        # Número de canais de áudio (ex: 1 para Mono, 2 para Estéreo)
            sampwidth = wf.getsampwidth()         # Largura de cada amostra em bytes (1, 2, 3 ou 4)
            framerate = wf.getframerate()         # Taxa de amostragem em Hz (ex: 44100)
            n_frames = wf.getnframes()            # Número total de fotogramas (amostras por canal)
            params = wf.getparams()               # Guarda todos os parâmetros originais para uso posterior
            
            # Lê todos os fotogramas de áudio do ficheiro para um buffer
            raw_data = wf.readframes(n_frames)
            
        # Converte os bytes para o tipo numérico adequado com base na largura da amostra (sampwidth)
        if sampwidth == 1:
            # 8-bit: dados sem sinal (0 a 255). Subtrai-se 128 e divide-se por 128 para centrar em torno de 0 (-1.0 a 1.0)
            audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sampwidth == 2:
            # 16-bit: dados com sinal (-32768 a 32767). Divide-se por 32768.0 para normalizar
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
        elif sampwidth == 3:
            # 24-bit: processamento especial de 3 bytes por amostra (complemento para dois)
            bytes_arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
            # Recombina os 3 bytes individuais num inteiro de 32 bits usando deslocamentos de bits (Little-Endian)
            audio_int = (
                bytes_arr[:, 0].astype(np.int32) 
                | (bytes_arr[:, 1].astype(np.int32) << 8) 
                | (bytes_arr[:, 2].astype(np.int32) << 16)
            )
            # Extensão de sinal: se o 24º bit estiver ativo (negativo), subtrai o valor máximo para preservar o sinal correto
            audio_int = np.where(audio_int & 0x800000, audio_int - 0x1000000, audio_int)
            # Normaliza dividindo pelo valor máximo representável em 24 bits (8388608.0)
            audio = audio_int.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            # 32-bit: dados inteiros com sinal, normalizados dividindo por 2^31
            audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")
            
        # Reorganiza o array unidimensional numa matriz bidimensional com a forma (n_frames, n_channels)
        audio = audio.reshape(-1, n_channels)
        return audio, framerate, params
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def write_wav(file_path, audio, framerate, original_params):
    try:
        # Restringe o sinal de áudio estritamente ao intervalo [-1.0, 1.0] para evitar distorção extrema (clipping)
        audio_clipped = np.clip(audio, -1.0, 1.0)
        
        sampwidth = original_params.sampwidth
        
        # Converte de volta para o formato inteiro correspondente à largura original da amostra
        if sampwidth == 1:
            # 8-bit: desfaz a normalização, arredonda e converte para uint8 de forma segura
            data_out = np.clip(np.rint(audio_clipped * 128.0 + 128.0), 0, 255).astype(np.uint8)
        elif sampwidth == 2:
            # 16-bit: desfaz a normalização, arredonda e converte para int16 de forma segura
            data_out = np.clip(np.rint(audio_clipped * 32768.0), -32768, 32767).astype(np.int16)
        elif sampwidth == 3:
            # 24-bit: converte para inteiro de 32 bits limitado ao intervalo de 24 bits
            audio_int = np.clip(np.rint(audio_clipped * 8388608.0), -8388608, 8388607).astype(np.int32)
            # Extrai cada amostra em 3 bytes individuais (Little-Endian) utilizando máscaras e deslocamentos
            b0 = (audio_int & 0xFF).astype(np.uint8)
            b1 = ((audio_int >> 8) & 0xFF).astype(np.uint8)
            b2 = ((audio_int >> 16) & 0xFF).astype(np.uint8)
            # Empilha os bytes e converte para um fluxo de bytes binários puro
            data_out = np.stack((b0, b1, b2), axis=-1).tobytes()
        elif sampwidth == 4:
            # 32-bit: usa float64 para evitar overflow na multiplicação e converte para int32
            data_out = np.clip(audio_clipped.astype(np.float64) * 2147483648.0, -2147483648, 2147483647).astype(np.int32)
        else:
            # Fallback seguro para 16-bit caso o formato não seja reconhecido
            data_out = np.clip(np.rint(audio_clipped * 32768.0), -32768, 32767).astype(np.int16)
            sampwidth = 2
            
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(audio.shape[1])       # Define o número de canais do ficheiro de saída
            wf.setsampwidth(sampwidth)            # Define a largura da amostra em bytes
            wf.setframerate(framerate)            # Define a taxa de amostragem
            
            # Escreve os fotogramas de áudio convertidos para o disco (tratamento especial para 24-bit)
            if sampwidth == 3:
                wf.writeframes(data_out)
            else:
                wf.writeframes(data_out.tobytes())
            
        print(f"Successfully processed and saved to {file_path}")
    except Exception as e:
        print(f"Error writing file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


# Efeitos de Áudio

# Adiciona um único eco
def effect_single_echo(audio, framerate, delay_sec=0.3, decay=0.7):
    delay_samples = int(delay_sec * framerate)  # Converte o atraso em amostras
    n_frames, n_channels = audio.shape
    
    output = np.zeros((n_frames + delay_samples, n_channels), dtype=np.float32) # Cria um vetor maior para o eco
    output[:n_frames, :] += audio                                               # Insere o áudio original
    output[delay_samples:delay_samples + n_frames, :] += audio * decay          # Adiciona o eco atenuado
    
    return output

# Adiciona múltiplos ecos
def effect_multi_echo(audio, framerate, delay_sec=0.25, decay=0.7, repeats=6):
    delay_samples = int(delay_sec * framerate)  # Converte o atraso em amostras
    n_frames, n_channels = audio.shape
    
    total_length = n_frames + delay_samples * repeats
    output = np.zeros((total_length, n_channels), dtype=np.float32)
    
    for i in range(repeats + 1):                                    # Gera o número definido de repetições
        start = i * delay_samples                                   # Deslocamento temporal de cada eco
        output[start:start + n_frames, :] += audio * (decay ** i)   # Atenuação progressiva por cada iteração
        
    return output

# Aplica Modulação de Amplitude
def effect_amplitude_modulation(audio, framerate, mod_freq=4.0, depth=0.8):
    n_frames, n_channels = audio.shape
    t = np.arange(n_frames) / framerate                             # Cria o vetor temporal para cada amostra
    
    # Gera um oscilador de baixa frequência (LFO) com base numa função sinusoidal
    lfo = 1.0 - depth / 2.0 + (depth / 2.0) * np.sin(2 * np.pi * mod_freq * t)          
    lfo = lfo[:, np.newaxis]
    
    return audio * lfo

# Aplica um atraso modulado no tempo
def effect_time_varying_delay(audio, framerate, base_delay_ms=15.0, mod_depth_ms=10.0, mod_freq=0.5):
    n_frames, n_channels = audio.shape

    # Define o atraso base e a profundidade de variação nas amostras
    base_samples = int((base_delay_ms / 1000.0) * framerate)
    depth_samples = (mod_depth_ms / 1000.0) * framerate
    
    t = np.arange(n_frames) / framerate
    output = np.zeros_like(audio)
    
    # Utiliza interpolação linear para calcular o valor exato da amostra no índice fracionário
    for i in range(n_frames):
        current_delay = base_samples + depth_samples * np.sin(2 * np.pi * mod_freq * t[i])
        
        src_idx = i - current_delay
        
        if src_idx >= 0 and src_idx < n_frames - 1:
            idx_low = int(np.floor(src_idx))
            frac = src_idx - idx_low
            if idx_low + 1 < n_frames:
                interpolated = (1.0 - frac) * audio[idx_low] + frac * audio[idx_low + 1]
            else:
                interpolated = audio[idx_low]
            output[i] = audio[i] + 0.6 * interpolated
        else:
            output[i] = audio[i]
            
    return output


def main():
    if len(sys.argv) != 4:
        print_help()
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    effect_name = sys.argv[3].lower()
    
    audio, framerate, params = read_wav(input_path)
    
    print(f"Applying effect '{effect_name}' to {input_path}...")
    
    if effect_name == "se":
        processed_audio = effect_single_echo(audio, framerate)
    elif effect_name == "me":
        processed_audio = effect_multi_echo(audio, framerate)
    elif effect_name == "am":
        processed_audio = effect_amplitude_modulation(audio, framerate)
    elif effect_name == "vd":
        processed_audio = effect_time_varying_delay(audio, framerate)
    else:
        print(f"Unknown effect: '{effect_name}'", file=sys.stderr)
        print_help()
        
    write_wav(output_path, processed_audio, framerate, params)

if __name__ == "__main__":
    main()