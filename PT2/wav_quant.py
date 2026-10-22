#!/usr/bin/env python3
import sys
import wave
import numpy as np


# Funções Auxiliares

def print_help():
    print("Usage: python wav_quant.py <input.wav> <output.wav> <bits> [mode]\n")
    print("Arguments:")
    print("bits  - número de bits a manter por amostra")
    print("mode  - tipo de quantizador (por omissão: round)")
    print("        round    - arredondamento ao nível mais próximo")
    print("        truncate - truncatura (descarta os bits menos significativos)")
    print("        midrise  - truncatura + meio passo (quantizador mid-rise)")
    sys.exit(1)

def read_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()        # Número de canais de áudio (ex: 1 para Mono, 2 para Estéreo)
            sampwidth = wf.getsampwidth()         # Largura de cada amostra em bytes (1, 2, 3 ou 4)
            framerate = wf.getframerate()         # Taxa de amostragem em Hz
            n_frames = wf.getnframes()            # Número total de frames (amostras por canal)
            params = wf.getparams()               # Guarda todos os parâmetros originais para uso posterior

            # Lê todos os frames de áudio do ficheiro para um buffer
            raw_data = wf.readframes(n_frames)

        # Converte os bytes para inteiros com sinal (int64) com base na largura da amostra (sampwidth)
        if sampwidth == 1:
            # 8-bit: dados sem sinal (0 a 255). Subtrai-se 128 para centrar em torno de 0 (-128 a 127)
            audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int64) - 128
        elif sampwidth == 2:
            # 16-bit: dados com sinal (-32768 a 32767), Little-Endian
            audio = np.frombuffer(raw_data, dtype="<i2").astype(np.int64)
        elif sampwidth == 3:
            # 24-bit: processamento especial de 3 bytes por amostra (complemento para dois)
            bytes_arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
            # Recombina os 3 bytes individuais num inteiro usando deslocamentos de bits (Little-Endian)
            audio = (
                bytes_arr[:, 0].astype(np.int64)
                | (bytes_arr[:, 1].astype(np.int64) << 8)
                | (bytes_arr[:, 2].astype(np.int64) << 16)
            )
            # Extensão de sinal: se o 24º bit estiver ativo (negativo), subtrai 2^24 para preservar o sinal correto
            audio = np.where(audio & 0x800000, audio - 0x1000000, audio)
        elif sampwidth == 4:
            # 32-bit: dados inteiros com sinal, Little-Endian
            audio = np.frombuffer(raw_data, dtype="<i4").astype(np.int64)
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
        sampwidth = original_params.sampwidth
        bits = sampwidth * 8

        # Restringe as amostras ao intervalo representável com a largura original para evitar overflow
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        audio_clipped = np.clip(audio, lo, hi).reshape(-1)

        # Converte de volta para o formato binário correspondente à largura original da amostra
        if sampwidth == 1:
            # 8-bit: volta a somar 128 para ficar sem sinal (0 a 255)
            data_out = (audio_clipped + 128).astype(np.uint8).tobytes()
        elif sampwidth == 2:
            # 16-bit: inteiro com sinal, Little-Endian
            data_out = audio_clipped.astype("<i2").tobytes()
        elif sampwidth == 3:
            # 24-bit: extrai cada amostra em 3 bytes individuais (Little-Endian) utilizando máscaras e deslocamentos
            b0 = (audio_clipped & 0xFF).astype(np.uint8)
            b1 = ((audio_clipped >> 8) & 0xFF).astype(np.uint8)
            b2 = ((audio_clipped >> 16) & 0xFF).astype(np.uint8)
            # Empilha os bytes e converte para um fluxo de bytes binários puro
            data_out = np.stack((b0, b1, b2), axis=-1).tobytes()
        elif sampwidth == 4:
            # 32-bit: inteiro com sinal, Little-Endian
            data_out = audio_clipped.astype("<i4").tobytes()
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(audio.shape[1])       # Define o número de canais do ficheiro de saída
            wf.setsampwidth(sampwidth)            # Define a largura da amostra em bytes
            wf.setframerate(framerate)            # Define a taxa de amostragem

            # Escreve os frames de áudio convertidos para o disco
            wf.writeframes(data_out)

        print(f"Successfully processed and saved to {file_path}")
    except Exception as e:
        print(f"Error writing file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


# Quantização

# Reduz o número de bits efetivos por amostra (quantização escalar uniforme)
def quantize(audio, orig_bits, target_bits, mode="round"):
    shift = orig_bits - target_bits                     # Número de bits menos significativos a descartar
    if shift == 0:
        return audio.copy()                             # Nada a fazer, mantém-se a resolução original

    step = 1 << shift                                   # Passo de quantização Δ = 2^shift
    q_min = -(1 << (target_bits - 1))                   # Índice mínimo representável com target_bits
    q_max = (1 << (target_bits - 1)) - 1                # Índice máximo representável com target_bits
    x = audio.astype(np.int64)

    if mode == "round":
        # Arredonda cada amostra ao nível de quantização mais próximo
        idx = np.floor(x / step + 0.5).astype(np.int64)
        idx = np.clip(idx, q_min, q_max)                # Evita ultrapassar o último nível
        output = idx * step
    elif mode == "truncate":
        # Descarta os bits menos significativos (arredonda sempre para baixo)
        output = (x >> shift) << shift
    elif mode == "midrise":
        # Trunca e soma meio passo, colocando o valor reconstruído no centro do intervalo
        output = ((x >> shift) << shift) + step // 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Garante que o resultado cabe na largura original
    lo, hi = -(1 << (orig_bits - 1)), (1 << (orig_bits - 1)) - 1
    return np.clip(output, lo, hi)


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print_help()

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        target_bits = int(sys.argv[3])
    except ValueError:
        print(f"Invalid number of bits: '{sys.argv[3]}'", file=sys.stderr)
        print_help()

    # Argumento opcional: modo de quantização (por omissão: round)
    mode = sys.argv[4].lower() if len(sys.argv) == 5 else "round"
    if mode not in ("round", "truncate", "midrise"):
        print(f"Unknown mode: '{mode}'", file=sys.stderr)
        print_help()

    audio, framerate, params = read_wav(input_path)

    orig_bits = params.sampwidth * 8
    if not 1 <= target_bits <= orig_bits:
        print(f"Bits must be between 1 and {orig_bits} (input file uses {orig_bits} bits)", file=sys.stderr)
        sys.exit(1)

    print(f"Quantizing {input_path}: {params.nchannels} channel(s), {framerate} Hz, "
          f"{orig_bits} bits -> {target_bits} effective bits ({mode})...")

    quantized = quantize(audio, orig_bits, target_bits, mode)

    write_wav(output_path, quantized, framerate, params)

if __name__ == "__main__":
    main()