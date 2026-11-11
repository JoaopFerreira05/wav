#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Estrutura para os cabeçalhos do WAV
typedef struct {
    uint32_t chunk_id;
    uint32_t chunk_size;
    uint32_t format;
    uint32_t subchunk1_id;
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint32_t subchunk2_id;
    uint32_t subchunk2_size;
} WavHeader;

void print_help() {
    printf("Usage:\n");
    printf("  wav_lossless enc [optional flags] input.wav compressed_file\n");
    printf("  wav_lossless dec compressed_file output.wav\n");
    exit(1);
}

// Função de Compressão Lossless
int encode_wav(const char *input_path, const char *output_path) {
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        fprintf(stderr, "Error opening input file: %s\n", input_path);
        return 0;
    }

    WavHeader header;
    fread(&header, sizeof(WavHeader), 1, fin);

    // Ler dados de áudio brutos (PCM data)
    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    long data_size = header.subchunk2_size;
    long data_start = file_size - data_size;

    fseek(fin, data_start, SEEK_SET);
    uint8_t *audio_bytes = malloc(data_size);
    fread(audio_bytes, 1, data_size, fin);
    fclose(fin);

    // Determinar largura da amostra em bytes
    int sampwidth = header.bits_per_sample / 8;
    long num_samples = data_size / sampwidth;

    // Aplicar DPCM (Predição Linear: resíduo = atual - anterior)
    int32_t *residuals = malloc(num_samples * sizeof(int32_t));
    
    for (long i = 0; i < num_samples; i++) {
        int32_t current = 0;
        if (sampwidth == 1) {
            current = (int32_t)audio_bytes[i] - 128;
        } else if (sampwidth == 2) {
            int16_t val = ((int16_t*)audio_bytes)[i];
            current = (int32_t)val;
        } else if (sampwidth == 3) {
            long idx = i * 3;
            int32_t val = audio_bytes[idx] | (audio_bytes[idx+1] << 8) | (audio_bytes[idx+2] << 16);
            if (val & 0x800000) val -= 0x1000000;
            current = val;
        } else if (sampwidth == 4) {
            current = ((int32_t*)audio_bytes)[i];
        }

        if (i == 0) {
            residuals[i] = current;
        } else {
            int32_t prev = 0;
            if (sampwidth == 1) prev = (int32_t)audio_bytes[i-1] - 128;
            else if (sampwidth == 2) prev = (int32_t)((int16_t*)audio_bytes)[i-1];
            else if (sampwidth == 3) {
                long idx = (i-1) * 3;
                int32_t val = audio_bytes[idx] | (audio_bytes[idx+1] << 8) | (audio_bytes[idx+2] << 16);
                if (val & 0x800000) val -= 0x1000000;
                prev = val;
            } else if (sampwidth == 4) prev = ((int32_t*)audio_bytes)[i-1];

            residuals[i] = current - prev;
        }
    }

    // Escrever ficheiro comprimido customizado
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "Error opening output file: %s\n", output_path);
        free(audio_bytes);
        free(residuals);
        return 0;
    }

    // Guardar cabeçalhos WAV essenciais e dados dos resíduos
    fwrite(&header, sizeof(WavHeader), 1, fout);
    fwrite(&num_samples, sizeof(long), 1, fout);
    fwrite(residuals, sizeof(int32_t), num_samples, fout);

    fclose(fout);
    free(audio_bytes);
    free(residuals);
    printf("Successfully compressed '%s' -> '%s'\n", input_path, output_path);
    return 1;
}

// Função de Descompressão Lossless (Exatamente igual ao original)
int decode_wav(const char *input_path, const char *output_path) {
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        fprintf(stderr, "Error opening compressed file: %s\n", input_path);
        return 0;
    }

    WavHeader header;
    long num_samples;
    fread(&header, sizeof(WavHeader), 1, fin);
    fread(&num_samples, sizeof(long), 1, fin);

    int32_t *residuals = malloc(num_samples * sizeof(int32_t));
    fread(residuals, sizeof(int32_t), num_samples, fin);
    fclose(fin);

    int sampwidth = header.bits_per_sample / 8;
    int32_t *reconstructed = malloc(num_samples * sizeof(int32_t));

    // Reconstrução por integração (x[n] = resíduo[n] + x[n-1])
    int32_t running_val = 0;
    for (long i = 0; i < num_samples; i++) {
        running_val += residuals[i];
        reconstructed[i] = running_val;
    }

    long data_size = num_samples * sampwidth;
    uint8_t *audio_bytes = malloc(data_size);

    for (long i = 0; i < num_samples; i++) {
        int32_t val = reconstructed[i];
        if (sampwidth == 1) {
            audio_bytes[i] = (uint8_t)(val + 128);
        } else if (sampwidth == 2) {
            ((int16_t*)audio_bytes)[i] = (int16_t)val;
        } else if (sampwidth == 3) {
            long idx = i * 3;
            audio_bytes[idx] = val & 0xFF;
            audio_bytes[idx+1] = (val >> 8) & 0xFF;
            audio_bytes[idx+2] = (val >> 16) & 0xFF;
        } else if (sampwidth == 4) {
            ((int32_t*)audio_bytes)[i] = val;
        }
    }

    // Atualizar tamanho do chunk de dados no header
    header.subchunk2_size = data_size;
    header.chunk_size = 36 + data_size;

    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "Error opening output WAV file: %s\n", output_path);
        free(residuals);
        free(reconstructed);
        free(audio_bytes);
        return 0;
    }

    fwrite(&header, sizeof(WavHeader), 1, fout);
    fwrite(audio_bytes, 1, data_size, fout);
    fclose(fout);

    free(residuals);
    free(reconstructed);
    free(audio_bytes);
    printf("Successfully decompressed '%s' -> '%s'\n", input_path, output_path);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        print_help();
    }

    char *mode = argv[1];
    char *input = NULL;
    char *output = NULL;

    // Ignorar flags opcionais se existirem
    int arg_idx = 2;
    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        arg_idx++; // Ignorar flag
    }

    if (arg_idx + 1 < argc) {
        input = argv[arg_idx];
        output = argv[arg_idx + 1];
    } else {
        print_help();
    }

    if (strcmp(mode, "enc") == 0) {
        encode_wav(input, output);
    } else if (strcmp(mode, "dec") == 0) {
        decode_wav(input, output);
    } else {
        print_help();
    }

    return 0;
}