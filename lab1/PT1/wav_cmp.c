/*
 * wav_cmp - Compara um ficheiro WAV original com uma versão modificada.
 *
 * Uso:  wav_cmp <original.wav> <modificado.wav>
 *
 * Para cada canal (L, R, MID, SIDE) calcula, sendo x a amostra original,
 * y a amostra modificada e "e = x - y":
 *
 *   MSE   = (1/N) * soma(e^2)                  erro quadrático médio (L2)
 *   L_inf = max |e|                            erro absoluto máximo (L-inf)
 *   SNR   = 10 * log10( soma(x^2) / soma(e^2) )   em dB
 *
 * MID = (L+R)/2 e SIDE = (L-R)/2, com divisão inteira (como em wav_hist).
 * Em ficheiros mono só existem L e MID.
 * Suporta WAV PCM de 16 bits, mono ou estéreo.
 */
    
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK_FRAMES 4096    /* frames lidas de cada vez, por ficheiro */

enum { CH_L, CH_R, CH_MID, CH_SIDE, NUM_CH };
static const char *CH_NAMES[NUM_CH] = { "L", "R", "MID", "SIDE" };

typedef struct {
    unsigned channels;       /* 1 ou 2                                  */
    uint32_t sample_rate;
    uint32_t num_frames;
} WavInfo;

/* Acumuladores das métricas de um canal (somas exatas em inteiros). */
typedef struct {
    uint64_t sum_sig2;       /* soma de x^2 (energia do sinal original) */
    uint64_t sum_err2;       /* soma de e^2 (energia do erro)           */
    uint32_t max_err;        /* max |e|                                 */
} Metrics;

/* ------------------------------------------------------------------ */
/* Leitura de inteiros little-endian (independente da máquina)        */
/* ------------------------------------------------------------------ */
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* ------------------------------------------------------------------ */
/* Cabeçalho WAV                                                      */
/* ------------------------------------------------------------------ */

/* Lê o cabeçalho e deixa o ficheiro posicionado no início dos dados.
 * Devolve 0 (top), -1 (algum não tá bem). */
static int wav_parse(FILE *f, const char *name, WavInfo *w){
    uint8_t hdr[12], ck[8];
    int have_fmt = 0;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 ||
        memcmp(hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "Erro: '%s' não é um ficheiro WAV válido.\n", name);
        return -1;
    }

    /* Percorre os chunks até encontrar "fmt" e "data". */
    while (fread(ck, 1, 8, f) == 8) {
        uint32_t size = le32(ck + 4);

        if (memcmp(ck, "fmt ", 4) == 0) {
            uint8_t fmt[16], ext[10];
            uint32_t consumed = 16;

            if (size < 16 || fread(fmt, 1, 16, f) != 16) {
                fprintf(stderr, "Erro: chunk 'fmt ' inválido em '%s'.\n", name);
                return -1;
            }
            unsigned tag  = le16(fmt);
            unsigned bits = le16(fmt + 14);
            w->channels    = le16(fmt + 2);
            w->sample_rate = le32(fmt + 4);

            /* WAVE_FORMAT_EXTENSIBLE: o formato real está no sub-formato. */
            if (tag == 0xFFFE && size >= 26 && fread(ext, 1, 10, f) == 10) {
                tag = le16(ext + 8);
                consumed = 26;
            }
            if (tag != 1 || bits != 16) {
                fprintf(stderr, "Erro: '%s' não é PCM de 16 bits.\n", name);
                return -1;
            }
            if (w->channels < 1 || w->channels > 2) {
                fprintf(stderr, "Erro: '%s' tem %u canais (só mono/estéreo).\n",
                        name, w->channels);
                return -1;
            }
            have_fmt = 1;
            fseek(f, (long)(size - consumed + (size & 1)), SEEK_CUR);

        } else if (memcmp(ck, "data", 4) == 0) {
            if (!have_fmt) {
                fprintf(stderr, "Erro: '%s': 'data' antes de 'fmt '.\n", name);
                return -1;
            }
            long avail = file_size - ftell(f);   /* protege contra tamanhos falsos */
            if ((long)size > avail) size = (uint32_t)avail;
            w->num_frames = size / (w->channels * 2);
            return 0;

        } else {                                 /* chunk desconhecido: salta */
            fseek(f, (long)(size + (size & 1)), SEEK_CUR);
        }
    }
    fprintf(stderr, "Erro: chunk 'data' não encontrado em '%s'.\n", name);
    return -1;
}

/* ------------------------------------------------------------------ */
/* Métricas                                                            */
/* ------------------------------------------------------------------ */

/* Extrai de uma frame os valores de L, R, MID e SIDE.
 * Em mono, R = L (só L e MID são usados). */
static void frame_channels(const uint8_t *p, unsigned channels, int v[NUM_CH]){
    int l = (int16_t)le16(p);
    int r = (channels == 2) ? (int16_t)le16(p + 2) : l;

    v[CH_L]    = l;
    v[CH_R]    = r;
    v[CH_MID]  = (l + r) / 2;        /* divisão inteira */
    v[CH_SIDE] = (l - r) / 2;        /* divisão inteira */
}

/* Acumula um par (x original, y modificada) nas métricas de um canal. */
static void metrics_add(Metrics *m, int x, int y)
{
    int e = x - y;
    uint32_t a = (uint32_t)(e < 0 ? -e : e);

    m->sum_sig2 += (uint64_t)((int64_t)x * x);
    m->sum_err2 += (uint64_t)a * a;
    if (a > m->max_err) m->max_err = a;
}

/* Percorre os dois ficheiros em paralelo (n frames) e preenche m[]. */
static void compute_metrics(FILE *fx, FILE *fy, unsigned channels, uint32_t n, Metrics m[NUM_CH]){
    uint8_t bx[CHUNK_FRAMES * 4], by[CHUNK_FRAMES * 4];
    unsigned frame_bytes = channels * 2;

    while (n > 0) {
        size_t want = n < CHUNK_FRAMES ? n : CHUNK_FRAMES;
        size_t gx = fread(bx, frame_bytes, want, fx);
        size_t gy = fread(by, frame_bytes, want, fy);
        size_t got = gx < gy ? gx : gy;
        if (got == 0) break;

        for (size_t i = 0; i < got; i++) {
            int x[NUM_CH], y[NUM_CH];
            frame_channels(bx + i * frame_bytes, channels, x);
            frame_channels(by + i * frame_bytes, channels, y);

            for (int c = 0; c < NUM_CH; c++) {
                if (channels == 1 && (c == CH_R || c == CH_SIDE)) continue;
                metrics_add(&m[c], x[c], y[c]);
            }
        }
        n -= (uint32_t)got;
    }
}

/* ------------------------------------------------------------------ */
/* Apresentação                                                       */
/* ------------------------------------------------------------------ */

static void print_table(const Metrics m[NUM_CH], unsigned channels, uint32_t n){
    printf("%-6s | %14s | %8s | %10s\n", "Canal", "MSE", "L_inf", "SNR (dB)");
    printf("-------+----------------+----------+-----------\n");

    for (int c = 0; c < NUM_CH; c++) {
        if (channels == 1 && (c == CH_R || c == CH_SIDE)) continue;

        double mse = (double)m[c].sum_err2 / n;
        char snr[16];

        if (m[c].sum_err2 == 0)                 /* sem erro: SNR infinita */
            snprintf(snr, sizeof snr, "inf");
        else if (m[c].sum_sig2 == 0)            /* original em silêncio   */
            snprintf(snr, sizeof snr, "-inf");
        else
            snprintf(snr, sizeof snr, "%.2f",
                     10.0 * log10((double)m[c].sum_sig2 / (double)m[c].sum_err2));

        printf("%-6s | %14.4f | %8u | %10s\n",
               CH_NAMES[c], mse, (unsigned)m[c].max_err, snr);
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    WavInfo wx, wy;
    Metrics m[NUM_CH];
    int status = 1;

    if (argc != 3) {
        fprintf(stderr, "Uso: %s <original.wav> <modificado.wav>\n", argv[0]);
        return 1;
    }

    FILE *fx = fopen(argv[1], "rb");
    if (!fx) { perror(argv[1]); return 1; }
    FILE *fy = fopen(argv[2], "rb");
    if (!fy) { perror(argv[2]); fclose(fx); return 1; }

    if (wav_parse(fx, argv[1], &wx) != 0 || wav_parse(fy, argv[2], &wy) != 0)
        goto done;

    if (wx.channels != wy.channels) {
        fprintf(stderr, "Erro: nº de canais diferente (%u vs %u).\n",
                wx.channels, wy.channels);
        goto done;
    }
    if (wx.sample_rate != wy.sample_rate)
        fprintf(stderr, "Aviso: taxas de amostragem diferentes (%u vs %u Hz).\n",
                (unsigned)wx.sample_rate, (unsigned)wy.sample_rate);

    /* Se os comprimentos diferirem, compara-se só a parte comum. */
    uint32_t n = wx.num_frames < wy.num_frames ? wx.num_frames : wy.num_frames;
    if (wx.num_frames != wy.num_frames)
        fprintf(stderr, "Aviso: nº de frames diferente (%u vs %u); "
                "comparadas as primeiras %u.\n",
                (unsigned)wx.num_frames, (unsigned)wy.num_frames, (unsigned)n);
    if (n == 0) {
        fprintf(stderr, "Erro: sem amostras para comparar.\n");
        goto done;
    }

    memset(m, 0, sizeof m);
    compute_metrics(fx, fy, wx.channels, n, m);

    printf("Original:   %s\nModificado: %s\n", argv[1], argv[2]);
    printf("%u canal(is), %u Hz, %u frames comparadas\n\n",
           wx.channels, (unsigned)wx.sample_rate, (unsigned)n);
    print_table(m, wx.channels, n);
    status = 0;

done:
    fclose(fx);
    fclose(fy);
    return status;
}