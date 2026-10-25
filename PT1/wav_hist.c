/*
 * wav_hist - Histograma em modo texto de ficheiros WAV
 *
 * Uso:  wav_hist <ficheiro.wav> <canal> [k]
 *
 * canal : L | R | MID | SIDE | ALL
 *         MID  = (L+R)/2   (divisão inteira)
 *         SIDE = (L-R)/2   (divisão inteira)
 *         ALL  = todos os canais disponíveis (L,R,MID,SIDE em estéreo;
 *                                             L,MID em mono)
 *   k   : cada bin agrupa 2^k valores consecutivos (0 <= k <= 16).
 *         Se omitido, escolhe-se o menor k que dê no máximo 64 bins.
 *
 * Para cada canal processado, o que é mostrado no terminal é também
 * guardado em <canal>.txt (ex.: L.txt, R.txt, MID.txt, SIDE.txt, ALL.txt).
 * Se o ficheiro já existir, é substituído.
 *
 * Suporta ficheiros mono e estéreo.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define NUM_VALUES   65536   /* valores possíveis de 16 bits                 */
#define OFFSET       32768   /* desloca os valores para índices 0..65535     */
#define MAX_K        16      /* 2^16 valores => um único bin                 */
#define AUTO_BINS    64      /* nº máximo de bins no modo automático         */
#define BAR_WIDTH    50      /* largura máxima das barras                    */
#define CHUNK_FRAMES 4096    /* frames lidas de cada vez                     */

typedef enum { CH_LEFT, CH_RIGHT, CH_MID, CH_SIDE, CH_ALL } Channel;

#define NUM_CHANNELS 4       /* canais reais (sem CH_ALL) */
static const char *CHANNEL_NAMES[NUM_CHANNELS] = { "L", "R", "MID", "SIDE" };

typedef struct {
    unsigned channels;       /* 1 ou 2                                       */
    uint32_t sample_rate;
    uint32_t data_bytes;     /* tamanho útil do chunk "data"                 */
    uint32_t num_frames;     /* data_bytes / (channels * 2)                  */
    long     data_offset;    /* posição do início dos dados no ficheiro      */
} WavInfo;

/* ------------------------------------------------------------------ */
/* Leitura de inteiros little-endian (independente da máquina)        */
/* ------------------------------------------------------------------ */
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* ------------------------------------------------------------------ */
/* Cabeçalho WAV                                                      */
/* ------------------------------------------------------------------ */

/* Lê o cabeçalho e deixa o ficheiro posicionado no início dos dados.
 * Devolve 0 se tudo correr bem, -1 caso isto dê para o torto. */
static int wav_parse(FILE *f, WavInfo *w){
    uint8_t hdr[12], ck[8];
    int have_fmt = 0;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 ||
        memcmp(hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "Erro: não é um ficheiro WAV válido.\n");
        return -1;
    }

    /* Percorre os chunks até encontrar "fmt" e "data". */
    while (fread(ck, 1, 8, f) == 8) {
        uint32_t size = le32(ck + 4);

        if (memcmp(ck, "fmt ", 4) == 0) {
            uint8_t fmt[16], ext[10];
            uint32_t consumed = 16;

            if (size < 16 || fread(fmt, 1, 16, f) != 16) {
                fprintf(stderr, "Erro: chunk 'fmt ' inválido.\n");
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
                fprintf(stderr, "Erro: apenas é suportado PCM de 16 bits.\n");
                return -1;
            }
            if (w->channels < 1 || w->channels > 2) {
                fprintf(stderr, "Erro: apenas mono ou estéreo (%u canais).\n",
                        w->channels);
                return -1;
            }
            have_fmt = 1;
            fseek(f, (long)(size - consumed + (size & 1)), SEEK_CUR);

        } else if (memcmp(ck, "data", 4) == 0) {
            if (!have_fmt) {
                fprintf(stderr, "Erro: chunk 'data' antes de 'fmt '.\n");
                return -1;
            }
            long avail = file_size - ftell(f);   /* protege contra tamanhos falsos */
            if ((long)size > avail) size = (uint32_t)avail;
            w->data_bytes = size;
            w->num_frames = size / (w->channels * 2);
            w->data_offset = ftell(f);
            return 0;

        } else {                                 /* chunk desconhecido: salta */
            fseek(f, (long)(size + (size & 1)), SEEK_CUR);
        }
    }
    fprintf(stderr, "Erro: chunk 'data' não encontrado.\n");
    return -1;
}

/* ------------------------------------------------------------------ */
/* Histograma                                                          */
/* ------------------------------------------------------------------ */

/* Conta as ocorrências de cada valor do canal escolhido.
 * hist[v + OFFSET] = nº de amostras com valor v. */
static void compute_histogram(FILE *f, const WavInfo *w, Channel ch, uint64_t *hist){
    uint8_t raw[CHUNK_FRAMES * 4];
    unsigned frame_bytes = w->channels * 2;
    uint32_t left = w->num_frames;

    while (left > 0) {
        size_t want = left < CHUNK_FRAMES ? left : CHUNK_FRAMES;
        size_t got  = fread(raw, frame_bytes, want, f);
        if (got == 0) break;

        for (size_t i = 0; i < got; i++) {
            const uint8_t *p = raw + i * frame_bytes;
            int l = (int16_t)le16(p);
            int r = (w->channels == 2) ? (int16_t)le16(p + 2) : l;
            int v;

            switch (ch) {
            case CH_LEFT:  v = l;           break;
            case CH_RIGHT: v = r;           break;
            case CH_MID:   v = (l + r) / 2; break;   /* divisão inteira */
            default:       v = (l - r) / 2; break;   /* divisão inteira */
            }
            hist[v + OFFSET]++;
        }
        left -= (uint32_t)got;
    }
}

/* Menor k tal que o intervalo [lo, hi] de valores ocupados cabe em
 * AUTO_BINS bins. */
static int auto_k(int lo, int hi){
    for (int k = 0; k < MAX_K; k++)
        if ((hi >> k) - (lo >> k) + 1 <= AUTO_BINS) return k;
    return MAX_K;
}

/* Desenha o histograma, um bin por linha, com barras horizontais.
 * O bin b agrupa os valores [b*2^k, (b+1)*2^k - 1] (em índices deslocados),
 * o que equivale a uma divisão inteira (floor) por 2^k. */
static void print_histogram(FILE *out, const uint64_t *hist, int k, uint64_t total){
    static uint64_t bins[NUM_VALUES];
    int lo = -1, hi = -1;

    memset(bins, 0, sizeof bins);            /* pode ser chamada várias vezes */

    for (int i = 0; i < NUM_VALUES; i++)
        if (hist[i]) { if (lo < 0) lo = i; hi = i; }
    if (lo < 0) { fprintf(out, "Sem amostras.\n"); return; }

    if (k < 0) k = auto_k(lo, hi);

    uint64_t max = 0;
    for (int i = 0; i < NUM_VALUES; i++) bins[i >> k] += hist[i];

    int first = lo >> k, last = hi >> k;
    for (int b = first; b <= last; b++)
        if (bins[b] > max) max = bins[b];

    fprintf(out, "Amostras: %llu | bin = %d valor(es) (k=%d) | máx. por bin: %llu\n\n",
           (unsigned long long)total, 1 << k, k, (unsigned long long)max);

    for (int b = first; b <= last; b++) {
        int v0 = (b << k) - OFFSET;          /* primeiro valor do bin */
        int v1 = v0 + (1 << k) - 1;          /* último valor do bin   */
        int len = (int)(bins[b] * BAR_WIDTH / max);
        if (bins[b] > 0 && len == 0) len = 1;    /* bins não vazios visíveis */

        if (k == 0) fprintf(out, "%7d         |", v0);
        else        fprintf(out, "[%6d,%6d] |", v0, v1);
        for (int i = 0; i < len; i++) fputc('#', out);
        fprintf(out, "%*s %8llu\n", BAR_WIDTH - len, "",
               (unsigned long long)bins[b]);
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

/* Escreve o relatório completo (cabeçalho + histograma) num stream. */
static void write_report(FILE *out, const char *wavname, const WavInfo *w, const char *chname, const uint64_t *hist, int k){
    fprintf(out, "%s: %u canal(is), %u Hz, %u frames | canal: %s\n",
            wavname, w->channels, (unsigned)w->sample_rate,
            (unsigned)w->num_frames, chname);
    print_histogram(out, hist, k, w->num_frames);
}

/* Calcula o histograma do canal, mostra-o no terminal e guarda-o em
 * <canal>.txt (um ficheiro existente com o mesmo nome é substituído). */
static int run_channel(FILE *f, const WavInfo *w, Channel ch, const char *wavname, int k){
    static uint64_t hist[NUM_VALUES];
    const char *name = CHANNEL_NAMES[ch];
    char path[16];

    memset(hist, 0, sizeof hist);
    fseek(f, w->data_offset, SEEK_SET);      /* volta ao início dos dados */
    compute_histogram(f, w, ch, hist);

    snprintf(path, sizeof path, "%s.txt", name);
    remove(path);                            /* apaga o anterior, se existir */
    FILE *out = fopen(path, "w");
    if (!out) { perror(path); return -1; }

    write_report(stdout, wavname, w, name, hist, k);
    write_report(out, wavname, w, name, hist, k);
    fclose(out);

    printf("-> guardado em %s\n", path);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static int parse_channel(const char *s, Channel *ch)
{
    char u[8];
    size_t n = strlen(s);
    if (n >= sizeof u) return -1;
    for (size_t i = 0; i <= n; i++) u[i] = (char)toupper((unsigned char)s[i]);

    if      (!strcmp(u, "L"))    *ch = CH_LEFT;
    else if (!strcmp(u, "R"))    *ch = CH_RIGHT;
    else if (!strcmp(u, "MID"))  *ch = CH_MID;
    else if (!strcmp(u, "SIDE")) *ch = CH_SIDE;
    else if (!strcmp(u, "ALL"))  *ch = CH_ALL;
    else return -1;
    return 0;
}

int main(int argc, char **argv)
{
    WavInfo w;
    Channel ch;
    int k = -1;                              /* -1 = "automático" */
    int status = 0;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s <ficheiro.wav> <L|R|MID|SIDE|ALL> [k]\n",
                argv[0]);
        return 1;
    }
    if (parse_channel(argv[2], &ch) != 0) {
        fprintf(stderr,
                "Erro: canal inválido '%s' (use L, R, MID, SIDE ou ALL).\n",
                argv[2]);
        return 1;
    }
    if (argc == 4) {
        char *end;
        long v = strtol(argv[3], &end, 10);
        if (*end || v < 0 || v > MAX_K) {
            fprintf(stderr, "Erro: k deve estar entre 0 e %d.\n", MAX_K);
            return 1;
        }
        k = (int)v;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }

    if (wav_parse(f, &w) != 0) { fclose(f); return 1; }

    if (w.channels == 1 && (ch == CH_RIGHT || ch == CH_SIDE)) {
        fprintf(stderr, "Erro: ficheiro mono só tem os canais L e MID.\n");
        fclose(f);
        return 1;
    }

    if (ch == CH_ALL) {
        int first = 1;
        for (int c = 0; c < NUM_CHANNELS; c++) {
            /* em mono só existem L e MID */
            if (w.channels == 1 && (c == CH_RIGHT || c == CH_SIDE)) continue;
            if (!first) printf("\n");
            first = 0;
            if (run_channel(f, &w, (Channel)c, argv[1], k) != 0) status = 1;
        }
    } else {
        status = (run_channel(f, &w, ch, argv[1], k) != 0);
    }

    fclose(f);
    return status;
}