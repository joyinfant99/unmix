// unmix-stretch: offline time stretch and pitch shift of raw float32 audio with Rubber Band (R3 "finer" engine).
// usage: unmix-stretch in.f32 out.f32 channels samplerate tempo semitones formant(0|1)
// Raw interleaved little-endian float32 in and out. Part of Unmix (MIT); links Rubber Band (GPL-2.0-or-later).
#include "rubberband/RubberBandStretcher.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace RubberBand;

int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr, "usage: unmix-stretch in out channels rate tempo semitones formant\n"); return 2; }
    size_t ch = atoi(argv[3]), rate = atoi(argv[4]);
    double tempo = atof(argv[5]), semis = atof(argv[6]);
    bool formant = atoi(argv[7]) != 0;
    FILE *fi = fopen(argv[1], "rb");
    if (!fi || ch < 1 || ch > 2 || tempo <= 0) return 2;
    fseek(fi, 0, SEEK_END); size_t n = ftell(fi) / (sizeof(float) * ch); fseek(fi, 0, SEEK_SET);
    std::vector<float> inter(n * ch);
    if (fread(inter.data(), sizeof(float), inter.size(), fi) != inter.size()) return 3;
    fclose(fi);
    std::vector<std::vector<float>> in(ch, std::vector<float>(n));
    for (size_t i = 0; i < n; i++) for (size_t c = 0; c < ch; c++) in[c][i] = inter[i * ch + c];

    int opts = RubberBandStretcher::OptionProcessOffline | RubberBandStretcher::OptionEngineFiner |
               RubberBandStretcher::OptionChannelsTogether | RubberBandStretcher::OptionPitchHighQuality;
    if (formant) opts |= RubberBandStretcher::OptionFormantPreserved;
    RubberBandStretcher rb(rate, ch, opts, 1.0 / tempo, std::pow(2.0, semis / 12.0));
    rb.setExpectedInputDuration(n);
    std::vector<const float *> ip(ch);
    const size_t block = 8192;
    for (size_t pos = 0; pos < n; pos += block) {  // study pass
        size_t k = std::min(block, n - pos);
        for (size_t c = 0; c < ch; c++) ip[c] = in[c].data() + pos;
        rb.study(ip.data(), k, pos + k >= n);
    }
    FILE *fo = fopen(argv[2], "wb");
    if (!fo) return 4;
    std::vector<std::vector<float>> out(ch, std::vector<float>(block));
    std::vector<float *> op(ch);
    std::vector<float> wr;
    auto drain = [&]() {
        int avail;
        while ((avail = rb.available()) > 0) {
            size_t k = std::min((size_t)avail, block);
            for (size_t c = 0; c < ch; c++) op[c] = out[c].data();
            k = rb.retrieve(op.data(), k);
            wr.resize(k * ch);
            for (size_t i = 0; i < k; i++) for (size_t c = 0; c < ch; c++) wr[i * ch + c] = out[c][i];
            fwrite(wr.data(), sizeof(float), wr.size(), fo);
        }
    };
    for (size_t pos = 0; pos < n; pos += block) {  // process pass
        size_t k = std::min(block, n - pos);
        for (size_t c = 0; c < ch; c++) ip[c] = in[c].data() + pos;
        rb.process(ip.data(), k, pos + k >= n);
        drain();
    }
    drain();
    fclose(fo);
    return 0;
}
