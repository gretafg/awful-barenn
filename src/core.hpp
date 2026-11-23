#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace aps {
constexpr int INPUT = 64, H1 = 32, H2 = 16, OUTPUT = 10;
constexpr int W1 = 0, B1 = 2048, W2 = 2080, B2 = 2592;
constexpr int W3 = 2608, B3 = 2768, PARAMS = 2778;
constexpr float RATE = 0.02f;
struct Sample { float x[INPUT]{}; int label = 0; };
using Data = std::vector<Sample>;
struct Model { std::array<float, PARAMS> p{}; };
struct Workspace {
    float a[H1]{}, b[H2]{}, out[OUTPUT]{};
    float d1[H1]{}, d2[H2]{}, d3[OUTPUT]{};
};
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Timing {
    double train_s = 0, infer_s = 0;
    double setup_s = 0, h2d_ms = 0, train_device_ms = 0;
    double infer_device_ms = 0, d2h_ms = 0;
};

inline Data read_data(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Nao foi possivel abrir " + path);
    Data data;
    std::string line;
    size_t number = 0;
    while (std::getline(f, line)) {
        ++number;
        std::istringstream row(line);
        Sample s;
        for (int j = 0; j < INPUT; ++j) {
            float v; char comma;
            if (!(row >> v >> comma) || comma != ',' || !std::isfinite(v) || v < 0 || v > 16)
                throw std::runtime_error(path + ":" + std::to_string(number) + ": atributo invalido");
            s.x[j] = v / 16.0f;
        }
        if (!(row >> s.label) || s.label < 0 || s.label >= OUTPUT || !(row >> std::ws).eof())
            throw std::runtime_error(path + ":" + std::to_string(number) + ": rotulo/formato invalido");
        data.push_back(s);
    }
    if (f.bad() || data.empty()) throw std::runtime_error("Falha de leitura ou arquivo vazio: " + path);
    return data;
}

inline Model initialize() {
    Model m;
    // Cada camada reinicia o gerador, exatamente como no legado.
    const std::array<std::pair<int,int>,3> segments{{{W1,INPUT*H1},{W2,H1*H2},{W3,H2*OUTPUT}}};
    for (auto segment : segments) {
        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist(-0.05f,0.05f);
        for (int k = 0; k < segment.second; ++k) m.p[segment.first+k] = dist(rng);
    }
    return m;
}

template<int In, int Out, bool Relu>
inline void dense(const float* w, const float* bias, const float* x, float* y) {
    for (int i = 0; i < Out; ++i) {
        float sum = bias[i];
        for (int j = 0; j < In; ++j) sum += x[j] * w[i*In+j];
        y[i] = Relu ? std::max(0.0f,sum) : sum;
    }
}
inline void forward(const Model& m, const float* x, Workspace& s) {
    dense<INPUT,H1,true>(m.p.data()+W1,m.p.data()+B1,x,s.a);
    dense<H1,H2,true>(m.p.data()+W2,m.p.data()+B2,s.a,s.b);
    dense<H2,OUTPUT,false>(m.p.data()+W3,m.p.data()+B3,s.b,s.out);
}
template<int In, int Out>
inline void update(float* weights, float* bias, const float* input, const float* delta) {
    for (int i = 0; i < Out; ++i) {
        for (int j = 0; j < In; ++j) weights[i*In+j] -= RATE * (delta[i] * input[j]);
        bias[i] -= RATE * delta[i];
    }
}
inline float step(Model& m, const Sample& sample, Workspace& s) {
    forward(m,sample.x,s);
    float loss = 0;
    for (int i = 0; i < OUTPUT; ++i) {
        s.d3[i] = s.out[i] - (sample.label == i ? 1.0f : 0.0f);
        loss += 0.5f * s.d3[i] * s.d3[i];
    }
    for (int i = 0; i < H2; ++i) {
        float grad = 0;
        for (int j = 0; j < OUTPUT; ++j) grad += s.d3[j] * m.p[W3+j*H2+i];
        s.d2[i] = grad * (s.b[i] > 0 ? 1.0f : 0.0f);
    }
    for (int i = 0; i < H1; ++i) {
        float grad = 0;
        for (int j = 0; j < H2; ++j) grad += s.d2[j] * m.p[W2+j*H1+i];
        s.d1[i] = grad * (s.a[i] > 0 ? 1.0f : 0.0f);
    }
    // Todos os gradientes consumiram os pesos antigos.
    update<H2,OUTPUT>(m.p.data()+W3,m.p.data()+B3,s.b,s.d3);
    update<H1,H2>(m.p.data()+W2,m.p.data()+B2,s.a,s.d2);
    update<INPUT,H1>(m.p.data()+W1,m.p.data()+B1,sample.x,s.d1);
    // O gradiente da entrada nao e calculado: o legado o descarta.
    return loss;
}
inline void train_cpu(Model& m, const Data& data, int epochs, std::vector<double>& losses) {
    Workspace s;
    losses.resize(epochs);
    for (int e = 0; e < epochs; ++e) {
        double loss = 0;
        for (const auto& sample : data) loss += step(m,sample,s);
        losses[e] = loss / data.size();
    }
}
inline std::vector<float> infer_cpu(const Model& m, const Data& data, int threads, int repeats) {
    std::vector<float> scores(data.size()*OUTPUT);
    for (int r = 0; r < repeats; ++r) {
        #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
        for (size_t i = 0; i < data.size(); ++i) {
            Workspace s;
            forward(m,data[i].x,s);
            std::copy(s.out,s.out+OUTPUT,scores.data()+i*OUTPUT);
        }
    }
    return scores;
}
} // namespace aps
