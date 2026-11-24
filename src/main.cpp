#include "core.hpp"
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <iostream>
// Reutiliza as classes originais para comparar sem alterar barenn.cpp.
#define main original_legacy_main
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#ifdef __CUDACC__
#pragma nv_diag_suppress 940
#endif
#include "../barenn.cpp"
#ifdef __CUDACC__
#pragma nv_diag_default 940
#endif
#pragma GCC diagnostic pop
#undef main
#ifdef BARENN_CUDA
#include "cuda_impl.cuh"
#endif

struct Options {
    std::string backend = "seq", train = "optdigits.tra", test = "optdigits.tes", output;
    int epochs = 51, threads = 1, infer_repeats = 1, block = 128;
};
static int positive_integer(const std::string& value, const std::string& key) {
    size_t end;
    int v = std::stoi(value,&end);
    if (end != value.size() || v <= 0)
        throw std::runtime_error("Inteiro positivo esperado: " + key);
    return v;
}
static Options parse(int argc, char** argv) {
    Options o;
    for (int k = 1; k < argc; ++k) {
        std::string key = argv[k];
        if (key == "--help") {
            std::cout << "--backend legacy|seq|omp|cuda-naive|cuda-global|cuda-shared "
                         "--train FILE --test FILE --epochs N --threads N "
                         "--infer-repeats N --block 64|128|256 --output FILE\n";
            std::exit(0);
        }
        if (k+1 >= argc) throw std::runtime_error("Valor ausente: " + key);
        const std::string value = argv[++k];
        if (key == "--backend") o.backend = value;
        else if (key == "--train") o.train = value;
        else if (key == "--test") o.test = value;
        else if (key == "--output") o.output = value;
        else if (key == "--epochs") o.epochs = positive_integer(value,key);
        else if (key == "--threads") o.threads = positive_integer(value,key);
        else if (key == "--infer-repeats") o.infer_repeats = positive_integer(value,key);
        else if (key == "--block") o.block = positive_integer(value,key);
        else throw std::runtime_error("Opcao desconhecida: " + key);
    }
    if (o.backend != "legacy" && o.backend != "seq" && o.backend != "omp" &&
        o.backend != "cuda-naive" && o.backend != "cuda-global" && o.backend != "cuda-shared")
        throw std::runtime_error("Backend desconhecido: " + o.backend);
    if (o.block != 64 && o.block != 128 && o.block != 256)
        throw std::runtime_error("--block deve ser 64, 128 ou 256");
    return o;
}
template<class T> static void array_json(std::ostream& out, const T& v) {
    out << '[';
    bool first = true;
    for (auto x : v) {
        if (!std::isfinite(static_cast<double>(x))) throw std::runtime_error("Resultado nao finito");
        if (!first) out << ',';
        out << x; first = false;
    }
    out << ']';
}
static void run_legacy(const aps::Data& train, const aps::Data& test, const Options& o,
                       std::vector<double>& losses, std::vector<float>& scores, aps::Timing& time) {
    Barenn::DenseLayer l1(64,32),l2(32,16),l3(16,10,true);
    std::vector<float> grad(10);
    std::vector<std::vector<float>> tx, vx;
    for (const auto& s : train) tx.emplace_back(s.x,s.x+aps::INPUT);
    for (const auto& s : test) vx.emplace_back(s.x,s.x+aps::INPUT);
    auto start = aps::Clock::now();
    losses.resize(o.epochs);
    for (int e = 0; e < o.epochs; ++e) {
        double loss = 0;
        for (size_t i = 0; i < train.size(); ++i) {
            auto out = l3.Forward(l2.Forward(l1.Forward(tx[i])));
            loss += Barenn::perda(out,train[i].label,&grad);
            auto g3 = l3.Backward(grad,aps::RATE);
            auto g2 = l2.Backward(g3,aps::RATE);
            l1.Backward(g2,aps::RATE);
        }
        losses[e] = loss / train.size();
    }
    time.train_s = aps::seconds(start);
    scores.resize(test.size()*10);
    start = aps::Clock::now();
    for (int r = 0; r < o.infer_repeats; ++r)
        for (size_t i = 0; i < test.size(); ++i) {
            auto out = l3.Forward(l2.Forward(l1.Forward(vx[i])));
            std::copy(out.begin(),out.end(),scores.begin()+i*10);
        }
    time.infer_s = aps::seconds(start);
}
int main(int argc, char** argv) {
    try {
        const auto o = parse(argc,argv);
        if (std::filesystem::equivalent(o.train,o.test))
            throw std::runtime_error("Use arquivos distintos para treino e teste");
        const auto train = aps::read_data(o.train), test = aps::read_data(o.test);
        auto model = aps::initialize();
        std::vector<double> losses;
        std::vector<float> scores;
        aps::Timing timing;
        if (o.backend == "legacy") run_legacy(train,test,o,losses,scores,timing);
        else if (o.backend == "seq" || o.backend == "omp") {
            auto start = aps::Clock::now();
            aps::train_cpu(model,train,o.epochs,losses);
            timing.train_s = aps::seconds(start);
            start = aps::Clock::now();
            scores = aps::infer_cpu(model,test,o.backend == "omp" ? o.threads : 1,o.infer_repeats);
            timing.infer_s = aps::seconds(start);
        } else {
#ifdef BARENN_CUDA
            aps::run_cuda(model,train,test,o.epochs,o.infer_repeats,o.block,o.backend,losses,scores,timing);
#else
            throw std::runtime_error("Compile src/cuda_main.cu com nvcc e execute em uma GPU NVIDIA; veja o README");
#endif
        }
        size_t correct = 0;
        double loss_test = 0;
        std::vector<int> predictions;
        std::array<size_t,100> confusion{};
        for (size_t i = 0; i < test.size(); ++i) {
            const float* row = scores.data()+i*10;
            const int pred = static_cast<int>(std::max_element(row,row+10)-row);
            predictions.push_back(pred);
            correct += pred == test[i].label;
            ++confusion[test[i].label*10+pred];
            float loss = 0;
            for (int j = 0; j < 10; ++j) {
                float d = row[j]-(j == test[i].label ? 1.0f:0.0f);
                loss += 0.5f*d*d;
            }
            loss_test += loss;
        }
        std::cout << std::setprecision(10) << "backend=" << o.backend << " train_s=" << timing.train_s
                  << " infer_s=" << timing.infer_s << " correct=" << correct << '/' << test.size()
                  << " test_loss=" << loss_test/test.size() << '\n';
        if (!o.output.empty()) {
            std::ofstream f(o.output);
            if (!f) throw std::runtime_error("Nao foi possivel criar " + o.output);
            f << std::setprecision(17) << "{\"backend\":\"" << o.backend << "\",\"epochs\":" << o.epochs
              << ",\"threads\":" << (o.backend == "omp" ? o.threads : 1) << ",\"block\":" << o.block
              << ",\"infer_repeats\":" << o.infer_repeats << ",\"train_count\":" << train.size()
              << ",\"test_count\":" << test.size() << ",\"train_s\":" << timing.train_s
              << ",\"infer_s\":" << timing.infer_s << ",\"setup_s\":" << timing.setup_s
              << ",\"h2d_ms\":" << timing.h2d_ms << ",\"train_device_ms\":" << timing.train_device_ms
              << ",\"infer_device_ms\":" << timing.infer_device_ms << ",\"d2h_ms\":" << timing.d2h_ms
              << ",\"correct\":" << correct << ",\"test_loss\":" << loss_test/test.size()
              << ",\"epoch_losses\":";
            array_json(f,losses); f << ",\"scores\":"; array_json(f,scores);
            f << ",\"predictions\":"; array_json(f,predictions);
            f << ",\"confusion\":"; array_json(f,confusion);
            if (o.backend != "legacy") { f << ",\"parameters\":"; array_json(f,model.p); }
            f << "}\n";
            if (!f) throw std::runtime_error("Falha ao gravar resultado");
        }
    } catch (const std::exception& e) { std::cerr << "Erro: " << e.what() << '\n'; return 1; }
}
