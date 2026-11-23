#include "../src/core.hpp"
#include <iostream>
#define main original_legacy_main
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#include "../barenn.cpp"
#pragma GCC diagnostic pop
#undef main

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        const auto samples = aps::read_data("optdigits.tra");
        auto m = aps::initialize();
        Barenn::DenseLayer l1(64,32), l2(32,16), l3(16,10,true);
        std::vector<float> grad(10);
        aps::Workspace workspace;
        // Comparacao passo a passo com classes originais, em todos os rotulos.
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 0; i < 256; ++i) {
                const auto& s = samples[i];
                std::vector<float> x(s.x,s.x+aps::INPUT);
                const auto ref = l3.Forward(l2.Forward(l1.Forward(x)));
                aps::forward(m,s.x,workspace);
                for (int j = 0; j < 10; ++j)
                    require(ref[j] == workspace.out[j],"Forward diferente do legado");
                const auto reference_loss = Barenn::perda(ref,s.label,&grad);
                const auto actual_loss = aps::step(m,s,workspace);
                require(reference_loss == actual_loss,"Perda diferente do legado");
                auto g3 = l3.Backward(grad,aps::RATE);
                auto g2 = l2.Backward(g3,aps::RATE);
                l1.Backward(g2,aps::RATE);
            }
        }
        aps::Data small(samples.begin(),samples.begin()+33);
        const auto serial = aps::infer_cpu(m,small,1,1);
        for (int threads : {2,4,8})
            require(serial == aps::infer_cpu(m,small,threads,2),"Inferencia paralela divergente");
        // Gradiente numerico independente em um parametro da ultima camada.
        const auto s = samples[0];
        const auto before = m;
        aps::step(m,s,workspace);
        constexpr float eps = 0.001f;
        const int k = aps::W3+3;
        auto loss = [&](aps::Model candidate) {
            aps::forward(candidate,s.x,workspace);
            double total = 0;
            for (int j=0;j<10;++j) {
                double d = workspace.out[j]-(j==s.label?1.0:0.0);
                total += 0.5*d*d;
            }
            return total;
        };
        auto plus=before,minus=before;
        plus.p[k]+=eps; minus.p[k]-=eps;
        double numeric=(loss(plus)-loss(minus))/(2*eps);
        double analytic=(before.p[k]-m.p[k])/aps::RATE;
        require(std::abs(numeric-analytic)<0.0002,"Gradiente numerico incorreto");
        std::cout << "PASS: 512 passos SGD, forward, perda, gradiente numerico e inferencia 1/2/4/8 threads\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
