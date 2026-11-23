// Testa a logica dos kernels com threads/barreiras da CPU.
// Nao simula memoria, escalonamento, performance ou verificacao de races da GPU.
#include "../src/core.hpp"
#include <barrier>
#include <thread>
#include <iostream>
#include <functional>
struct Index { unsigned x=0,y=0,z=0; };
thread_local Index threadIdx;
Index blockIdx,blockDim;
std::barrier<>* active_barrier=nullptr;
inline void __syncthreads() { active_barrier->arrive_and_wait(); }
#define __global__
#define __shared__ static
#define APS_KERNEL_TEST
#include "../src/cuda_impl.cuh"
#undef __global__
#undef __shared__

static void launch(unsigned x,unsigned y,const std::function<void()>& kernel) {
    blockDim={x,y,1};
    std::barrier barrier(static_cast<std::ptrdiff_t>(x*y));
    active_barrier=&barrier;
    std::vector<std::thread> workers;
    for(unsigned i=0;i<x*y;++i)
        workers.emplace_back([&,i] {threadIdx={i%x,i/x,0};kernel();});
    for(auto& worker:workers) worker.join();
}
static void same(float a,float b) {
    if(a!=b) throw std::runtime_error("Kernels divergiram da referencia CPU");
}
int main() {
    try {
        auto raw=aps::read_data("optdigits.tra");
        aps::Data train(raw.begin(),raw.begin()+3), test(raw.begin(),raw.begin()+17);
        auto reference=aps::initialize();
        std::vector<double> reference_loss;
        aps::train_cpu(reference,train,2,reference_loss);
        auto expected_scores=aps::infer_cpu(reference,test,1,1);
        for(unsigned block: {64,128,256}) {
            for(bool shared: {false,true}) {
                auto m=aps::initialize();
                double loss[2]{};
                blockIdx={0,0,0};
                launch(block,1,[&] {
                    if(shared) aps::train_kernel<true>(m.p.data(),train.data(),3,2,loss,false);
                    else aps::train_kernel<false>(m.p.data(),train.data(),3,2,loss,false);
                });
                for(int k=0;k<aps::PARAMS;++k) same(m.p[k],reference.p[k]);
                for(int e=0;e<2;++e)
                    if(loss[e]!=reference_loss[e]) throw std::runtime_error("Perdas diferentes");
                std::vector<float> scores(test.size()*10,-999);
                unsigned batch=block/32;
                for(unsigned b=0;b<(test.size()+batch-1)/batch;++b) {
                    blockIdx={b,0,0};
                    launch(32,batch,[&] {aps::infer_tiled(m.p.data(),test.data(),17,scores.data());});
                }
                for(size_t i=0;i<scores.size();++i) same(scores[i],expected_scores[i]);
            }
        }
        auto naive=aps::initialize();
        double loss[2]{};
        for(int e=0;e<2;++e)
            for(size_t i=0;i<train.size();++i)
                launch(64,1,[&,i,e] {aps::train_kernel<false>(naive.p.data(),train.data()+i,1,1,loss+e,true);});
        for(int k=0;k<aps::PARAMS;++k) same(naive.p[k],reference.p[k]);
        std::vector<float> scores(test.size()*10,-999);
        blockIdx={0,0,0};
        launch(64,1,[&] {aps::infer_naive(naive.p.data(),test.data(),17,scores.data());});
        for(size_t i=0;i<scores.size();++i) same(scores[i],expected_scores[i]);
        std::cout<<"PASS: logica dos kernels global/shared/naive e blocos parciais 64/128/256 (CPU)\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
