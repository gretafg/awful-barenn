#pragma once
#include <climits>
#include <memory>
#ifndef APS_KERNEL_TEST
#include <cuda_runtime.h>
#endif

namespace aps {
#ifndef APS_KERNEL_TEST
inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
#define CUDA_CHECK(call) ::aps::cuda_check((call), #call)
template<class T> class DeviceBuffer {
    T* ptr_ = nullptr;
public:
    explicit DeviceBuffer(size_t count) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_),count*sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
};
class Event {
    cudaEvent_t event_{};
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    cudaEvent_t get() const { return event_; }
    void record() { CUDA_CHECK(cudaEventRecord(event_)); }
    float elapsed_to(const Event& stop) const {
        CUDA_CHECK(cudaEventSynchronize(stop.event_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms,event_,stop.event_));
        return ms;
    }
};
#endif

// Um unico bloco preserva todas as dependencias do SGD.
// cuda-global usa parametros globais; cuda-shared mantem-nos no SM.
template<bool Shared> __global__
void train_kernel(float* global, const Sample* samples, int count, int epochs,
                  double* losses, bool accumulate) {
    __shared__ float local[Shared ? PARAMS : 1];
    __shared__ Workspace s;
    __shared__ double epoch_loss;
    float* p = global;
    if constexpr (Shared) {
        for (int k=threadIdx.x;k<PARAMS;k+=blockDim.x) local[k]=global[k];
        p=local;
    }
    __syncthreads();
    const int t=threadIdx.x;
    for (int e=0;e<epochs;++e) {
        if(t==0) epoch_loss=0;
        __syncthreads();
        for (int n=0;n<count;++n) {
            const Sample& sample=samples[n];
            if (t<H1) {
                float v=p[B1+t];
                for(int j=0;j<INPUT;++j) v+=sample.x[j]*p[W1+t*INPUT+j];
                s.a[t]=fmaxf(0.0f,v);
            }
            __syncthreads();
            if (t<H2) {
                float v=p[B2+t];
                for(int j=0;j<H1;++j) v+=s.a[j]*p[W2+t*H1+j];
                s.b[t]=fmaxf(0.0f,v);
            }
            __syncthreads();
            if (t<OUTPUT) {
                float v=p[B3+t];
                for(int j=0;j<H2;++j) v+=s.b[j]*p[W3+t*H2+j];
                s.out[t]=v;
                s.d3[t]=v-(sample.label==t?1.0f:0.0f);
            }
            __syncthreads();
            if(t==0) {
                float v=0;
                for(int j=0;j<OUTPUT;++j) v+=0.5f*s.d3[j]*s.d3[j];
                epoch_loss+=v;
            }
            if(t<H2) {
                float v=0;
                for(int j=0;j<OUTPUT;++j) v+=s.d3[j]*p[W3+j*H2+t];
                s.d2[t]=v*(s.b[t]>0?1.0f:0.0f);
            }
            __syncthreads();
            if(t<H1) {
                float v=0;
                for(int j=0;j<H2;++j) v+=s.d2[j]*p[W2+j*H1+t];
                s.d1[t]=v*(s.a[t]>0?1.0f:0.0f);
            }
            // Nenhum peso pode mudar enquanto outro thread calcula o gradiente.
            __syncthreads();
            for(int k=t;k<H2*OUTPUT;k+=blockDim.x)
                p[W3+k]-=RATE*(s.d3[k/H2]*s.b[k%H2]);
            for(int k=t;k<H1*H2;k+=blockDim.x)
                p[W2+k]-=RATE*(s.d2[k/H1]*s.a[k%H1]);
            for(int k=t;k<INPUT*H1;k+=blockDim.x)
                p[W1+k]-=RATE*(s.d1[k/INPUT]*sample.x[k%INPUT]);
            if(t<OUTPUT) p[B3+t]-=RATE*s.d3[t];
            if(t<H2) p[B2+t]-=RATE*s.d2[t];
            if(t<H1) p[B1+t]-=RATE*s.d1[t];
            // A proxima amostra deve enxergar todos os pesos atualizados.
            __syncthreads();
        }
        if(t==0) {
            if(accumulate) losses[e]+=epoch_loss;
            else losses[e]=epoch_loss/count;
        }
        __syncthreads();
    }
    if constexpr(Shared)
        for(int k=t;k<PARAMS;k+=blockDim.x) global[k]=p[k];
}
__global__ void mean_losses(double* loss,int epochs,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<epochs) loss[i]/=count;
}

// Na versao inicial cada thread faz uma imagem completa.
__global__ void infer_naive(const float* p,const Sample* samples,int n,float* scores) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    Workspace s;
    for(int k=0;k<H1;++k) {
        float v=p[B1+k];
        for(int j=0;j<INPUT;++j) v+=samples[i].x[j]*p[W1+k*INPUT+j];
        s.a[k]=fmaxf(0.0f,v);
    }
    for(int k=0;k<H2;++k) {
        float v=p[B2+k];
        for(int j=0;j<H1;++j) v+=s.a[j]*p[W2+k*H1+j];
        s.b[k]=fmaxf(0.0f,v);
    }
    for(int k=0;k<OUTPUT;++k) {
        float v=p[B3+k];
        for(int j=0;j<H2;++j) v+=s.b[j]*p[W3+k*H2+j];
        scores[i*OUTPUT+k]=v;
    }
}
// Cada bloco classifica ate oito imagens e reutiliza os mesmos pesos.
// A dimensao x representa neuronios; a y representa imagens do lote.
__global__ void infer_tiled(const float* global,const Sample* samples,int n,float* scores) {
    __shared__ float p[PARAMS];
    __shared__ float a[8][H1],b[8][H2];
    int flat=threadIdx.y*blockDim.x+threadIdx.x;
    for(int k=flat;k<PARAMS;k+=blockDim.x*blockDim.y) p[k]=global[k];
    __syncthreads();
    int i=blockIdx.x*blockDim.y+threadIdx.y,t=threadIdx.x;
    // Threads fora do lote tambem participam de todas as barreiras.
    if(i<n) {
        float v=p[B1+t];
        for(int j=0;j<INPUT;++j) v+=samples[i].x[j]*p[W1+t*INPUT+j];
        a[threadIdx.y][t]=fmaxf(0.0f,v);
    }
    __syncthreads();
    if(i<n && t<H2) {
        float v=p[B2+t];
        for(int j=0;j<H1;++j) v+=a[threadIdx.y][j]*p[W2+t*H1+j];
        b[threadIdx.y][t]=fmaxf(0.0f,v);
    }
    __syncthreads();
    if(i<n && t<OUTPUT) {
        float v=p[B3+t];
        for(int j=0;j<H2;++j) v+=b[threadIdx.y][j]*p[W3+t*H2+j];
        scores[i*OUTPUT+t]=v;
    }
}

#ifndef APS_KERNEL_TEST
inline void run_cuda(Model& model,const Data& train,const Data& test,int epochs,int repeats,
                     int block,const std::string& backend,std::vector<double>& losses,
                     std::vector<float>& scores,Timing& timing) {
    if(train.size()>INT_MAX || test.size()>INT_MAX) throw std::runtime_error("Dataset excede indices CUDA");
    auto setup=Clock::now();
    CUDA_CHECK(cudaFree(nullptr)); // Inicializacao do contexto fora do treino.
    DeviceBuffer<float> dmodel(PARAMS),dscores(test.size()*OUTPUT);
    DeviceBuffer<Sample> dtrain(train.size()),dtest(test.size());
    DeviceBuffer<double> dloss(epochs);
    Event start,stop;
    start.record();
    CUDA_CHECK(cudaMemcpy(dmodel.get(),model.p.data(),sizeof(Model),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtrain.get(),train.data(),train.size()*sizeof(Sample),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtest.get(),test.data(),test.size()*sizeof(Sample),cudaMemcpyHostToDevice));
    stop.record(); timing.h2d_ms=start.elapsed_to(stop);
    CUDA_CHECK(cudaMemset(dloss.get(),0,epochs*sizeof(double)));
    timing.setup_s=seconds(setup);
    losses.resize(epochs);scores.resize(test.size()*OUTPUT);
    auto begin=Clock::now();start.record();
    if(backend=="cuda-naive") {
        for(int e=0;e<epochs;++e)
            for(size_t i=0;i<train.size();++i)
                train_kernel<false><<<1,block>>>(dmodel.get(),dtrain.get()+i,1,1,dloss.get()+e,true);
        mean_losses<<<(epochs+127)/128,128>>>(dloss.get(),epochs,static_cast<int>(train.size()));
    } else if(backend=="cuda-global") {
        for(int e=0;e<epochs;++e)
            train_kernel<false><<<1,block>>>(dmodel.get(),dtrain.get(),static_cast<int>(train.size()),1,dloss.get()+e,false);
    } else {
        train_kernel<true><<<1,block>>>(dmodel.get(),dtrain.get(),static_cast<int>(train.size()),epochs,dloss.get(),false);
    }
    CUDA_CHECK(cudaGetLastError());
    stop.record();timing.train_device_ms=start.elapsed_to(stop);
    timing.train_s=seconds(begin);
    begin=Clock::now();start.record();
    for(int r=0;r<repeats;++r) {
        if(backend=="cuda-naive")
            infer_naive<<<(test.size()+block-1)/block,block>>>(dmodel.get(),dtest.get(),static_cast<int>(test.size()),dscores.get());
        else
            infer_tiled<<<(test.size()+block/32-1)/(block/32),dim3(32,block/32)>>>(
                dmodel.get(),dtest.get(),static_cast<int>(test.size()),dscores.get());
    }
    CUDA_CHECK(cudaGetLastError());
    stop.record();timing.infer_device_ms=start.elapsed_to(stop);
    timing.infer_s=seconds(begin);
    start.record();
    CUDA_CHECK(cudaMemcpy(model.p.data(),dmodel.get(),sizeof(Model),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(losses.data(),dloss.get(),epochs*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scores.data(),dscores.get(),scores.size()*sizeof(float),cudaMemcpyDeviceToHost));
    stop.record();timing.d2h_ms=start.elapsed_to(stop);
}
#endif
} // namespace aps
