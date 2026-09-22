// Experimental wrappers around the installed PyTorch sampling routines.
// binomial_inversion_cached is mechanically derived at compile time from the
// installed header, with its probability logarithm supplied as an argument.
// All paths use the same PyTorch Philox engine; no RNG optimization is tested.
struct Uniform {
  at::philox_engine engine;
  __device__ Uniform(unsigned long long seed, unsigned long long id):engine(seed,id,0) {}
  __device__ float operator()() {
    // Match the CUDA PyTorch binomial wrapper's 24-bit [0,1) conversion.
    return float(engine() & 0x00ffffffu) * (1.0f/16777216.0f);
  }
};

template<int MODE>
__device__ void sample_one(const float* counts, const float* p,
                          const float* logs, const int* ids, float* out,
                          int size,int boots,unsigned long long seed) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=size) return;
  int row=i/boots;
  float n=counts[i],prob=p[row];
  Uniform uniform(seed,i);
  BaseSampler<float,Uniform> rng(uniform);
  if constexpr (MODE==3) {
    out[i]=sample_binomial<float,float,Uniform>(n,prob,rng);
    return;
  }
  if (n<=0 || prob<=0) {out[i]=0;return;}
  if (prob>=1) {out[i]=n;return;}
  float q=prob<=0.5f ? prob : 1.0f-prob;
  float draw;
  if(n*q>=10.0f) {
    draw=btrs<float,float,Uniform>(n,q,rng);
  } else {
    float logprob=0;
    if constexpr(MODE==0) logprob=compat_log1p(-q);
    if constexpr(MODE==1) logprob=logs[row];
    if constexpr(MODE==2) logprob=logs[ids[row]];
    draw=binomial_inversion_cached<float,float,Uniform>(n,q,rng,logprob);
  }
  out[i]=prob<=0.5f ? draw : n-draw;
}

#define SAMPLER(NAME,MODE) \
extern "C" __global__ void NAME(const float* counts,const float* p,const float* logs, \
 const int* ids,float* out,int size,int boots,unsigned long long seed) { \
 sample_one<MODE>(counts,p,logs,ids,out,size,boots,seed); }
SAMPLER(direct,0)
SAMPLER(state_cache,1)
SAMPLER(shared_cache,2)
SAMPLER(header_reference,3)

extern "C" __global__ void make_logs(const float* p,float* out,int size) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<size) {
    float q=p[i]<=0.5f ? p[i] : 1.0f-p[i];
    out[i]=compat_log1p(-q);
  }
}
extern "C" __global__ void rng_words(unsigned int* out,int size,unsigned long long seed) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<size) {
    at::philox_engine engine(seed,i,0);
    for(int j=0;j<12;++j) out[i*12+j]=engine();
  }
}
