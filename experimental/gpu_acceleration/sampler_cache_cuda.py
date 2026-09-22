"""Standalone NVRTC binomial experiment, using installed PyTorch header routines.

No package backend integration. NVRTC and the CUDA driver are loaded from the
existing environment; no compiler/toolkit installation is required. All variants
use the same Philox engine, stream assignment, and sampling arithmetic.
"""
import ctypes as C
import hashlib
from pathlib import Path
import torch


class CudaSampler:
    def __init__(self):
        root=Path(torch.__file__).parent
        header=(root/'include/ATen/native/Distributions.h').read_text()
        philox=(root/'include/ATen/core/PhiloxRNGEngine.h').read_text()
        self.source_hashes={'Distributions.h':hashlib.sha256(header.encode()).hexdigest(),
                            'PhiloxRNGEngine.h':hashlib.sha256(philox.encode()).hexdigest()}
        # Reuse the installed implementation, removing host-only dependencies.
        engine=philox[philox.index('namespace at {'):]
        lo=engine.index('  inline float randn(')
        hi=engine.index('  /**',lo)
        engine=engine[:lo]+engine[hi:]
        engine=engine.replace('std::array','DeviceArray')
        sampler=header[header.index('template<typename scalar_t, typename sampler_t>'):]
        sampler=sampler[:sampler.index('// The function `sample_gamma`')]
        routines=header[header.index('/* the functions stirling_approx_tail'):]
        routines=routines[:routines.index('\n/*',routines.index('C10_DEVICE scalar_t sample_binomial'))]
        inv=routines[routines.index('template<typename scalar_t, typename accscalar_t, typename uniform_sampler_t>'):]
        inv=inv[:inv.index('\ntemplate',1)]
        cached=inv.replace('binomial_inversion(', 'binomial_inversion_cached(').replace(
            '>& standard_uniform) {','>& standard_uniform, accscalar_t logprob) {').replace(
            '  accscalar_t logprob = compat_log1p(-prob);','')
        assert cached!=inv and 'accscalar_t logprob = ' not in cached
        preamble='''
#define C10_DEVICE __device__
#define C10_HOST_DEVICE __device__
#define NAN __int_as_float(0x7fc00000)
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef int int32_t;
template<class T, int N> struct DeviceArray {
  T values[N];
  __device__ T& operator[](int i) {return values[i];}
  __device__ const T& operator[](int i) const {return values[i];}
};
'''
        for name,fn in [('log','log'),('log1p','log1p'),('sqrt','sqrt'),
                        ('ceil','ceil'),('floor','floor'),('abs','fabs')]:
            preamble+=f'__device__ float compat_{name}(float x) {{return {fn}f(x);}}\n'
            preamble+=f'__device__ double compat_{name}(double x) {{return {fn}(x);}}\n'
        source=preamble+engine+sampler+routines+cached+Path(__file__).with_suffix('.cu').read_text()
        self.source_hashes['assembled_source']=hashlib.sha256(source.encode()).hexdigest()
        nvrtc_path=root.parent/'nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
        self.nvrtc=C.CDLL(str(nvrtc_path))
        self.driver=C.CDLL('libcuda.so.1')
        nv=self.nvrtc
        self._bind(nv,'nvrtcCreateProgram',[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p])
        self._bind(nv,'nvrtcCompileProgram',[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)])
        for name in ('nvrtcGetProgramLogSize','nvrtcGetPTXSize'):
            self._bind(nv,name,[C.c_void_p,C.POINTER(C.c_size_t)])
        for name in ('nvrtcGetProgramLog','nvrtcGetPTX'):
            self._bind(nv,name,[C.c_void_p,C.c_void_p])
        self._bind(nv,'nvrtcDestroyProgram',[C.POINTER(C.c_void_p)])
        d=self.driver
        self._bind(d,'cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self._bind(d,'cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self._bind(d,'cuLaunchKernel',[C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.c_void_p,C.c_void_p])
        self._bind(d,'cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p])
        torch.cuda.init()
        # Establish the primary context on this host thread before driver calls.
        torch.empty(1,device='cuda')
        major,minor=torch.cuda.get_device_capability()
        # No fast math. Match ordinary CUDA compilation's default FMA behavior.
        self.options=[f'--gpu-architecture=compute_{major}{minor}','--std=c++17']
        opts=(C.c_char_p*len(self.options))(*(x.encode() for x in self.options))
        program=C.c_void_p()
        self._check(nv.nvrtcCreateProgram(C.byref(program),source.encode(),b'sampler_cache.cu',0,None,None))
        try:
            status=nv.nvrtcCompileProgram(program,len(opts),opts)
            size=C.c_size_t()
            self._check(nv.nvrtcGetProgramLogSize(program,C.byref(size)))
            log=C.create_string_buffer(size.value)
            self._check(nv.nvrtcGetProgramLog(program,log))
            self.compile_log=log.value.decode()
            if status: raise RuntimeError(f'NVRTC error {status}: {self.compile_log}')
            self._check(nv.nvrtcGetPTXSize(program,C.byref(size)))
            ptx=C.create_string_buffer(size.value)
            self._check(nv.nvrtcGetPTX(program,ptx))
        finally:
            nv.nvrtcDestroyProgram(C.byref(program))
        self.module=C.c_void_p()
        self._check(d.cuModuleLoadData(C.byref(self.module),ptx))
        self.kernels={}
        self.registers={}
        for name in ('direct','state_cache','shared_cache','header_reference','make_logs','rng_words'):
            fn=C.c_void_p()
            self._check(d.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
            self.kernels[name]=fn
            regs=C.c_int()
            self._check(d.cuFuncGetAttribute(C.byref(regs),4,fn))
            self.registers[name]=regs.value

    @staticmethod
    def _bind(lib,name,args):
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=C.c_int

    @staticmethod
    def _check(status):
        if status: raise RuntimeError(f'CUDA/NVRTC status {status}')

    def launch(self,name,size,args):
        params=(C.c_void_p*len(args))(*(C.cast(C.byref(x),C.c_void_p) for x in args))
        stream=torch.cuda.current_stream().cuda_stream
        self._check(self.driver.cuLaunchKernel(self.kernels[name],(size+255)//256,1,1,256,1,1,0,
                                               C.c_void_p(stream),params,None))

    def logs(self,p,out):
        self.launch('make_logs',p.numel(),[C.c_void_p(p.data_ptr()),C.c_void_p(out.data_ptr()),C.c_int(p.numel())])

    def sample(self,mode,counts,p,logs,ids,out,seed):
        assert counts.is_contiguous() and counts.dtype==torch.float32
        assert counts.ndim==2 and len(p)==counts.shape[0]
        self.launch(mode,counts.numel(),[
            C.c_void_p(t.data_ptr()) for t in (counts,p,logs,ids,out)]+[
            C.c_int(counts.numel()),C.c_int(counts.shape[1]),C.c_ulonglong(seed)])

    def rng(self,out,seed):
        self.launch('rng_words',out.shape[0],[C.c_void_p(out.data_ptr()),C.c_int(out.shape[0]),C.c_ulonglong(seed)])
