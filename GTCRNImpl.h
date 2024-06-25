
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>

#include "onnxruntime_cxx_api.h"
#include "pocketfft_hdronly.h"
#include "AudioFile.h"



#define SAMEPLERATE  (16000)
#define BLOCK_LEN		(512)
#define BLOCK_SHIFT  (256)
#define FFT_OUT_SIZE (257)
typedef complex<double> cpx_type;

struct grctn_engine {
    float mic_buffer[BLOCK_LEN] = { 0 };
    float out_buffer[BLOCK_LEN] = { 0 };
    float conv_cache[2*16*16*33] = { 0 };
    float tra_cache[2*3*16] = { 0 };
    float inter_cache[2*33*16] = { 0 };


};

class GTCRNImpl{
public:

    int Enhance(std::string in_audio,std::string out_audio);
    

private:
    void init_engine_threads(int inter_threads, int intra_threads){
        // The method should be called in each thread/proc in multi-thread/proc work
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

    void init_onnx_model(const std::string ModelPath){
        // Init threads = 1 for 
        init_engine_threads(1, 1);
        // Load model
        session = std::make_shared<Ort::Session>(env, ModelPath.c_str(), session_options);
    };
    void ResetInout(){
        memset(m_pEngine.mic_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.out_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.conv_cache,0,2*16*16*33*sizeof(float));
        memset(m_pEngine.tra_cache,0,2*3*16*sizeof(float));
        memset(m_pEngine.inter_cache,0,2*33*16*sizeof(float));

    };
    void ExportWAV(const std::string & Filename, 
		const std::vector<float>& Data, unsigned SampleRate);
    void OnnxInfer();

    
public:
     GTCRNImpl(const std::string ModelPath){
    init_onnx_model(ModelPath);
    for (int i=0;i<BLOCK_LEN;i++){
        m_windows[i]=sinf(PI*i/(BLOCK_LEN-1));
    }
    ResetInout();
   }

private:
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    grctn_engine m_pEngine;
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char *> input_node_names = {"mix","conv_cache","tra_cache","inter_cache"};

    std::vector<Ort::Value> ort_outputs;
	std::vector<const char *> output_node_names = {"enh","conv_cache_out","tra_cache_out","inter_cache_out"};

	const int64_t infea_node_dims[4] = {1,FFT_OUT_SIZE,1,2}; 
	const int64_t conv_cache_dims[5] = {2,1,16,16,33};
	const int64_t tra_cache_dims[5] = {2,3,1,1,16};
	const int64_t inter_cache_dims[4] = {2,1,33,16};

    float m_windows[BLOCK_LEN]={0};

};
