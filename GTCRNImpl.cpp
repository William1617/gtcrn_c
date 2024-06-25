
#include "GTCRNImpl.h"

void GTCRNImpl::ExportWAV(
        const std::string & Filename, 
		const std::vector<float>& Data, 
		unsigned SampleRate) {
    AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(SAMEPLERATE);
	File.save(Filename, AudioFileFormat::Wave);		
}

void GTCRNImpl::Enhance(std::string in_audio,std::string out_audio){
	std::vector<float>  testdata; //vector used to store enhanced data in a wav file
    AudioFile<float> inputfile;
	inputfile.load(in_audio);
	int audiolen=inputfile.getNumSamplesPerChannel();
    int process_num=audiolen/BLOCK_SHIFT;

	for(int i=0;i<process_num;i++){
        memmove(m_pEngine.mic_buffer, m_pEngine.mic_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
      
        for(int n=0;n<BLOCK_SHIFT;n++){
            m_pEngine.mic_buffer[n+BLOCK_LEN-BLOCK_SHIFT]=inputfile.samples[0][n+i*BLOCK_SHIFT];} 
        OnnxInfer();
        for(int j=0;j<BLOCK_SHIFT;j++){
            testdata.push_back(m_pEngine.out_buffer[j]);    //for one forward process save first BLOCK_SHIFT model output samples
        }
    }
    ExportWAV(out_audio,testdata,SAMEPLERATE);
}


void GTCRNImpl::OnnxInfer() {

	float mic_fea[FFT_OUT_SIZE*2] = { 0 };
    float estimated_block[BLOCK_LEN];
    
    double mic_in[BLOCK_LEN];
    std::vector<cpx_type> mic_res(BLOCK_LEN);

	std::vector<size_t> shape;
    shape.push_back(BLOCK_LEN);
    std::vector<size_t> axes;
    axes.push_back(0);
    std::vector<ptrdiff_t> stridel, strideo;
    strideo.push_back(sizeof(cpx_type));
    stridel.push_back(sizeof(double));
 
	for (int i = 0; i < BLOCK_LEN; i++){
        fft_in[i] = m_pEngine.in_buffer[i]*m_windows[i];
	}

	pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, fft_in, fft_res.data(), 1.0);

	for (int i=0;i<FFT_OUT_SIZE;i++){
        mic_fea[2*i]=mic_res[i].real();
        mic_fea[2*i+1]=mic_res[i].imag();
    }

    Ort::Value input_feature = Ort::Value::CreateTensor<float>(
		memory_info, mic_fea, FFT_OUT_SIZE*2, infea_node_dims, 4);
	Ort::Value conv_cache = Ort::Value::CreateTensor<float>(
		memory_info,  m_pEngine.conv_cache,2*16*16*33, conv_cache_dims, 5);

	Ort::Value tra_cache = Ort::Value::CreateTensor<float>(
		memory_info, m_pEngine.tra_cache,2*3*16, tra_cache_dims, 5);
	Ort::Value inter_cache = Ort::Value::CreateTensor<float>(
		memory_info, m_pEngine.inter_cache,2*33*16, inter_cache_dims, 4);

	ort_inputs.clear();
	ort_inputs.emplace_back(std::move(input_feature));
	ort_inputs.emplace_back(std::move(conv_cache));
	ort_inputs.emplace_back(std::move(tra_cache));
	ort_inputs.emplace_back(std::move(inter_cache));
	
    ort_outputs = session->Run(Ort::RunOptions{nullptr},
		input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
		output_node_names.data(), output_node_names.size());
    
	float *out_fea = ort_outputs[0].GetTensorMutableData<float>();
	float *out_concache = ort_outputs[1].GetTensorMutableData<float>();
	std::memcpy(m_pEngine.conv_cache, out_concache, 2*16*16*33 * sizeof(float));
	float *out_tracache = ort_outputs[2].GetTensorMutableData<float>();
	std::memcpy(m_pEngine.tra_cache, out_tracache, 2*3*16 * sizeof(float));

	float *out_intercache = ort_outputs[3].GetTensorMutableData<float>();
	std::memcpy(m_pEngine.inter_cache, out_intercache, 2*33*16 * sizeof(float));


	for (int i = 0; i < FFT_OUT_SIZE; i++) {
        mic_res[i] = cpx_type(output_fea[2*i] , output_fea[2*i+1]);
	}
    pocketfft::c2r(shape, strideo, stridel, axes, pocketfft::BACKWARD, mic_res.data(), mic_in, 1.0);   
    
    for (int i = 0; i < BLOCK_LEN; i++)
        estimated_block[i] = mic_in[i] / BLOCK_LEN;   

	memmove(m_pEngine.out_buffer, m_pEngine.out_buffer + BLOCK_SHIFT, 
        (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
    memset(m_pEngine.out_buffer + (BLOCK_LEN - BLOCK_SHIFT), 
        0, BLOCK_SHIFT * sizeof(float));
    for (int i = 0; i < BLOCK_LEN; i++){
        m_pEngine.out_buffer[i] += estimated_block[i]*m_windows[i];
    }
   
}
 
