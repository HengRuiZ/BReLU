#include <stdio.h>
#include <stdlib.h>
#include "cnn.cuh"

class VDSR {
public:
	VDSR(cudnnHandle_t cudnnHandle, int batch, int inHeight, int inWidth, int inChannel, int layers);
	int load_para(const char*fn);
	int load_data(xwtype*input);
	int forward(cudnnHandle_t cudnnHandle);
	int quantizeNsave(const char*STEP_FILE, const char*BLU_FILE, const char*QUANT_MODEL);
	~VDSR(void);

	int batch;
	int inHeight, inWidth, inChannel;
	int layers;
	InputLayer I1;//input layer
	CovLayer C_in;//first conv layer
	CovLayer*C_layers;
	CovLayer C_out;
	float ratio_out;
	int workspaceSize;
	void*workspace;
};