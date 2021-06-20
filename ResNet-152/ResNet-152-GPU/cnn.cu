#include "cnn.cuh"

CovLayer::CovLayer()
{
	// initialize descriptors
	cudnnCreateFilterDescriptor(&wDesc);
	cudnnCreateTensorDescriptor(&uDesc);
	cudnnCreateTensorDescriptor(&yDesc);
	cudnnCreateTensorDescriptor(&MeanVarScaleBiasDesc);
	cudnnCreateConvolutionDescriptor(&convDesc);
	cudnnCreateActivationDescriptor(&actiDesc);
}
int CovLayer::build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int height, int width, int inChannel, int outChannel, int ksize, int stride)
{
	this->batch = batch;
	this->inHeight = height;
	this->inWidth = width;
	if (XWFORMAT == CUDNN_TENSOR_NCHW_VECT_C)
	{
		this->inChannel = ceil((float)inChannel / 4) * 4;
		this->outChannel = ceil((float)outChannel / 4) * 4;
	}
	else
	{
		this->inChannel = inChannel;
		this->outChannel = outChannel;
	}
	this->ksize = ksize;
	this->stride = stride;
	this->outHeight = height / stride;
	this->outWidth = width / stride;
	this->wSize = outChannel*inChannel*ksize*ksize;
	this->uySize = batch*outChannel*outHeight*outWidth;
	check(cudnnSetFilter4dDescriptor(wDesc, XWTYPE, XWFORMAT, this->outChannel, this->inChannel, ksize, ksize));
	check(cudnnSetTensor4dDescriptor(uDesc, YFORMAT, YTYPE, batch, this->outChannel, this->outHeight, this->outWidth));
	check(cudnnSetTensor4dDescriptor(yDesc, XWFORMAT, XWTYPE, batch, this->outChannel, this->outHeight, this->outWidth));
	check(cudnnSetTensor4dDescriptor(MeanVarScaleBiasDesc,BFORMAT,BTYPE,1, outChannel, 1, 1));
	check(cudnnSetConvolution2dDescriptor(convDesc,
		(ksize - 1) / 2, (ksize - 1) / 2,//padding
		stride, stride,//stride
		1, 1,//dilation
		CUDNN_CROSS_CORRELATION,
		CONVTYPE));
	int ArgoCount;
	//check(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle, xDesc, wDesc, convDesc, uDesc, 8, &ArgoCount, AlgoPerf));
	//check(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, xDesc, wDesc, convDesc, uDesc, ConvPerf, 0, &AlgoPerf));
	check(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,uDesc,ALGO,&workspaceSize));
	check(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
	//build layer in GPU
	cudaDeviceSynchronize();
	check(cudaMalloc(&w, sizeof(xwtype)*this->wSize));
	check(cudaMalloc(&u, sizeof(convtype)*this->uySize));
	check(cudaMalloc(&y, sizeof(xwtype)*this->uySize));
	check(cudaMalloc(&mean, sizeof(float)*outChannel));
	check(cudaMalloc(&var, sizeof(float)*outChannel));
	check(cudaMalloc(&scale, sizeof(float)*outChannel));
	check(cudaMalloc(&bias, sizeof(float)*outChannel));
	builded = 1;
	return 0;
}

#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
int CovLayer::load_para(FILE *fp)//copy paras from file to GPU memory
{
	xwtype *w_h;
	btype *b_h;
	int *b_int, i;
	w_h = (xwtype*)malloc(sizeof(xwtype)*this->wSize);
	b_h = (btype*)malloc(sizeof(btype)*this->outChannel);
	b_int = (int*)malloc(sizeof(int)*this->outChannel);
	//convert format if necessary
	fread(w_h, sizeof(xwtype), this->wSize, fp);
	fread(b_int, sizeof(int), this->outChannel, fp);
	fread(&this->blu, sizeof(int), 1, fp);
	fread(&this->mul, sizeof(int), 1, fp);
	fread(&this->shift, sizeof(int), 1, fp);
	for (i = 0;i < this->outChannel;i++)b_h[i] = b_int[i];
	check(cudaMemcpyAsync(w, w_h, sizeof(xwtype) * this->wSize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(mean, b_h, sizeof(btype) * this->outChannel, cudaMemcpyHostToDevice));
	//memdbg((convtype*)u, (xwtype*)v, (btype*)b, uSize);
	free(w_h);
	free(b_h);
	free(b_int);
	return 0;
}
#elif defined(FLOAT_CONFIG)
int CovLayer::load_para(FILE *fp)
{
	xwtype *w_h;
	float *MeanVarScaleBias_h;
	w_h = (xwtype*)malloc(sizeof(xwtype)*this->wSize);
	MeanVarScaleBias_h = (float*)malloc(sizeof(float)*this->outChannel*4);
	fread(w_h, sizeof(xwtype), this->wSize, fp);
	fread(MeanVarScaleBias_h, sizeof(float), this->outChannel*4, fp);
	check(cudaMemcpyAsync(w, w_h, sizeof(xwtype) * this->wSize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(mean, MeanVarScaleBias_h, sizeof(float) * outChannel, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(var, MeanVarScaleBias_h+outChannel, sizeof(float) * outChannel, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(scale, MeanVarScaleBias_h+outChannel*2, sizeof(float) * outChannel, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(bias, MeanVarScaleBias_h+outChannel*3, sizeof(float) * outChannel, cudaMemcpyHostToDevice));
	free(w_h);
	free(MeanVarScaleBias_h);
	return 0;
}
#endif
int CovLayer::quantizeNsave(FILE *f_step, FILE *f_blu, FILE *f_quant, float ratio, int fc)
{
	// only float to uchar
	int i, j, sub_ksize;
	float*wf = new float[wSize];
	char*wq = new char[wSize];
	char*wq_VECT;
	int wSize_VECT;
	float*bias = new float[outChannel];
	float*mean = new float[outChannel];
	int *bq = new int[outChannel];
	float*step = new float[outChannel];
	float stepm;
	float temp;
	//read float param
	check(cudaMemcpyAsync(wf, w, sizeof(float) * this->wSize, cudaMemcpyDeviceToHost));
	check(cudaMemcpyAsync(bias, this->bias, sizeof(float) * outChannel, cudaMemcpyDeviceToHost));
	check(cudaMemcpyAsync(mean, this->mean, sizeof(float) * outChannel, cudaMemcpyDeviceToHost));
	//read quantize param
	fread(step, sizeof(float), outChannel, f_step);
	fread(&stepm, sizeof(float), 1, f_step);
	fread(&this->blu, sizeof(int), 1, f_blu);
	fread(&this->mul, sizeof(int), 1, f_blu);
	fread(&this->shift, sizeof(int), 1, f_blu);
	//quantize
	sub_ksize = inChannel*ksize*ksize;
	for (i = 0;i < outChannel;i++)
	{
		for (j = 0;j < sub_ksize; j++)
		{
			temp = wf[i*sub_ksize + j] / step[i];
			if (temp > 127)
				wq[i*sub_ksize + j] = 127;
			else if (temp > 0)
				wq[i*sub_ksize + j] = temp + 0.5;
			else if (temp > -128)
				wq[i*sub_ksize + j] = temp - 0.5;
			else
				wq[i*sub_ksize + j] = -128;
		}
		if(fc)
			temp = mean[i] / stepm * ratio;
		else
			temp = (bias[i] / stepm - mean[i] / step[i]) * ratio;
		if (temp > 0)
			bq[i] = temp + 0.5;
		else
			bq[i] = temp - 0.5;
	}
	wq_VECT = NCHW2NCHW_VECT_C_CPU(wq, outChannel, inChannel, ksize, ksize, &wSize_VECT);
	fwrite(wq_VECT, sizeof(char), wSize_VECT, f_quant);
	fwrite(bq, sizeof(int), outChannel, f_quant);
	fwrite(&blu, sizeof(int), 1, f_quant);
	fwrite(&mul, sizeof(int), 1, f_quant);
	fwrite(&shift, sizeof(int), 1, f_quant);
	return 0;
}
int CovLayer::ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize)//¾í»ý
{
	//cudaDeviceSynchronize();
	check(cudnnConvolutionForward(cudnnHandle, &alpha, xDesc,
		x, wDesc, w, convDesc,
		ALGO, workspace, workspaceSize, &beta,
		uDesc, u));//convolution
	//convdbg((xwtype*)x, (xwtype*)w, (convtype*)u, (btype*)b);
	return 0;
}
int CovLayer::applyBias(cudnnHandle_t cudnnHandle)
{
	//cudaDeviceSynchronize();
	// use mean as bias
	check(cudnnAddTensor(cudnnHandle, &alpha, MeanVarScaleBiasDesc, mean, &alpha, uDesc, u));
	return 0;
}
int CovLayer::batch_norm(cudnnHandle_t cudnnHandle)
{
	//cudaDeviceSynchronize();
	check(cudnnBatchNormalizationForwardInference(cudnnHandle,bnMode,&alpha,&beta,uDesc,u,yDesc,y,MeanVarScaleBiasDesc,scale,bias,mean,var,epsilon));
	return 0;
}
#if defined INT8x4_EXT_CONFIG
int CovLayer::applyRes(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t resDesc, void*res)
{
	//cudaDeviceSynchronize();
	check(cudnnAddTensor(cudnnHandle, &alpha, resDesc, res, &alpha, uDesc, u));
	return 0;
}
#elif defined FLOAT_CONFIG
int CovLayer::applyRes(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t resDesc, void*res)
{
	cudaDeviceSynchronize();
	check(cudnnAddTensor(cudnnHandle, &alpha, resDesc, res, &alpha, yDesc, y));
	return 0;
}
#endif

int CovLayer::activate(cudnnHandle_t cudnnHandle)
{
	//cudaDeviceSynchronize();
	check(cudnnActivationForward(cudnnHandle, actiDesc, &alpha, uDesc, u, &beta, uDesc, u));
	return 0;
}

int CovLayer::quantize_out(void)//u->y,blu,mul,shift
{
	//cudaDeviceSynchronize();
	NCHW2NCHW_VECT_C_QUANT_BLU((convtype*)u, (xwtype*)y, batch, outChannel, outHeight, outWidth, blu, mul, shift);
	return 0;
}
int CovLayer::quantize_u(int mul, int shift)
{
	mul_shift_inplace << <BLOCKSIZE, GRIDSIZE >> > ((convtype*)u, uySize, mul, shift);
	return 0;
}
#if defined INT8x4_EXT_CONFIG
int CovLayer::viewmem(xwtype*x)
{
	int i, j, k;
	xwtype*x_h, *w_h, *y_h;
	convtype*u_h;
	x_h = new xwtype[inHeight*inWidth*inChannel*batch];
	w_h = new xwtype[wSize];
	u_h = new convtype[uySize];
	y_h = new xwtype[uySize];
	float mean_h[16], var_h[16], scale_h[16], bias_h[16];
	cudaMemcpy(x_h, x, sizeof(xwtype)*inHeight*inWidth*inChannel*batch, cudaMemcpyDeviceToHost);
	cudaMemcpy(w_h, w, sizeof(xwtype)*wSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_h, u, sizeof(convtype)*uySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y, sizeof(xwtype)*uySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(mean_h, mean, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(var_h, var, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(scale_h, scale, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(bias_h, bias, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	printf("x:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 8;j++)
			printf("%d	", x_h[i*inWidth*4 + j]);
		printf("\n");
	}
	printf("weights:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 8;j++)
			printf("%d	", w_h[i*ksize * 4 + j]);
		printf("\n");
	}
	printf("\nbatch norm:\nmean	var	scale	bias\n%f	%f	%f	%f\n", mean_h[0], var_h[0], scale_h[0], bias_h[0]);
	printf("u:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%f	", u_h[i*outWidth + j]);
		printf("\n");
	}
	//printf("mul:%d,shift:%d\n",mul,shift);
	printf("y:\n");
	for (k = 0;k < 16;k++)
	{
		printf("%d\n", k * 4);
		for (i = 0;i < 5;i++)
		{
			for (j = 0;j < 8;j++)
				printf("%d	", y_h[k*outWidth*outHeight*4+i*outWidth * 4 + j]);
			printf("\n");
		}
	}
	/*
	int k, temp = 0;
	for (i = 0;i < 4;i++)//channel
		for (j = 0;j < 2;j++)//height
			for (k = 0;k < 2;k++)//width
				temp += x_h[j * 4 * inWidth + k * 4 + i] * w_h[(j + 1)*ksize * 4 + (k + 1) * 4 + i];

	//*/
	delete[] x_h;
	delete[] w_h;
	delete[] u_h;
	delete[] y_h;
	return 0;
}
#elif defined FLOAT_CONFIG
int CovLayer::viewmem(xwtype*x)
{
	int i,j;
	xwtype*x_h, *w_h, *y_h;
	convtype*u_h;
	x_h = new xwtype[inHeight*inWidth*inChannel*batch];
	w_h = new xwtype[wSize];
	u_h = new convtype[uySize];
	y_h = new xwtype[uySize];
	float mean_h[16],var_h[16],scale_h[16],bias_h[16];
	cudaMemcpy(x_h, x, sizeof(xwtype)*inHeight*inWidth*inChannel*batch, cudaMemcpyDeviceToHost);
	cudaMemcpy(w_h, w, sizeof(xwtype)*wSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_h, u, sizeof(convtype)*uySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y, sizeof(xwtype)*uySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(mean_h, mean, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(var_h, var, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(scale_h, scale, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(bias_h, bias, sizeof(float) * 16, cudaMemcpyDeviceToHost);
	printf("x:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%d	", x_h[i*inWidth + j]);
		printf("\n");
	}
	printf("weights:\n");
	for (i = 0;i < 5;i++)
		printf("%d	", w_h[i]);
	printf("\nbatch norm:\nmean	var	scale	bias\n%f	%f	%f	%f\n", mean_h[0],var_h[0],scale_h[0],bias_h[0]);
	printf("u:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%f	", u_h[i*outWidth + j]);
		printf("\n");
	}
	//printf("mul:%d,shift:%d\n",mul,shift);
	printf("y:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%d	", y_h[(i*outWidth + j)]);
		printf("\n");
	}
	delete[] x_h;
	delete[] w_h;
	delete[] u_h;
	delete[] y_h;
	return 0;
}
#endif

CovLayer::~CovLayer()
{
	if (builded)
	{
		cudnnDestroyFilterDescriptor(wDesc);
		cudnnDestroyTensorDescriptor(uDesc);
		cudnnDestroyTensorDescriptor(yDesc);
		cudnnDestroyTensorDescriptor(MeanVarScaleBiasDesc);
		cudnnDestroyConvolutionDescriptor(convDesc);
		cudnnDestroyActivationDescriptor(actiDesc);
		cudaFree(w);
		cudaFree(u);
		cudaFree(y);
		cudaFree(mean);
		cudaFree(var);
		cudaFree(scale);
		cudaFree(bias);
	}
}
InputLayer::InputLayer(void)
{
	cudnnCreateTensorDescriptor(&xDesc);
	cudnnCreateTensorDescriptor(&x_outDesc);
}
int InputLayer::build(int batch, int channel, int height, int width)
{
	if (XWTYPE == CUDNN_DATA_INT8x4)
		if (channel % 4)
		{
			printf("inChannel is not multiple of 4\n");
			exit(1);
		}
	this->batch = batch;
	this->inHeight = height;
	this->inWidth = width;
	this->inChannel = channel;
	this->inSize = batch*height*width*channel;
	this->outHeight = height;
	this->outWidth = width;
	this->outChannel = channel;
	this->outSize = inSize;
	check(cudnnSetTensor4dDescriptor(xDesc, XWFORMAT, XWTYPE, batch, inChannel, height, width));
	check(cudnnSetTensor4dDescriptor(x_outDesc, XWFORMAT, XWTYPE, batch, outChannel, height, width));
	cudaDeviceSynchronize();
	check(cudaMalloc(&x, sizeof(XWTYPE)*inSize));
	check(cudaMalloc(&x_out, sizeof(XWTYPE)*outSize));
	//check(cudaMemset(x_ppro,0, sizeof(xwtype)*outSize));
	return 0;
}
int InputLayer::load(xwtype *input)
{
	cudaMemcpy(x, input, sizeof(xwtype)*inSize, cudaMemcpyHostToDevice);
	return 0;
}
int InputLayer::ppro(cudnnHandle_t cudnnHandle)
{
	cudaDeviceSynchronize();
	cudaMemcpy(x_out, x, sizeof(xwtype)*inSize, cudaMemcpyDeviceToDevice);
	return 0;
}

int InputLayer::viewmem(xwtype*res)
{
	xwtype*x_h = new xwtype[inSize];
	xwtype*x_out_h = new xwtype[inSize];
	cudaMemcpy(x_h, x, sizeof(xwtype)*inSize, cudaMemcpyDeviceToHost);
	delete[] x_h;
	delete[] x_out_h;
	return 0;
}
InputLayer::~InputLayer(void)
{
	cudnnDestroyTensorDescriptor(xDesc);
	cudaFree(x);
	cudaFree(x_out);
}
PoolLayer::PoolLayer(void)
{
	cudnnCreateTensorDescriptor(&uDesc);
	cudnnCreateTensorDescriptor(&yDesc);
	cudnnCreatePoolingDescriptor(&poolingDesc);
}
int PoolLayer::build(cudnnPoolingMode_t mode, int batch, int channel, int inHeight, int inWidth, int ksize, int stride)
{
	this->mode = mode;
	this->batch = batch;
	this->channel = channel;
	this->ksize = ksize;
	this->stride = stride;
	this->inHeight = inHeight;
	this->inWidth = inWidth;
	this->outHeight = inHeight / stride;
	this->outWidth = inWidth / stride;
	this->outSize = batch*channel*outHeight*outWidth;
	if (mode == CUDNN_POOLING_MAX)
		padding = (ksize - 1) / 2;
	else
		padding = 0;
	cudaDeviceSynchronize();
	check(cudnnSetTensor4dDescriptor(uDesc, YFORMAT, YTYPE, batch, channel, outHeight, outWidth));
	check(cudnnSetTensor4dDescriptor(yDesc, XWFORMAT, XWTYPE, batch, channel, outHeight, outWidth));
	check(cudnnSetPooling2dDescriptor(
		poolingDesc, mode, CUDNN_PROPAGATE_NAN,
		ksize, ksize,
		padding, padding,
		stride, stride));
	check(cudaMalloc(&u, sizeof(YTYPE)*outSize));
	check(cudaMalloc(&y, sizeof(XWTYPE)*outSize));
	return 0;
}
int PoolLayer::pool(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t inDesc, void*in)
{
	cudaDeviceSynchronize();
	check(cudnnPoolingForward(cudnnHandle,poolingDesc,&alpha,inDesc,in,&beta,uDesc,u));
	return 0;
}
int PoolLayer::quantizeNsave(FILE *blu_fp, FILE *quant)
{
	fread(&blu, sizeof(int), 1, blu_fp);
	fread(&mul, sizeof(int), 1, blu_fp);
	fread(&shift, sizeof(int), 1, blu_fp);
	fwrite(&blu, sizeof(int), 1, quant);
	fwrite(&mul, sizeof(int), 1, quant);
	fwrite(&shift, sizeof(int), 1, quant);
	return 0;
}
int PoolLayer::load_blu(FILE*fp)
{
	fread(&blu, sizeof(int), 1, fp);
	fread(&mul, sizeof(int), 1, fp);
	fread(&shift, sizeof(int), 1, fp);
	return 0;
}
int PoolLayer::quantize_out(void)
{
	cudaDeviceSynchronize();
	NCHW2NCHW_VECT_C_QUANT_BLU((convtype*)u, (xwtype*)y, batch, channel, outHeight, outWidth, blu, mul, shift);
	return 0;
}
int PoolLayer::quantize_u(int mul, int shift)
{
	mul_shift_inplace << <BLOCKSIZE, GRIDSIZE >> > ((convtype*)u, outSize, mul, shift);
	return 0;
}
int PoolLayer::viewmem(void)
{
	int i, j;
	convtype*u_h = new convtype[outSize];
	xwtype*y_h = new xwtype[outSize];
	cudaMemcpy(u_h, u, sizeof(convtype)*outSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y, sizeof(xwtype)*outSize, cudaMemcpyDeviceToHost);
	printf("pooling:\nu:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%f	", u_h[i*outWidth + j]);
		printf("\n");
	}
	printf("y:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 8;j++)
			printf("%d	", y_h[i*outWidth*4 + j]);
		printf("\n");
	}
	delete[] u_h;
	delete[] y_h;
	return 0;
}
PoolLayer::~PoolLayer(void)
{
	cudnnDestroyPoolingDescriptor(poolingDesc);
	cudnnDestroyTensorDescriptor(uDesc);
	cudnnDestroyTensorDescriptor(yDesc);
	cudaFree(u);
	cudaFree(y);
}

