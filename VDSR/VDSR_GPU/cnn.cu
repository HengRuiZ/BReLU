#include "cnn.cuh"

CovLayer::CovLayer()
{
	// initialize descriptors
	cudnnCreateFilterDescriptor(&wDesc);
	cudnnCreateTensorDescriptor(&uDesc);
	cudnnCreateTensorDescriptor(&bDesc);
	cudnnCreateTensorDescriptor(&yDesc);
	cudnnCreateConvolutionDescriptor(&convDesc);
	cudnnCreateActivationDescriptor(&actiDesc);
}
int CovLayer::build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int height, int width, int inChannel, int outChannel, int ksize, int stride)
{
	this->batch = batch;
	this->inHeight = height;
	this->inWidth = width;
	this->inChannel = inChannel;
	if (XWTYPE == CUDNN_DATA_INT8x4)
	{
		if (inChannel % 4)
		{
			printf("invalid inChannel.\n");
			exit(1);
		}
		this->outChannel = ceil(outChannel / 4.0) * 4;
	}
	else
		this->outChannel = outChannel;
	this->ksize = ksize;
	this->stride = stride;
	this->outHeight = height / stride;
	this->outWidth = width / stride;
	this->wSize = outChannel*inChannel*ksize*ksize;
	this->uySize = batch*outChannel*outHeight*outWidth;
	check(cudnnSetFilter4dDescriptor(wDesc, XWTYPE, XWFORMAT, outChannel, inChannel, ksize, ksize));
	check(cudnnSetTensor4dDescriptor(uDesc, YFORMAT, YTYPE, batch, outChannel, this->outHeight, this->outWidth));
	check(cudnnSetTensor4dDescriptor(bDesc, YFORMAT, YTYPE, 1, outChannel, 1, 1));
	check(cudnnSetTensor4dDescriptor(yDesc, XWFORMAT, XWTYPE, batch, this->outChannel, this->outHeight, this->outWidth));
	check(cudnnSetConvolution2dDescriptor(convDesc,
		(ksize - 1) / 2, (ksize - 1) / 2,//padding
		stride, stride,//stride
		1, 1,//dilation
		CUDNN_CROSS_CORRELATION,
		CONVTYPE));
	check(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xDesc, wDesc, convDesc, uDesc, ALGO, &workspaceSize));
	check(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
	//build layer in GPU
	cudaDeviceSynchronize();
	check(cudaMalloc(&w, sizeof(xwtype)*this->wSize));
	check(cudaMalloc(&u, sizeof(convtype)*this->uySize));
	check(cudaMalloc(&b, sizeof(convtype)*this->outChannel));
	check(cudaMalloc(&y, sizeof(xwtype)*this->uySize));
	builded = 1;
	return 0;
}

#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
int CovLayer::load_para(FILE *fp)//copy paras from file to GPU memory
{
	xwtype *w_h;
	convtype *b_h;
	w_h = (xwtype*)malloc(sizeof(xwtype)*this->wSize);
	b_h = (convtype*)malloc(sizeof(convtype)*this->outChannel);
	fread(w_h, sizeof(xwtype), this->wSize, fp);
	fread(b_h, sizeof(convtype), this->outChannel, fp);
	fread(&this->blu, sizeof(int), 1, fp);
	fread(&this->mul, sizeof(int), 1, fp);
	fread(&this->shift, sizeof(int), 1, fp);
	check(cudaMemcpyAsync(w, w_h, sizeof(xwtype) * this->wSize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(b, b_h, sizeof(convtype) * this->outChannel, cudaMemcpyHostToDevice));
	free(w_h);
	free(b_h);
	return 0;
}
#elif defined(FLOAT_CONFIG)
int CovLayer::load_para(FILE *fp)
{
	xwtype *w_h;
	convtype *b_h;
	w_h = (xwtype*)malloc(sizeof(xwtype)*this->wSize);
	b_h = (convtype*)malloc(sizeof(convtype)*this->outChannel);
	fread(w_h, sizeof(xwtype), this->wSize, fp);
	fread(b_h, sizeof(convtype), this->outChannel, fp);
	check(cudaMemcpyAsync(w, w_h, sizeof(xwtype) * this->wSize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(b, b_h, sizeof(convtype) * this->outChannel, cudaMemcpyHostToDevice));
	free(w_h);
	free(b_h);
	return 0;
}
int CovLayer::quantizeNsave(FILE *f_step, FILE *f_blu, FILE *f_quant)
{
	// only float to uchar
	int i;
	float*wf = new float[wSize];
	char*wq = new char[wSize];
	char*wq_VECT;
	int wSize_VECT;
	float *bf = new float[outChannel];
	float *bq = new float[outChannel];
	float step;
	float ratio_conv;
	float temp;
	//read float param
	check(cudaMemcpyAsync(wf, w, sizeof(float) * this->wSize, cudaMemcpyDeviceToHost));
	check(cudaMemcpyAsync(bf, b, sizeof(float) * this->outChannel, cudaMemcpyDeviceToHost));
	//read quantize param
	fread(&step, sizeof(float), 1, f_step);
	fread(&temp, sizeof(float), 1, f_blu);//blu
	fread(&ratio_conv, sizeof(float), 1, f_blu);//ratio_out
	fread(&this->blu, sizeof(int), 1, f_blu);
	fread(&this->mul, sizeof(int), 1, f_blu);
	fread(&this->shift, sizeof(int), 1, f_blu);
	//quantize
	for (i = 0;i < wSize;i++)
	{
		temp = wf[i] / step;
		if (temp > 127)
			wq[i] = 127;
		else if (temp > 0)
			wq[i] = temp + 0.5;
		else if (temp > -128)
			wq[i] = temp - 0.5;
		else
			wq[i] = -128;
	}
	for (i = 0;i < outChannel;i++)
	{
		temp = bf[i] * ratio_conv;
		if (temp > 0)
			bq[i] = (int)(temp + 0.5);
		else
			bq[i] = (int)(temp - 0.5);
	}
	wq_VECT = NCHW2NCHW_VECT_C_CPU(wq, outChannel, inChannel, ksize, ksize, &wSize_VECT);
	fwrite(wq_VECT, sizeof(char), wSize_VECT, f_quant);
	fwrite(bq, sizeof(float), outChannel, f_quant);
	fwrite(&blu, sizeof(int), 1, f_quant);
	fwrite(&mul, sizeof(int), 1, f_quant);
	fwrite(&shift, sizeof(int), 1, f_quant);
	delete[] wf;
	delete[] wq;
	delete[] wq_VECT;
	delete[] bf;
	delete[] bq;
	return 0;
}
#endif
int CovLayer::ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace)//¾í»ý
{
	//cudaDeviceSynchronize();
	check(cudnnConvolutionForward(cudnnHandle, &alpha, xDesc,
		x, wDesc, w, convDesc,
		ALGO, workspace, workspaceSize, &beta,
		uDesc, u));//convolution
	check(cudnnAddTensor(cudnnHandle, &alpha, bDesc, b, &alpha, uDesc, u));//apply bias
	return 0;
}
#if defined INT8x4_EXT_CONFIG
int CovLayer::activate(cudnnHandle_t cudnnHandle)
{
	//cudaDeviceSynchronize();
	check(cudnnActivationForward(cudnnHandle, actiDesc, &alpha, uDesc, u, &beta, uDesc, u));
	return 0;
}
#elif defined FLOAT_CONFIG
int CovLayer::activate(cudnnHandle_t cudnnHandle)
{
	//cudaDeviceSynchronize();
	check(cudnnActivationForward(cudnnHandle, actiDesc, &alpha, uDesc, u, &beta, yDesc, y));
	return 0;
}
#endif
#if defined INT8x4_EXT_CONFIG
int CovLayer::quantize_out(void)//u->y,blu,mul,shift
{
	//cudaDeviceSynchronize();
	NCHW2NCHW_VECT_C_QUANT_BLU((convtype*)u, (xwtype*)y, batch, outChannel, outHeight, outWidth, blu, mul, shift);
	return 0;
}
int CovLayer::viewmem(xwtype*x)
{
	int i, j;
	xwtype*x_h, *w_h, *y_h;
	convtype*u_h;
	x_h = new xwtype[inHeight*inWidth*inChannel*batch];
	w_h = new xwtype[wSize];
	u_h = new convtype[uySize];
	y_h = new xwtype[uySize];
	cudaMemcpy(x_h, x, sizeof(xwtype)*inHeight*inWidth*inChannel*batch, cudaMemcpyDeviceToHost);
	cudaMemcpy(w_h, w, sizeof(xwtype)*wSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_h, u, sizeof(convtype)*uySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y, sizeof(xwtype)*uySize, cudaMemcpyDeviceToHost);
	printf("x:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 8;j++)
			printf("%d	", x_h[i*inWidth * 4 + j]);
		printf("\n");
	}
	printf("weights:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 8;j++)
			printf("%d	", w_h[i*ksize * 4 + j]);
		printf("\n");
	}
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
		for (j = 0;j < 8;j++)
			printf("%d	", y_h[i*outWidth * 4 + j]);
		printf("\n");
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
	int i, j;
	xwtype*x_h, *w_h, *y_h;
	convtype*u_h;
	x_h = new xwtype[inHeight*inWidth*inChannel*batch];
	w_h = new xwtype[wSize];
	u_h = new convtype[uySize];
	y_h = new xwtype[uySize];
	cudaMemcpy(x_h, x, sizeof(xwtype)*inHeight*inWidth*inChannel*batch, cudaMemcpyDeviceToHost);
	cudaMemcpy(w_h, w, sizeof(xwtype)*wSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_h, u, sizeof(convtype)*uySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y, sizeof(xwtype)*uySize, cudaMemcpyDeviceToHost);
	printf("x:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%f	", x_h[i*inWidth + j]);
		printf("\n");
	}
	printf("weights:\n");
	for (i = 0;i < 5;i++)
		printf("%f	", w_h[i]);
	printf("\nu:\n");
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
			printf("%f	", y_h[(i*outWidth + j)]);
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
		cudnnDestroyTensorDescriptor(bDesc);
		cudnnDestroyTensorDescriptor(yDesc);
		cudnnDestroyConvolutionDescriptor(convDesc);
		cudnnDestroyActivationDescriptor(actiDesc);
		cudaFree(w);
		cudaFree(u);
		cudaFree(b);
		cudaFree(y);
	}
}

InputLayer::InputLayer(void)
{
	cudnnCreateTensorDescriptor(&xDesc);
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
	this->height = height;
	this->width = width;
	this->channel = channel;
	this->size = batch*height*width*channel;
	check(cudnnSetTensor4dDescriptor(xDesc, XWFORMAT, XWTYPE, batch, channel, height, width));
	cudaDeviceSynchronize();
	check(cudaMalloc(&x, sizeof(XWTYPE)*size));
	//check(cudaMemset(x_ppro,0, sizeof(xwtype)*outSize));
	return 0;
}
int InputLayer::load(xwtype *input)
{
	cudaMemcpy(x, input, sizeof(xwtype)*size, cudaMemcpyHostToDevice);
	return 0;
}
int InputLayer::viewmem(xwtype*res)
{
	xwtype*x_h = new xwtype[size];
	cudaMemcpy(x_h, x, sizeof(xwtype)*size, cudaMemcpyDeviceToHost);
	delete[] x_h;
	return 0;
}
InputLayer::~InputLayer(void)
{
	cudnnDestroyTensorDescriptor(xDesc);
	cudaFree(x);
}
