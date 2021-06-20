#include <iostream>
#include "mat.cuh"
__global__ void applyRes(unsigned char *in, xwtype *res, unsigned char *recon)
{
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	recon[i] = (int)in[i] + res[i];
}
/*
__global__ void conv2mid(convtype *conv, midtype *mid, int num)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < num)
	{
		mid[i] = conv[i];
		i += gridDim.x*blockDim.x;
	}
}
__global__ void VectorDiv(midtype *dividend, xwtype *quotient, int divisor, int n)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < n)
	{
		quotient[i] = dividend[i] / divisor;
		i += gridDim.x*blockDim.x;
	}
}
*/
__host__ void findMax(convtype *data, convtype *buffer, int n, convtype *max)
{
	findMax_reduce1 << <MAXGRID, BLOCKSIZE, BLOCKSIZE * sizeof(convtype) >> >(data, buffer, n);
	cudaDeviceSynchronize();
	findMax_reduce2 << <1, MAXGRID / 2, sizeof(convtype)*MAXGRID / 2 >> >(buffer);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(max, buffer, sizeof(convtype), cudaMemcpyDeviceToHost);
}
__global__ void findMax_reduce1(convtype *g_idata, convtype *g_odata, int n)
{
	extern __shared__ convtype sdata[];//BLOCKSIZE>=blockDim.x
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	unsigned int gridSize = blockDim.x * gridDim.x;
	sdata[tid] = (abs(g_idata[i])>abs(g_idata[i + gridSize])) ? abs(g_idata[i]) : abs(g_idata[i + gridSize]);
	i += gridSize * 2;
	while (i < n) { if (sdata[tid] < abs(g_idata[i]))sdata[tid] = abs(g_idata[i]); i += gridSize; }
	__syncthreads();
	if (tid < 512) { if (sdata[tid] < sdata[tid + 512]) sdata[tid] = sdata[tid + 512]; }__syncthreads();
	if (tid < 256) { if (sdata[tid] < sdata[tid + 256]) sdata[tid] = sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { if (sdata[tid] < sdata[tid + 128]) sdata[tid] = sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { if (sdata[tid] < sdata[tid + 64]) sdata[tid] = sdata[tid + 64]; } __syncthreads();
	if (tid < 32)
	{
		if (sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
		if (sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
		if (sdata[tid] < sdata[tid + 8]) sdata[tid] = sdata[tid + 8];
		if (sdata[tid] < sdata[tid + 4]) sdata[tid] = sdata[tid + 4];
		if (sdata[tid] < sdata[tid + 2]) sdata[tid] = sdata[tid + 2];
		if (sdata[tid] < sdata[tid + 1]) sdata[tid] = sdata[tid + 1];
	}
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}
__global__ void findMax_reduce2(convtype *data)//number = 1024
{
	extern __shared__ convtype sdata[];
	unsigned int tid = threadIdx.x;
#if MAXGRID == 2048
	sdata[tid] = data[tid] > data[tid + 1024] ? data[tid] : data[tid + 1024];
#elif MAXGRID == 1024
	sdata[tid] = data[tid] > data[tid + 512] ? data[tid] : data[tid + 512];
#elif MAXGRID == 512
	sdata[tid] = data[tid] > data[tid + 256] ? data[tid] : data[tid + 256];
#elif MAXGRID == 256
	sdata[tid] = data[tid] > data[tid + 128] ? data[tid] : data[tid + 128];
#endif
#if MAXGRID >=2048
	if (tid < 512) { if (sdata[tid] < sdata[tid + 512]) sdata[tid] = sdata[tid + 512]; } __syncthreads();
#endif
#if MAXGRID >=1024
	if (tid < 256) { if (sdata[tid] < sdata[tid + 256]) sdata[tid] = sdata[tid + 256]; } __syncthreads();
#endif
#if MAXGRID >=512
	if (tid < 128) { if (sdata[tid] < sdata[tid + 128]) sdata[tid] = sdata[tid + 128]; } __syncthreads();
#endif
#if MAXGRID >=256
	if (tid < 64) { if (sdata[tid] < sdata[tid + 64]) sdata[tid] = sdata[tid + 64]; } __syncthreads();
#endif
	if (tid < 32)
	{
		if (sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
		if (sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
		if (sdata[tid] < sdata[tid + 8]) sdata[tid] = sdata[tid + 8];
		if (sdata[tid] < sdata[tid + 4]) sdata[tid] = sdata[tid + 4];
		if (sdata[tid] < sdata[tid + 2]) sdata[tid] = sdata[tid + 2];
		if (sdata[tid] < sdata[tid + 1]) sdata[tid] = sdata[tid + 1];
	}
	if (tid == 0)data[0] = sdata[tid];
}
char* HWCN2NCHW_VECT_C_CPU(char *HWCN, int H, int W, int C, int N, int *outSize)
{
	int i, j, k, m;
	int HWC_O, HW, HW4, W4, C_O, c_o, cv_o;
	HW = H*W;
	HW4 = HW * 4;
	W4 = W * 4;
	C_O = ceil((float)C / 4);
	HWC_O = H*W*C_O * 4;
	*outSize = N*C_O*H*W * 4;
	char *NCHW_VECT_C = new char[*outSize];
	memset(NCHW_VECT_C, 0, *outSize);
	for (i = 0;i < N;i++)
		for (j = 0;j < C;j++)
		{
			c_o = j >> 2;
			cv_o = j & 3;
			for (k = 0;k < H;k++)
				for (m = 0;m < W;m++)
					NCHW_VECT_C[i*HWC_O+c_o*HW4+k*W4+m*4+cv_o] = HWCN[k*W*C*N + m*C*N + j*N + i];
		}
	return NCHW_VECT_C;
}
char* NCHW2NCHW_VECT_C_CPU(char *NCHW, int N, int C, int H, int W, int *outSize)
{
	int i, j, k, m;
	int CHW_VECT, CHW, HW, HW4, W4, C_O, c_o, cv_o;
	CHW = C*H*W;
	HW = H*W;
	C_O = ceil((float)C / 4)*4;
	*outSize = N*C_O*H*W;
	CHW_VECT = C_O*H*W;
	HW4 = HW * 4;
	W4 = W * 4;
	char *NCHW_VECT_C = new char[*outSize];
	memset(NCHW_VECT_C, 0, *outSize);
	for (i = 0;i < N;i++)
		for (j = 0;j < C;j++)
		{
			c_o = j >> 2;
			cv_o = j & 3;
			for (k = 0;k < H;k++)
				for (m = 0;m < W;m++)
					NCHW_VECT_C[i*CHW_VECT + c_o*HW4 + k*W4 + m * 4 + cv_o] = NCHW[i*CHW + j*HW + k*W + m];
		}
	return NCHW_VECT_C;
}
char* NCWH2HWCN_CPU(char *NCWH, int H, int W, int C, int N, int *outSize)
{
	int i, j, k, l;
	int WCN, CN, CWH, WH;
	WCN = W*C*N;
	CN = C*N;
	CWH = C*W*H;
	WH = W*H;
	*outSize = N*C*W*H;
	char *HWCN = new char[*outSize];
	for (i = 0;i < N;i++)
		for (j = 0;j < C;j++)
			for (k = 0;k < W;k++)
				for (l = 0;l < H;l++)
					HWCN[l*WCN + k*CN + j*N + i] = NCWH[i*CWH + j*WH + k*H + l];
	return HWCN;
}
xwtype* HWCN2NCHW_CPU(xwtype*HWCN, int H, int W, int C, int N, int *outSize)
{
	int i, j, k, l;
	int WCN, CN, CHW, HW;
	WCN = W*C*N;
	CN = C*N;
	CHW = C*H*W;
	HW = H*W;
	*outSize = N*C*H*W;
	xwtype *NCHW = new xwtype[*outSize];
	for (i = 0;i < H;i++)
		for (j = 0;j < W;j++)
			for (k = 0;k < C;k++)
				for (l = 0;l < N;l++)
					NCHW[l*CHW + k*HW + i*W + j] = HWCN[i*WCN + j*CN + k*N + l];
	return NCHW;
}
xwtype* HWCN2NHWC4_CPU(xwtype*HWCN, int H, int W, int C, int N, int *outSize)
{
	//output numbers and channels must be multiple of 4 in inference, only channels are ensured here though.
	int i, j, k, l;
	int C_O = ceil((float)C / 4) * 4;
	int WCN, CN, HWC_O, WC_O;
	WCN = W*C*N;
	CN = C*N;
	HWC_O = H*W*C_O;
	WC_O = W*C_O;
	*outSize = N*H*W*C_O;
	xwtype *NHWC4 = new xwtype[*outSize];
	memset(NHWC4, 0, *outSize);
	for (i = 0;i < H;i++)
		for (j = 0;j < W;j++)
			for (k = 0;k < C;k++)
				for (l = 0;l < N;l++)
					NHWC4[l*HWC_O + i*WC_O + j*C_O + k] = HWCN[i*WCN + j*CN + k*N + l];
	return NHWC4;
}
__global__ void CHW2CHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int channelSize, int channel, int gridSize)//应明确指定用途，尤其是正负数
{
	int tid;
	int i, addr;
	convtype temp;
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	i = 0;
	addr = tid;
	while (i<channel)
	{
		while (addr < channelSize)
		{
			temp = dividend[addr];
			if (temp < 0)
			{
				temp = (temp - (divisor >> 1)) / divisor;
				if (temp < -128)
					quotient[addr * 4 + (i & 3)] = -128;
				else
					quotient[addr * 4 + (i & 3)] = temp;
			}
			else
			{
				temp = (temp + (divisor >> 1)) / divisor;
				if (temp > 127)
					quotient[addr * 4 + (i & 3)] = 127;
				else
					quotient[addr * 4 + (i & 3)] = temp;
			}
			addr += gridSize;
		}
		i++;
		dividend += channelSize;
		addr = tid;
		if ((i & 3) == 0)
		{
			quotient += channelSize * 4;
		}
	}
}
int NCHW2NCHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int N, int C, int H, int W)
{
	int i, j, c;
	int frameSize = C*H*W;
	int channelSize = H*W;
	for (i = 0;i < N;i++)
	{
		CHW2CHW_VECT_C << <GRIDSIZE, BLOCKSIZE >> >(dividend + i*frameSize, quotient + i*frameSize, divisor, channelSize, C, GRIDSIZE*BLOCKSIZE);
	}
	return 0;
}
__global__ void mul_shift(convtype *input, xwtype *output, int inSize, int multiplier, int shifts)
{
	int tid,bias,gridSize;
	convtype temp;
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	bias = (1 << shifts-1) / multiplier;
	gridSize = gridDim.x*blockDim.x;
	while (tid < inSize)
	{
		temp = input[tid];
		output[tid] = (((int)temp + bias) * multiplier) >> shifts;
		tid += gridSize;
	}
}
__global__ void mul_shift_inplace(convtype *input, int inSize, int multiplier, int shifts)
{
	int tid, bias, gridSize;
	convtype temp;
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	bias = (1 << shifts - 1) / multiplier;
	gridSize = gridDim.x*blockDim.x;
	while (tid < inSize)
	{
		temp = input[tid];
		if(temp>0)
			input[tid] = (((int)temp + bias) * multiplier) >> shifts;
		else
			input[tid] = (((int)temp - bias) * multiplier) >> shifts;
		tid += gridSize;
	}
}
__global__ void CHW2CHW_VECT_C_QUANT_BLU(convtype *input, xwtype *output, int channelSize, int channel, int gridSize, int blu, int multiplier, int shifts)
{
	int tid;
	int i, addr, bias,temp1;
	convtype temp;
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	bias = (1 << shifts-1) / multiplier;
	i = 0;
	addr = tid;
	while (i<channel)
	{
		while (addr < channelSize)
		{
			temp = input[addr];
			/*
			temp = ((temp + bias) * multiplier) >> shifts;
			if(temp>THRESHOLD)
				output[addr * 4 + (i & 3)] = THRESHOLD;
			else if(temp<0)
				output[addr * 4 + (i & 3)] = 0;
			else
				output[addr * 4 + (i & 3)] = temp;
			*/
			//temp1 = (((int)temp + bias) * multiplier) >> shifts;
			if (temp > blu)
			{
				input[addr] = blu;
				output[addr * 4 + (i & 3)] = THRESHOLD;
			}
			else if (temp < 0)
			{
				input[addr] = 0;
				output[addr * 4 + (i & 3)] = 0;
			}
			else
				output[addr * 4 + (i & 3)] = (((int)temp + bias) * multiplier) >> shifts;
			
			addr += gridSize;
		}
		i++;
		input += channelSize;
		addr = tid;
		if ((i & 3) == 0)
		{
			output += channelSize * 4;
		}
	}
}
int NCHW2NCHW_VECT_C_QUANT_BLU(convtype *before, xwtype *after, int N, int C, int H, int W, int blu, int multiplier, int shifts)
{
	int i, j, c;
	int frameSize = C*H*W;
	int channelSize = H*W;
	for (i = 0;i < N;i++)
	{
		CHW2CHW_VECT_C_QUANT_BLU << <GRIDSIZE, BLOCKSIZE >> > (before + i*frameSize, after + i*frameSize, channelSize, C, GRIDSIZE*BLOCKSIZE, blu, multiplier, shifts);
	}
	return 0;
}
double mse(float*x, int size, int gpu, const char*fn, int offset)
{
	float*buffer, *input;
	int i;
	double mse;
	FILE*fp;
	if (fopen_s(&fp, fn, "rb"))
	{
		printf("%s\nopen feature file failed\n", fn);
		exit(1);
	}
	buffer = new float[size];
	fseek(fp, offset, SEEK_SET);
	fread(buffer, sizeof(float), size, fp);
	fclose(fp);
	if (gpu)
	{
		input = new float[size];
		cudaDeviceSynchronize();
		cudaMemcpy(input, x, sizeof(float)*size, cudaMemcpyDeviceToHost);
	}
	else
		input = x;
	mse = 0;
	for (i = 0;i < size;i++)
		if (buffer[i] != input[i])
		{
			printf("diff at %d\n", i);
			mse += fabs(buffer[i] - input[i]);
		}
	mse /= size;
	delete[] buffer;
	if (gpu)
		delete[] input;
	printf("mse=%f\n", mse);
	return mse;
}
int load_tensor(float*x, int size, const char*fn, int offset)
{
	float*x_h;
	FILE*fp;
	int error_code;
	if (fopen_s(&fp, fn, "rb"))
	{
		printf("failed to open %s\n", fn);
		exit(1);
	}
	x_h = new float[size];
	fseek(fp, offset, SEEK_SET);
	fread(x_h, sizeof(float), size, fp);
	fclose(fp);
	error_code = cudaMemcpy(x, x_h, sizeof(float)*size, cudaMemcpyHostToDevice);
	delete[] x_h;
	return error_code;
}
