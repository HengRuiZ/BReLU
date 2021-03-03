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
	C_O = ceil((float)C / 4) * 4;
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
__global__ void CHW2CHW_VECT_C_QUANT(convtype *dividend, xwtype *quotient, int channelSize, int channel, int gridSize, int multiplier, int shifts)//应明确指定用途，尤其是正负数
{
	int tid;
	int i, addr, bias;
	convtype temp;
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	bias = (1 << shifts) / multiplier;
	i = 0;
	addr = tid;
	while (i<channel)
	{
		while (addr < channelSize)
		{
			temp = dividend[addr];
			if(temp>0)
				quotient[addr * 4 + (i & 3)] = (((int)temp + bias) * multiplier) >> shifts;
			else
				quotient[addr * 4 + (i & 3)] = (((int)temp - bias) * multiplier) >> shifts;
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
__global__ void CHW2CHW_VECT_C_QUANT_BLU(convtype *input, xwtype *output, int channelSize, int channel, int gridSize, int blu, int multiplier, int shifts)
{
	int tid;
	int i, addr, bias, temp1;
	convtype temp;
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	bias = (1 << shifts - 1) / multiplier;
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
int NCHW2NCHW_VECT_C_QUANT(convtype *before, xwtype *after, int N, int C, int H, int W, int multiplier, int shifts)
{
	int i, j, c;
	int frameSize = C*H*W;
	int channelSize = H*W;
	for (i = 0;i < N;i++)
	{
		CHW2CHW_VECT_C_QUANT << <GRIDSIZE, BLOCKSIZE >> > (before + i*frameSize, after + i*frameSize, channelSize, C, GRIDSIZE*BLOCKSIZE, multiplier, shifts);
	}
	return 0;
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
