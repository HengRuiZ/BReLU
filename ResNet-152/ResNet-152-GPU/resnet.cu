#include <stdio.h>
#include "resnet.cuh"

BasicBlock::BasicBlock(void)
{
	return;
}
int BasicBlock::build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int inHeight, int inWidth, int inChannel, int outChannel)
{
	this->stride = outChannel / inChannel;
	this->batch = batch;
	this->inHeight = inHeight;
	this->inWidth = inWidth;
	this->inChannel = inChannel;
	this->outHeight = inHeight / stride;
	this->outWidth = inWidth / stride;
	this->outChannel = outChannel;
	C1.build(cudnnHandle, xDesc, batch, inHeight, inWidth, inChannel, outChannel, ksize, stride);
	C2.build(cudnnHandle, C1.yDesc, batch, C1.outHeight, C1.outWidth, outChannel, outChannel, ksize, 1);
	if (inChannel != outChannel)
	{
		downsample = 1;
		Cd.build(cudnnHandle, xDesc, batch, inHeight, inWidth, inChannel, outChannel, 1, stride);
		this->workspaceSize = Cd.workspaceSize;
	}
	else
	{
		downsample = 0;
		workspaceSize = 0;
	}
	workspaceSize = (workspaceSize > C1.workspaceSize) ? workspaceSize : C1.workspaceSize;
	workspaceSize = (workspaceSize > C2.workspaceSize) ? workspaceSize : C2.workspaceSize;
	return 0;
}
int BasicBlock::load_para(FILE*fp)
{
	C1.load_para(fp);
	C2.load_para(fp);
	if (downsample)
		Cd.load_para(fp);
	return 0;
}
int BasicBlock::quantizeNsave(FILE *f_step, FILE *f_blu, FILE *f_quant)
{
	float ratio[2];
	fread(ratio, sizeof(float), 2, f_blu);
	C1.quantizeNsave(f_step, f_blu, f_quant,ratio[0]);
	C2.quantizeNsave(f_step, f_blu, f_quant, ratio[1]);
	if (downsample)
		Cd.quantizeNsave(f_step, f_blu, f_quant, ratio[0]);
	else
	{
		fread(&mul_d, sizeof(int), 1, f_blu);
		fread(&mul_d, sizeof(int), 1, f_blu);
		fread(&shift_d, sizeof(int), 1, f_blu);
	}
}
int BasicBlock::forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void*x, void*workspace)
{
	C1.ConvForward(cudnnHandle, xDesc, x, workspace, workspaceSize);
	C1.batch_norm(cudnnHandle);
	C1.activate(cudnnHandle);
	//C1.viewmem((xwtype*)x);
	C2.ConvForward(cudnnHandle, C1.yDesc, C1.y, workspace, workspaceSize);
	C2.batch_norm(cudnnHandle);
	//C2.viewmem((xwtype*)C1.y);
	if (downsample)
	{
		Cd.ConvForward(cudnnHandle, xDesc, x, workspace, workspaceSize);
		Cd.batch_norm(cudnnHandle);
		C2.applyRes(cudnnHandle, Cd.yDesc, Cd.y);
	}
	else
		C2.applyRes(cudnnHandle, xDesc, x);
	C2.activate(cudnnHandle);
	//C2.viewmem((xwtype*)C1.y);
	return 0;
}
BasicBlock::~BasicBlock(void)
{
	return;
}
BottleneckBlock::BottleneckBlock(void)
{
	return;
}
int BottleneckBlock::build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int inHeight, int inWidth, int inChannel, int inChannel_expasion, int outChannel)
{
	this->stride = outChannel / inChannel;
	this->batch = batch;
	this->inHeight = inHeight;
	this->inWidth = inWidth;
	this->inChannel = inChannel*inChannel_expasion;
	this->outHeight = inHeight / stride;
	this->outWidth = inWidth / stride;
	this->outChannel = outChannel;
	C1.build(cudnnHandle, xDesc, batch, inHeight, inWidth, this->inChannel, outChannel, 1, 1);//1x1 conv
	C2.build(cudnnHandle, C1.yDesc, batch, C1.outHeight, C1.outWidth, outChannel, outChannel, ksize, stride);//3x3 conv
	C3.build(cudnnHandle, C2.yDesc, batch, C2.outHeight, C2.outWidth, outChannel, outChannel*expansion, 1, 1);//1x1 conv
	if (this->inChannel != outChannel*expansion)
	{
		downsample = 1;
		Cd.build(cudnnHandle, xDesc, batch, inHeight, inWidth, this->inChannel, outChannel*expansion, 1, stride);
		this->workspaceSize = Cd.workspaceSize;
	}
	else
	{
		downsample = 0;
		workspaceSize = 0;
	}
	workspaceSize = (workspaceSize > C1.workspaceSize) ? workspaceSize : C1.workspaceSize;
	workspaceSize = (workspaceSize > C2.workspaceSize) ? workspaceSize : C2.workspaceSize;
	workspaceSize = (workspaceSize > C3.workspaceSize) ? workspaceSize : C3.workspaceSize;
	return 0;
}
int BottleneckBlock::load_para(FILE*fp)
{
	C1.load_para(fp);
	C2.load_para(fp);
	C3.load_para(fp);
	if (downsample)
		Cd.load_para(fp);
	return 0;
}
int BottleneckBlock::forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void*x, void*workspace)
{
	C1.ConvForward(cudnnHandle, xDesc, x, workspace, workspaceSize);
	C1.batch_norm(cudnnHandle);
	C1.activate(cudnnHandle);
	//C1.viewmem((xwtype*)x);
	C2.ConvForward(cudnnHandle, C1.yDesc, C1.y, workspace, workspaceSize);
	C2.batch_norm(cudnnHandle);
	C2.activate(cudnnHandle);
	//C2.viewmem((xwtype*)C1.y);
	C3.ConvForward(cudnnHandle, C2.yDesc, C2.y, workspace, workspaceSize);
	C3.batch_norm(cudnnHandle);
	//C3.viewmem((xwtype*)C2.y);
	if (downsample)
	{
		Cd.ConvForward(cudnnHandle, xDesc, x, workspace, workspaceSize);
		Cd.batch_norm(cudnnHandle);
		C3.applyRes(cudnnHandle, Cd.yDesc, Cd.y);
	}
	else
		C3.applyRes(cudnnHandle, xDesc, x);
	C3.activate(cudnnHandle);
	//C3.viewmem((xwtype*)C1.y);
	return 0;
}
BottleneckBlock::~BottleneckBlock(void)
{
	return;
}
BlockLayer::BlockLayer(void)
{
	return;
}
cudnnTensorDescriptor_t BlockLayer::build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int inHeight, int inWidth, int inChannel, int inChannel_expansion, int outChannel, int block_class, int block_num)
{
	int i;
	this->block_class = block_class;
	this->block_num = block_num;
	this->batch = batch;
	this->stride = outChannel / inChannel;
	this->inHeight = inHeight;
	this->inWidth = inWidth;
	this->inChannel = inChannel;
	this->outHeight = inHeight / stride;
	this->outWidth = inWidth / stride;
	this->outChannel = outChannel;
	workspaceSize = 0;
	if (block_class == 0)
	{
		//build basic blocks
		expansion = 1;
		B_Blocks = new BasicBlock[block_num];
		B_Blocks[0].build(cudnnHandle, xDesc, batch, inHeight, inWidth, inChannel, outChannel);
		workspaceSize = B_Blocks[0].workspaceSize;
		for (i = 1;i < block_num;i++)
		{
			B_Blocks[i].build(cudnnHandle, B_Blocks[i - 1].C2.yDesc, batch, B_Blocks[i - 1].outHeight, B_Blocks[i - 1].outWidth, outChannel, outChannel);
			workspaceSize = workspaceSize > B_Blocks[i].workspaceSize ? workspaceSize : B_Blocks[i].workspaceSize;
		}
		return B_Blocks[block_num-1].C2.yDesc;
	}
	else
	{
		//build bottleneck blocks
		expansion = 4;
		BN_Blocks = new BottleneckBlock[block_num];
		BN_Blocks[0].build(cudnnHandle, xDesc, batch, inHeight, inWidth, inChannel, inChannel_expansion, outChannel);
		workspaceSize = BN_Blocks[0].workspaceSize;
		for (i = 1;i < block_num;i++)
		{
			BN_Blocks[i].build(cudnnHandle, BN_Blocks[i - 1].C3.yDesc, batch, BN_Blocks[i - 1].outHeight, BN_Blocks[i - 1].outWidth, outChannel, 4, outChannel);
			workspaceSize = workspaceSize > BN_Blocks[i].workspaceSize ? workspaceSize : BN_Blocks[i].workspaceSize;
		}
		return BN_Blocks[block_num-1].C3.yDesc;
	}
}
int BlockLayer::load_para(FILE*fp)
{
	int i;
	if (block_class == 0)
		for (i = 0;i < block_num;i++)
			B_Blocks[i].load_para(fp);
	else
		for (i = 0;i < block_num;i++)
			BN_Blocks[i].load_para(fp);
	return 0;
}
int BlockLayer::quantizeNsave(FILE *f_step, FILE *f_blu, FILE *f_quant)
{
	int i;
	if (block_class == 0)
		for (i = 0;i < block_num;i++)
			B_Blocks[i].quantizeNsave(f_step, f_blu, f_quant);
	else
		//TBD
		return 0;
	return 0;
}
void* BlockLayer::forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void*x, void*workspace)
{
	int i;
	if (block_class == 0)
	{
		B_Blocks[0].forward(cudnnHandle, xDesc, x, workspace);
		for (i = 1;i < block_num;i++)
			B_Blocks[i].forward(cudnnHandle, B_Blocks[i - 1].C2.yDesc, B_Blocks[i - 1].C2.y, workspace);
		return B_Blocks[block_num-1].C2.y;
	}
	else
	{
		BN_Blocks[0].forward(cudnnHandle, xDesc, x, workspace);
		for (i = 1;i < block_num;i++)
			BN_Blocks[i].forward(cudnnHandle, BN_Blocks[i - 1].C3.yDesc, BN_Blocks[i - 1].C3.y, workspace);
		return BN_Blocks[block_num-1].C3.y;
	}
}
BlockLayer::~BlockLayer(void)
{
	if (block_class == 0)
		delete[] B_Blocks;
	else
		delete[] BN_Blocks;
	return;
}
Resnet::Resnet(cudnnHandle_t cudnnHandle, int batch, int inHeight, int inWidth, int inChannel, int blockclass, int*blocks)
{
	int i;
	int outChannel;
	if (inHeight != 224 || inWidth != 224 || inChannel != 3)
	{
		printf("invalid input size!");
		exit(1);
	}
	this->batch = batch;
	this->inHeight = inHeight;
	this->inWidth = inWidth;
	this->inChannel = inChannel;
	this->blockclass = blockclass;
	for(i=0;i<4;i++)
		this->block_num[i] = blocks[i];
	I1.build(batch, 3, 224, 224);//224*224
	outChannel = 64;
	C1.build(cudnnHandle, I1.x_outDesc, batch, I1.outHeight, I1.outWidth, I1.outChannel, 64, 7, 2);//112*112
	P1.build(CUDNN_POOLING_MAX, batch, 64, C1.outHeight, C1.outWidth, 3, 2);//56*56
	outDesc[0]=B1.build(cudnnHandle, P1.outDesc, batch, P1.outHeight, P1.outWidth, 64, 1, 64, blockclass, block_num[0]);
	outDesc[1]=B2.build(cudnnHandle, outDesc[0], batch, 56, 56, 64, 4, 128, blockclass, block_num[1]);
	outDesc[2]=B3.build(cudnnHandle, outDesc[1], batch, 28, 28, 128, 4, 256, blockclass, block_num[2]);
	outDesc[3]=B4.build(cudnnHandle, outDesc[2], batch, 14, 14, 256, 4, 512, blockclass, block_num[3]);
	P2.build(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, batch, 512*B4.expansion, 7, 7, 7, 7);
	FC1.build(cudnnHandle, P2.outDesc, batch, 1, 1, 512*B4.expansion, 1000, 1, 1);
	workspaceSize = C1.workspaceSize;
	workspaceSize = workspaceSize > B1.workspaceSize ? workspaceSize : B1.workspaceSize;
	workspaceSize = workspaceSize > B2.workspaceSize ? workspaceSize : B2.workspaceSize;
	workspaceSize = workspaceSize > B3.workspaceSize ? workspaceSize : B3.workspaceSize;
	workspaceSize = workspaceSize > B4.workspaceSize ? workspaceSize : B4.workspaceSize;
	workspaceSize = workspaceSize > FC1.workspaceSize ? workspaceSize : FC1.workspaceSize;
	cudaMalloc(&workspace, workspaceSize);
}
int Resnet::load_para(const char*fn)
{
	FILE *fp;
	if (fopen_s(&fp, fn, "rb"))
	{
		printf("%s\n", fn);
		printf("open image file failed\n");
		exit(1);
	}
	C1.load_para(fp);
	B1.load_para(fp);
	B2.load_para(fp);
	B3.load_para(fp);
	B4.load_para(fp);
	FC1.load_para(fp);
	return 0;
}
int Resnet::load_data(xwtype*input)
{
	cudaDeviceSynchronize();
	I1.load(input);
	return 0;
}
int Resnet::forward(cudnnHandle_t cudnnHandle)
{
	I1.ppro(cudnnHandle);
	C1.ConvForward(cudnnHandle, I1.x_outDesc, I1.x_out, workspace, workspaceSize);
	//mse((float*)C1.u, 2 * 112 * 112 * 64, 1, "output.data", 224 * 224 * 3 * 2 * 4);
	C1.batch_norm(cudnnHandle);
	//C1.viewmem((xwtype*)I1.x_out);
	//mse((float*)C1.y, 2 * 112 * 112 * 64, 1, "output.data", 224 * 224 * 3 * 2 * 4 + 2 * 112 * 112 * 64*4);
	C1.activate(cudnnHandle);
	//mse((float*)C1.y, 2 * 112 * 112 * 64, 1, "output.data", 224 * 224 * 3 * 2 * 4 + 2 * 112 * 112 * 64*2*4);
	//C1.viewmem((xwtype*)I1.x);
	P1.pool(cudnnHandle, C1.yDesc, C1.y);
	//mse((float*)P1.out, 2 * 56 * 56 * 64, 1, "output.data", 224 * 224 * 3 * 2 * 4 + 2 * 112 * 112 * 64*3*4);
	//P1.viewmem();
	out = B1.forward(cudnnHandle, P1.outDesc, P1.out, workspace);
	//mse((float*)B1.B_Blocks[block_num[0] - 1].C2.y, 2 * 56 * 56 * 64, 1, "output.data", 0);
	out = B2.forward(cudnnHandle, outDesc[0], out, workspace);
	//mse((float*)B2.B_Blocks[0].C1.y, 2 * 28 * 28 * 128, 1, "output.data", 0);
	out = B3.forward(cudnnHandle, outDesc[1], out, workspace);
	//mse((float*)B3.B_Blocks[block_num[2] - 1].C2.y, 2 * 14 * 14 * 256, 1, "output.data", 2 * 28 * 28 * 128 * 4);
	out = B4.forward(cudnnHandle, outDesc[2], out, workspace);
	//mse((float*)B4.B_Blocks[block_num[3] - 1].C2.y, 2 * 7 * 7 * 512, 1, "output.data", 2 * 28 * 28 * 128 * 4 + 2 * 14 * 14 * 256);
	P2.pool(cudnnHandle, outDesc[3], out);
	//P2.viewmem();
	FC1.ConvForward(cudnnHandle, P2.outDesc, P2.out, workspace, workspaceSize);
	FC1.applyBias(cudnnHandle);
	//mse((float*)FC1.u, 2000, 1, "fc1.data", 0);
	//FC1.viewmem((xwtype*)P2.out);
	return 0;
}
int Resnet::quantizeNsave(const char*STEP_FILE, const char*BLU_FILE, const char*QUANT_MODEL)
{
	FILE *f_step, *f_blu, *f_quant;
	if (fopen_s(&f_step, STEP_FILE, "rb"))
	{
		printf("%s\n", STEP_FILE);
		printf("open step file failed\n");
		exit(1);
	}
	if (fopen_s(&f_blu, BLU_FILE, "rb"))
	{
		printf("%s\n", BLU_FILE);
		printf("open step file failed\n");
		exit(1);
	}
	if (fopen_s(&f_quant, QUANT_MODEL, "rb"))
	{
		printf("%s\n", QUANT_MODEL);
		printf("open step file failed\n");
		exit(1);
	}
	B1.quantizeNsave(f_step, f_blu, f_quant);
	fclose(f_step);
	fclose(f_blu);
	fclose(f_quant);
	return 0;
}
Resnet::~Resnet(void)
{
	cudaFree(workspace);
	return;
}
/*
qvrcnn::qvrcnn(int gpu_num, int batch, int channel, int height, int width)//build qvrcnn
{
	check(cudaSetDevice(gpu_num));
	check(cudnnCreate(&cudnnHandle));
	this->batch = batch, this->channel = channel, this->height = height, this->width = width;
	
	I1.build(batch, channel, height, width);
	C1.build(cudnnHandle, I1.xDesc, batch, height, width, I1.outChannel, 64, 5);
	C2_1.build(cudnnHandle, C1.vDesc, batch, height, width, C1.outChannel, 32, 3);
	C2_2.build(cudnnHandle, C1.vDesc, batch, height, width, C1.outChannel, 16, 5);
	Conc1.build(batch, height, width, C2_1.outChannel, C2_2.outChannel);
	C3_1.build(cudnnHandle, Conc1.concDesc, batch, height, width, Conc1.outChannel, 16, 3);
	C3_2.build(cudnnHandle, Conc1.concDesc, batch, height, width, Conc1.outChannel, 32, 1);
	Conc2.build(batch, height, width, C3_1.outChannel, C3_2.outChannel);
	C4.build(cudnnHandle, Conc2.concDesc, batch, height, width, Conc2.outChannel, 1, 3);

	workspaceSize = MAXGRID * sizeof(convtype);
	workspaceSize = (workspaceSize > C1.workspaceSize) ? workspaceSize : C1.workspaceSize;
	workspaceSize = (workspaceSize > C2_1.workspaceSize) ? workspaceSize : C2_1.workspaceSize;
	workspaceSize = (workspaceSize > C2_2.workspaceSize) ? workspaceSize : C2_2.workspaceSize;
	workspaceSize = (workspaceSize > C3_1.workspaceSize) ? workspaceSize : C3_1.workspaceSize;
	workspaceSize = (workspaceSize > C3_2.workspaceSize) ? workspaceSize : C3_2.workspaceSize;
	workspaceSize = (workspaceSize > C4.workspaceSize) ? workspaceSize : C4.workspaceSize;
	cudaDeviceSynchronize();
	check(cudaMalloc(&workspace, workspaceSize));
}
int qvrcnn::load_para(char *filename)
{
	FILE *fp;
	if (fopen_s(&fp, filename, "rb"))
	{
		printf("cannot open model file.\n");
		exit(1);
	}
	C1.load_para(fp);
	C2_1.load_para(fp);
	C2_2.load_para(fp);
	C3_1.load_para(fp);
	C3_2.load_para(fp);
	C4.load_para(fp);
	fclose(fp);
	return 0;
}
int qvrcnn::load_static_para(char *filename)
{
	FILE *fp;
	if (fopen_s(&fp, filename, "rb"))
	{
		printf("cannot open model file.\n");
		exit(1);
	}
	C1.load_static_para(fp);
	C2_1.load_static_para(fp);
	C2_2.load_static_para(fp);
	C3_1.load_static_para(fp);
	C3_2.load_static_para(fp);
	C4.load_static_para(fp);
	fclose(fp);
	return 0;
}
int qvrcnn::load_data(datatype *input)
{
	I1.load(input);
	return 0;
}
#if defined(INT8x4_EXT_CONFIG)||defined(INT8_EXT_CONFIG)
int save_steps(int *max_u, const char*filename)
{
	FILE*fp;
	if (fopen_s(&fp, filename, "ab"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	fwrite(max_u, sizeof(int), 1, fp);
	fclose(fp);
	return 0;
}
int qvrcnn::forward(void)
{
	int layer=0;

	//input layer
	I1.ppro();

	//layer 1
	layer = 1;
	adjustBasic<<<1,C1.outChannel>>>(steps, (btype*)C1.b, (btype*)C1.b_adj, layer-1);
	C1.ConvForward(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, workspaceSize);
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	//C1.activate(cudnnHandle);
	cudaDeviceSynchronize();
	//C1.quantize_out(workspace);
	//C1.quantize_out_fix(2689);//ori=53788
	C1.quantize_out_static();
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	insert_w(C1.step_w, layer);
	insert_y(C1.step_y, layer);

	//layer 2
	layer = 2;
	adjustBasic <<<1, C2_1.outChannel >>>(steps, (btype*)C2_1.b, (btype*)C2_1.b_adj, layer-1);
	adjustBasic <<<1, C2_2.outChannel >>>(steps, (btype*)C2_2.b, (btype*)C2_2.b_adj, layer - 1);
	C2_1.ConvForward(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_1.viewmem((xwtype*)C1.v);
#endif
	C2_1.activate(cudnnHandle);
	C2_2.ConvForward(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_2.viewmem((xwtype*)C1.v);
#endif
	C2_2.activate(cudnnHandle);
	//Conc1.concat(&C2_1, &C2_2, workspace);
	Conc1.concat_blu(&C2_1, &C2_2);
	insert_w(C2_1.step_w, layer);
	insert_y(C2_1.step_y, layer);

	//layer 3
	layer = 3;
	adjustBasic <<<1, C3_1.outChannel>>>(steps, (btype*)C3_1.b, (btype*)C3_1.b_adj, layer-1);
	adjustBasic <<<1, C3_2.outChannel>>>(steps, (btype*)C3_2.b, (btype*)C3_2.b_adj, layer-1);
	C3_1.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_1.viewmem((xwtype*)Conc1.conc);
#endif
	C3_1.activate(cudnnHandle);
	C3_2.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_2.viewmem((xwtype*)Conc1.conc);
#endif
	C3_2.activate(cudnnHandle);
	//Conc2.concat(&C3_1, &C3_2, workspace);
	Conc2.concat_blu(&C3_1, &C3_2);
	insert_w(C3_1.step_w, layer);
	insert_y(C3_1.step_y, layer);

	//layer 4
	layer = 4;
	adjustBasic <<<1, C4.outChannel>>>(steps, (btype*)C4.b, (btype*)C4.b_adj, layer-1);
	C4.ConvForward(cudnnHandle, Conc2.concDesc, Conc2.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C4.viewmem((xwtype*)Conc2.conc);
#endif
	insert_w(C4.step_w, layer);
	
	//restore
	cudaDeviceSynchronize();
	//adjustOutput<<<GRIDSIZE, BLOCKSIZE>>>(steps, (convtype*)C4.u, (xwtype*)C4.v, layer,C4.uSize,GRIDSIZE*BLOCKSIZE);//scale n times
	adjustOutput_static<<<GRIDSIZE, BLOCKSIZE>>>((convtype*)C4.u, (xwtype*)C4.v, 141, 16, C4.uSize, GRIDSIZE*BLOCKSIZE);
#ifdef MEM_DBG
	C4.viewmem((xwtype*)C4.v);
#endif
	cudaDeviceSynchronize();
	I1.applyRes((xwtype*)C4.v);
	save_steps(&C1.max_u, "max_u_C1.data");
	//save_steps(&C3_2.max_u, "max_u_C3_2.data");
	//save_b_adj("b_adj.data");
	return 0;
}
int qvrcnn::forward_blu(void)
{
	int layer = 0;

	//input layer
	I1.ppro();

	//layer 1
	//layer = 1;
	//adjustBasic <<<1, C1.outChannel >>>(steps, (btype*)C1.b, (btype*)C1.b_adj, layer - 1);
	C1.ConvForward_static(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, workspaceSize);
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	cudaDeviceSynchronize();
	C1.quantize_out_blu();
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	//insert_w(C1.step_w, layer);
	//insert_y(C1.step_y, layer);

	//layer 2
	layer = 2;
	//adjustBasic << <1, C2_1.outChannel >> >(steps, (btype*)C2_1.b, (btype*)C2_1.b_adj, layer - 1);
	//adjustBasic << <1, C2_2.outChannel >> >(steps, (btype*)C2_2.b, (btype*)C2_2.b_adj, layer - 1);
	C2_1.ConvForward_static(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_1.viewmem((xwtype*)C1.v);
#endif
	C2_2.ConvForward_static(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_2.viewmem((xwtype*)C1.v);
#endif
	Conc1.concat_blu(&C2_1, &C2_2);
	//insert_w(C2_1.step_w, layer);
	//insert_y(C2_1.step_y, layer);

	//layer 3
	layer = 3;
	//adjustBasic << <1, C3_1.outChannel >> >(steps, (btype*)C3_1.b, (btype*)C3_1.b_adj, layer - 1);
	//adjustBasic << <1, C3_2.outChannel >> >(steps, (btype*)C3_2.b, (btype*)C3_2.b_adj, layer - 1);
	C3_1.ConvForward_static(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_1.viewmem((xwtype*)Conc1.conc);
#endif
	C3_2.ConvForward_static(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_2.viewmem((xwtype*)Conc1.conc);
#endif
	Conc2.concat_blu(&C3_1, &C3_2);
	//insert_w(C3_1.step_w, layer);
	//insert_y(C3_1.step_y, layer);

	//layer 4
	layer = 4;
	//adjustBasic << <1, C4.outChannel >> >(steps, (btype*)C4.b, (btype*)C4.b_adj, layer - 1);
	C4.ConvForward_static(cudnnHandle, Conc2.concDesc, Conc2.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C4.viewmem((xwtype*)Conc2.conc);
#endif
#ifdef MEM_DBG
	//restore
	C4.quantize_out_static();
	C4.viewmem((xwtype*)Conc2.conc);
#endif
	I1.applyRes_y((convtype*)C4.u, C4.mul, C4.shift);
#ifdef MEM_DBG
	I1.viewmem((xwtype*)C4.v);
#endif
	//save_steps(&C1.max_u, "max_u_C1.data");
	//save_steps(&C3_2.max_u, "max_u_C3_2.data");
	//save_b_adj("b_adj.data");
	return 0;
}
#elif defined(FLOAT_CONFIG)
int qvrcnn::forward(void)
{
	int layer = 0;

	//input layer
	I1.ppro();

	//layer 1
	layer = 1;
	C1.ConvForward(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, workspaceSize);
	//C1.viewmem((xwtype*)I1.x_ppro);
	C1.activate(cudnnHandle);

	//layer 2
	layer = 2;
	C2_1.ConvForward(cudnnHandle, C1.uDesc, C1.u, workspace, workspaceSize);
	//C2_1.viewmem((xwtype*)C1.u);
	C2_1.activate(cudnnHandle);
	C2_2.ConvForward(cudnnHandle, C1.uDesc, C1.u, workspace, workspaceSize);
	//C2_2.viewmem((xwtype*)C1.u);
	C2_2.activate(cudnnHandle);
	Conc1.concat(&C2_1, &C2_2, workspace);

	//layer 3
	layer = 3;
	C3_1.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
	//C3_1.viewmem((xwtype*)Conc1.conc);
	C3_1.activate(cudnnHandle);
	C3_2.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
	//C3_2.viewmem((xwtype*)Conc1.conc);
	C3_2.activate(cudnnHandle);
	Conc2.concat(&C3_1, &C3_2, workspace);

	//layer 4
	layer = 4;
	C4.ConvForward(cudnnHandle, Conc2.concDesc, Conc2.conc, workspace, workspaceSize);
	//C4.viewmem((xwtype*)Conc2.conc);																		  //C4.viewmem();
	cudaDeviceSynchronize();
	I1.applyRes((xwtype*)C4.u);
	//I1.viewmem((xwtype*)C4.u);
	return 0;
}
#endif

int qvrcnn::save_b_adj(const char*filename)
{
	FILE*fp;
	if (fopen_s(&fp, filename, "a+"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	fwrite(C1.b_adj, sizeof(btype), C1.outChannel, fp);
	fwrite(C2_1.b_adj, sizeof(btype), C2_1.outChannel, fp);
	fwrite(C2_2.b_adj, sizeof(btype), C2_2.outChannel, fp);
	fwrite(C3_1.b_adj, sizeof(btype), C3_1.outChannel, fp);
	fwrite(C3_2.b_adj, sizeof(btype), C3_2.outChannel, fp);
	fwrite(C4.b_adj, sizeof(btype), C4.outChannel, fp);
	fclose(fp);
	return 0;
}
void qvrcnn::insert_w(int stepw, int layer)//layer means current layer
{
	int i, j;
	for (i = 0;i < layer - 1;i++)
		if (steps.stepw[i] < stepw)
		{
			for (j = layer - 1;j > i;j--)steps.stepw[j] = steps.stepw[j - 1];
			steps.stepw[j] = stepw;
			return;
		}
	steps.stepw[i] = stepw;
	return;
}
void qvrcnn::insert_y(int stepy, int layer)//layer means current layer
{
	int i, j;
	for (i = 0;i < layer - 1;i++)
		if (steps.stepy[i] > stepy)
		{
			for (j = layer - 1;j > i;j--)steps.stepy[j] = steps.stepy[j - 1];
			steps.stepy[j] = stepy;
			return;
		}
	steps.stepy[i] = stepy;
	return;
}
qvrcnn::~qvrcnn()
{
	cudnnDestroy(cudnnHandle);
	check(cudaFree(workspace));
}
*/
__global__ void adjustOutput_static(convtype*o, xwtype *o_adj, int multiplier, int shifts, int num, int gridSize)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int temp;
	int bias = 1 << shifts - 1;
	while (tid < num)
	{
		temp = o[tid];
		o_adj[tid] = temp*multiplier + bias >> shifts;
		tid += gridSize;
	}
}
int layer_HWCN2NCHW_VECT_C(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	char *w_in = new char[wSize];
	char *w_out;
	int *b = new int[outChannel];
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(w_in, sizeof(char), wSize, fp_in);
	w_out = HWCN2NCHW_VECT_C_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(char), wSize_out, fp_out);
	fread(b, sizeof(int), outChannel, fp_in);
	fwrite(b, sizeof(int), outChannel, fp_out);
	free(w_in);
	free(w_out);
	free(b);
	return 0;
}
int model_HWCN2NCHW_VECT_C(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 1, 64);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 64, 32);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 64, 16);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 16);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 1, 48, 32);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
int layer_HWCN2NCHW(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	xwtype *w_in = new xwtype[wSize];
	xwtype *w_out;
	btype *b = new btype[outChannel];
#if defined(INT8x4_EXT_CONFIG)||defined(INT8_EXT_CONFIG)
	fread(b, sizeof(btype), 1, fp_in);
	fwrite(b, sizeof(btype), 1, fp_out);
#endif
	fread(w_in, sizeof(xwtype), wSize, fp_in);
	w_out = HWCN2NCHW_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(xwtype), wSize_out, fp_out);
	fread(b, sizeof(btype), outChannel, fp_in);
	fwrite(b, sizeof(btype), outChannel, fp_out);
	free(w_in);
	free(w_out);
	free(b);
	return 0;
}
int model_HWCN2NCHW(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_HWCN2NCHW(fp_in, fp_out, 5, 1, 64);
	layer_HWCN2NCHW(fp_in, fp_out, 3, 64, 32);
	layer_HWCN2NCHW(fp_in, fp_out, 5, 64, 16);
	layer_HWCN2NCHW(fp_in, fp_out, 3, 48, 16);
	layer_HWCN2NCHW(fp_in, fp_out, 1, 48, 32);
	layer_HWCN2NCHW(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
int layer_NCWH2HWCN(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	char *w_in = new char[wSize];
	char *w_out;
	int *b = new int[outChannel];
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(w_in, sizeof(char), wSize, fp_in);
	w_out = NCWH2HWCN_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(char), wSize_out, fp_out);
	fread(b, sizeof(int), outChannel, fp_in);
	fwrite(b, sizeof(int), outChannel, fp_out);
	return 0;
}
int model_NCWH2HWCN(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_NCWH2HWCN(fp_in, fp_out, 5, 1, 64);
	layer_NCWH2HWCN(fp_in, fp_out, 3, 64, 32);
	layer_NCWH2HWCN(fp_in, fp_out, 5, 64, 16);
	layer_NCWH2HWCN(fp_in, fp_out, 3, 48, 16);
	layer_NCWH2HWCN(fp_in, fp_out, 1, 48, 32);
	layer_NCWH2HWCN(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
int layer_qfp_HWCN2NCHW_VECT_C(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	char *w_in = new char[wSize];
	char *w_out;
	int *b = new int[outChannel];

	fread(w_in, sizeof(char), wSize, fp_in);
	fread(b, sizeof(int), outChannel, fp_in);
	
	w_out = HWCN2NCHW_VECT_C_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(char), wSize_out, fp_out);
	fwrite(b, sizeof(int), outChannel, fp_out);
	
	//blu_q,mul,shift
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	return 0;
}
int model_qfp_HWCN2NCHW_VECT_C(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 1, 64);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 64, 32);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 64, 16);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 16);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 1, 48, 32);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
