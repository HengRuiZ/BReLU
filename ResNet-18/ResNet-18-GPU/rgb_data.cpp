#include "rgb_data.h"

rgb_data::rgb_data(int batch_size, int height, int width, int channel)
{
	this->batch_size = batch_size;
	this->h = height;
	this->w = width;
	this->channel = channel;
	this->inSize = batch_size*height*width*channel;
	this->image = new datatype[inSize];
	this->ppro = new xwtype[inSize];
	this->label = new labeltype[batch_size];
	this->pred = new convtype[batch_size*NUM_CLASSES];
	this->iter = 0;
	//this->top1_num = 0;
	//this->top5_num = 0;
}
int rgb_data::next_batch(const char *imgfile, const char *labelfile)
{
	long long offset;
	FILE *imgfp, *labelfp;
	if (fopen_s(&imgfp, imgfile, "rb"))
	{
		printf("%s\n", imgfile);
		printf("open image file failed\n");
		exit(1);
	}
	if (fopen_s(&labelfp, labelfile, "rb"))
	{
		printf("%s\n", labelfile);
		printf("open label file failed\n");
		fclose(imgfp);
		exit(1);
	}
	offset = iter;
	offset *= batch_size*w*h*channel * sizeof(datatype);
	_fseeki64(imgfp, offset, SEEK_SET);
	//fseek(imgfp, offset, SEEK_SET);
	fread(this->image, sizeof(datatype), inSize, imgfp);
	fseek(labelfp, iter*batch_size*sizeof(labeltype), SEEK_SET);
	fread(this->label, sizeof(labeltype), batch_size, labelfp);
	iter++;
	fclose(imgfp);
	fclose(labelfp);
	return 0;
}
int rgb_data::read_frame(const char *imgfile, const char *labelfile, int n)
{
	FILE *imgfp, *labelfp;
	if (fopen_s(&imgfp, imgfile, "rb"))
	{
		printf("%s\n", imgfile);
		printf("open image file failed\n");
		exit(1);
	}
	if (fopen_s(&labelfp, labelfile, "rb"))
	{
		printf("%s\n", labelfile);
		printf("open label file failed\n");
		fclose(imgfp);
		exit(1);
	}
	fseek(imgfp, n*w*h*channel * sizeof(datatype), SEEK_SET);
	fread(this->image, sizeof(datatype), w*h*channel, imgfp);
	fseek(labelfp, n * sizeof(labeltype), SEEK_SET);
	fread(this->label, sizeof(labeltype), 1, labelfp);
	fclose(imgfp);
	fclose(labelfp);
	return 0;
}
int rgb_data::preprocess(void)
{
	// normalize mean[3] and std[3]
	float norm1, norm2;
	float norm_param[2][3] = { {0.485,0.456,0.406},{ 0.229, 0.224, 0.225 } };
	int i,j,k;
	/*
	FILE*fp = fopen("input.data", "rb");
	unsigned char *input = new unsigned char[224 * 224 * 3];
	fread(input, 1, 224 * 224 * 3, fp);
	for (i = 0;i < 224 * 224 * 3;i++)
		if (input[i] != image[i])
			printf("image[%d](%d)!=input(%d)\n", i, image[i], input[i]);
	fclose(fp);
	*/

	for (i = 0;i < batch_size;i++)
		for (j = 0;j<channel;j++)
			for (k = 0;k < h*w;k++)
			{
				// pixel level proprocess
				norm1 = (float)image[i*w*h*channel + j*h*w + k] / 255;
				norm2 = (norm1 - norm_param[0][j]) / norm_param[1][j];
				ppro[i*w*h*channel + j*h*w + k] = norm2;
			}
	//mse(ppro, h*w*3, 0, "input.data", h*w*3);
	/*
	for (i = 0;i < batch_size;i++)
		for (j = 0;j<channel;j++)
			for (k = 0;k < h*w;k++)
			{
				// pixel level proprocess
				//norm1 = (float)image[i*w*h*channel + j*h*w + k] / 255;
				norm2 = (ppro[i*w*h*channel + j*h*w + k] - norm_param[0][j]) / norm_param[1][j];
				ppro[i*w*h*channel + j*channel + k] = norm1;
			}
	*/
	//mse(ppro, h*w*channel, 0, "input.data", h*w*3*5);
	return 0;
}
int rgb_data::loadPred_GPU(void*v)
{
	cudaDeviceSynchronize();
	cudaMemcpy(pred, v, sizeof(convtype)*batch_size*NUM_CLASSES, cudaMemcpyDeviceToHost);
	return 0;
}
double rgb_data::batch_accuracy(int topk)
{
	topk = topk > 5 ? topk : 5;
	int i, j, k, l, count[3] = {0};
	int *pos=new int[topk];
	float *value = new float[topk+1], temp;
	value[topk] = 1000.0;//MAX VALUE
	double accr=0;
	for (i = 0; i < batch_size; i++)
	{
		for (j = 0;j < topk;j++)value[j] = -1000.0;//MIN VALUE
		for (j = 0;j < NUM_CLASSES;j++)
		{
			temp = pred[i*NUM_CLASSES + j];
			if (temp < value[0])continue;
			for (k = 0;k < topk;k++)
			{
				if (value[k + 1] > temp)
				{
					for (l = 0;l < k;l++)
					{
						value[l] = value[l + 1];
						pos[l] = pos[l + 1];
					}
					value[k] = temp;
					pos[k] = j;
					break;
				}
			}
		}
		// top1 accr
		if (label[i] == pos[topk-1])
		{
			top1_true[batch_size*iter + i] = true;
			count[0]++;
		}
		// top5 accr
		for (j = 0;j < 5;j++)
			if (label[i] == pos[topk - 1 - j])
			{
				top5_true[batch_size*iter + i] = true;
				count[1]++;
				break;
			}
		// topk accr
		for (j = 0;j<topk;j++)
			if (label[i] == pos[topk - 1 - j])
			{
				count[2]++;
				break;
			}
		//printf("%d:acc1:%d,acc5:%d\n", i, top1_true[batch_size*iter + i], top5_true[batch_size*iter + i]);
	}
	printf("iter%d:", iter);
	printf("top1:%.2f%%,	top5:%.2f%%,	top%d:%.2f%%\n", count[0] * 100.0 / batch_size, count[1] * 100.0 / batch_size, topk, count[2] * 100.0 / batch_size);
	delete[] pos;
	delete[] value;
	return accr;
}
double rgb_data::accuracy(void)
{
	int i;
	int count1 = 0, count5 = 0;
	for (i = 0;i < NUM_IMAGES;i++)
	{
		if (top1_true[i])count1++;
		if (top5_true[i])count5++;
	}
	printf("total:top1:%.2f%%,	top5:%.2f%%\n", count1*100.0 / NUM_IMAGES, count5 * 100.0 / NUM_IMAGES);
	return 0;
}
int rgb_data::save_pred_as(const char* filename)
{
	FILE  *fp;
	if (fopen_s(&fp, filename, "ab"))
		printf("open saved file failed\n");
	fwrite(pred, sizeof(convtype), NUM_CLASSES*batch_size, fp);
	fclose(fp);
	return 0;
}
rgb_data::~rgb_data(void)
{
	free(image);
	free(ppro);
	free(label);
	free(pred);
}
