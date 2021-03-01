import torch
from torch.autograd import Variable
from vdsr import Net
import numpy as np
import time, math, glob, os
import scipy.io as sio

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def main(model,blu,dataset):
    scales = [2, 3, 4]
    image_list = glob.glob(dataset + "_mat/*.*")
    result=''
    for scale in scales:
        avg_psnr_predicted = 0.0
        avg_psnr_bicubic = 0.0
        avg_elapsed_time = 0.0
        count = 0.0
        for image_name in image_list:
            if 'x'+str(scale) in image_name:
                count += 1
                im_gt_y = sio.loadmat(image_name)['im_gt_y']
                im_b_y = sio.loadmat(image_name)['im_b_y']

                im_gt_y = im_gt_y.astype(np.float32)
                im_b_y = im_b_y.astype(np.float32)

                psnr_bicubic = PSNR(im_gt_y.round(), im_b_y.round(), shave_border=scale)
                avg_psnr_bicubic += psnr_bicubic

                im_input = (im_b_y.round() - 128) / 255.0  # renorm
                # im_input = im_b_y/255.0#original

                im_input = Variable(torch.from_numpy(im_input)).view(1, -1, im_input.shape[0], im_input.shape[1])

                im_input = im_input.cuda()

                start_time = time.time()
                if blu:
                    residual = model.forward_blu(im_input)
                else:
                    residual = model(im_input)
                elapsed_time = time.time() - start_time
                avg_elapsed_time += elapsed_time

                residual = residual[0, 0].detach().cpu().numpy()
                im_h_y = residual * 255 + im_b_y
                im_h_y[im_h_y < 0] = 0
                im_h_y[im_h_y > 255.] = 255.
                im_h_y = im_h_y.round()

                psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
                avg_psnr_predicted += psnr_predicted
                #print(image_name,':Bicubic ',psnr_bicubic,'predicted:',psnr_predicted)
        result+="Scale=%d, PSNR_bicubic=%.3f PSNR_predicted=%.3f\n" % (scale,avg_psnr_bicubic/count, avg_psnr_predicted / count)
    print(result)
    return result

if __name__ =='__main__':
    model_path='model\\model_bias_blu_epoch_54.pth'
    gpu=0
    quant=True
    quant_param='quant.data'
    blu=True
    torch.cuda.set_device(gpu)
    model = Net()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"].state_dict(),strict=False)
    print('loaded ' + model_path)

    #model.dump('model.data')
    #exit(0)
    model.cuda()
    if quant:
        if os.path.isfile(quant_param):
            model.quantize_from(quant_param)
            print('model quantized from ' + quant_param)
        else:
            model.quantize('quant.data')
    if blu:
        model.load_blu('blu_train.data')
        print('Loaded BLU from ' + 'blu_train.data')
    main(model,blu,"Set14")