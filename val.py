import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
# from medcam import medcam
from torchsummary import summary
# from torchstat import stat
# from albumentations.augmentations import transforms
# from albumentations.augmentations.geometric import rotate
# from albumentations.augmentations.geometric import resize
# from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import archs
from dataset import Dataset
from metrics import iou_score, dice_coef, sensitivity, ppv, accuracy, tp, tn, fp, fn
from utils import AverageMeter
import time
import shutil
# from features_map import draw_features
import SimpleITK as itk

def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # for i in range(width * height):
    # plt.subplot(height, width, 1)
    # plt.axis('off')
    # img = x[0, :, :, :]
    img = x
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)# 转成unit8

    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
    plt.imshow(img)
    # print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))

def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    listpath = os.listdir(path)
    # listpath.sort(key=lambda x:int(x.split('_')[1]))
    for file in range(0, len(listpath)):
        print(listpath[file])
        # name = listpath[file].split('_')[0] + '_' + listpath[file].split('_')[1]
        name = listpath[file].split('_')[0]
        if name == case:
            file_path = path +'/'+ listpath[file]
            tmp_list.append(file_path)
    # tmp_list.sort(key=lambda x:int(x.split('.')[0].split('_')[-1]))
    return tmp_list

savepath = r'F:\python-ccta\hp\features'
if not os.path.exists(savepath):
    os.mkdir(savepath)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# device_ids = [0,1,2,3]
device_ids = [0]
"""
需要指定参数：--name dsb2018_96_NestedUNet_wDS
"""
model_path = r'F:\python-ccta\hp\models\data_ccta_Unet_2023-03-09-09-25_NDS'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='data_ccta_Unet_2023-03-09-09-25_NDS',
                        help='model name')

    args = parser.parse_args()

    return args

#test_dataset = r'F:\processed_dataset\new_test_dataset\patient9'
test_dataset = r'F:\python-ccta\hp\inputs\test_dataset'
#test_dataset = r'F:\processed_dataset\new_center_test_dataset'


# test_dataset = r'C:\Users\shinkou\Desktop\hp\hp_unet++\inputs\dsb2018_962'
def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['train'] = False
        config['train']= False


    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    start = time.time()
    model = model.cuda()
    # summary(model, (1, 96, 96))

    # Data loading code
    img_ids = glob(os.path.join('inputs', test_dataset, 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.99, random_state=None)
    # model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))


    model.eval()

    # val_transform = Compose([
    #     resize.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])
    val_dataset = Dataset(
        img_ids=img_ids,
        img_dir=os.path.join('inputs', test_dataset, 'images'),
        mask_dir=os.path.join('inputs', test_dataset, 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'])
    print(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        # num_workers=1,
        drop_last=False)

    avg_meters = {'train_loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'DM': AverageMeter(),
                  'Recall': AverageMeter(),
                  'Precision': AverageMeter(),
                  'FP':AverageMeter(),
                  'FN':AverageMeter(),
                  'TP':AverageMeter(),
                  'TN':AverageMeter()}

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)


            time_used = time.time() - start
            iou = iou_score(output, target)
            DM = dice_coef(output, target)
            Recall = sensitivity(output, target)
            Precision = ppv(output, target)
            #Accuracy =  accuracy(output, target)
            TP = tp(output, target)
            TN = tn(output, target)
            FP = fp(output, target)
            FN = fn(output, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['DM'].update(DM, input.size(0))
            avg_meters['Recall'].update(Recall, input.size(0))
            avg_meters['Precision'].update(Precision, input.size(0))
            #avg_meters['Accuracy'].update(Accuracy, input.size(0))
            avg_meters['TP'].update(TP, input.size(0))
            avg_meters['TN'].update(TN, input.size(0))
            avg_meters['FP'].update(FP, input.size(0))
            avg_meters['FN'].update(FN, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255.0).astype('uint8'))
    print('IoU: %.4f' % avg_meters['iou'].avg,
          'Dice: %.4f' % avg_meters['DM'].avg,
          'Recall: %.4f'% avg_meters['Recall'].avg,
          'Precision: %.4f'% avg_meters['Precision'].avg,
          'TP: %.4f'% avg_meters['TP'].avg,
          'TN: %.4f' % avg_meters['TN'].avg,
          'FP: %.4f' % avg_meters['FP'].avg,
          'FN: %.4f' % avg_meters['FN'].avg,
          )
    print(time_used)
    plot_examples(input, target, model, num_examples=3)
    
    torch.cuda.empty_cache()

    return avg_meters['DM'].avg, avg_meters['Recall'].avg, avg_meters['Precision'].avg

def plot_examples(datax, datay, model,num_examples=1):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.30)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    Dice, Recall, Precision = main()
    ###
    sample1=r'./outputs/data_ccta_Unet_2023-03-09-09-25_NDS/0'
    sample2 = r'./outputs/data_ccta_Unet_2023-03-09-09-25_NDS/1'
    if not os.path.exists(sample2):
        os.makedirs(sample2)

    masks_list = os.listdir(sample1)
    num_masks = len(masks_list)
    case_example = [
        '1', '2'
    ]
    volume = []
    for patient in range(0, len(case_example)):
        case = case_example[patient]
        prd_list = get_listdir(sample1)
        save_path = sample2 + '/' + case
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sum_case = 0

        for item in range(len(prd_list)):
            pre_path = prd_list[item]
            pre = plt.imread(pre_path)
            pre = pre>0.5
            sum_pre = sum(sum(pre))
            sum_case = sum_pre + sum_case
            print(sum_case)
            shutil.copy(prd_list[item], save_path)
        sum_volume = sum_case*0.0556
        volume.append(sum_volume)
        print(sum_volume)
    print('*'*20,'测试指标', '*'*20)
    print('Dice: %.4f' % Dice, 'Recall: %.4f' % Recall,'Precision: %.4f' % Precision)
    print('患者1脂肪体积分数：', volume[0],'患者2脂肪体积分数：', volume[1])


