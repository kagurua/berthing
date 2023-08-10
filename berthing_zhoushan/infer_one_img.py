import os
import time

from berthing_zhoushan.DeepLab_utils.modeling.deeplab import DeepLab
from berthing_zhoushan.DeepLab_utils.dataloaders import custom_transforms as tr
from PIL import Image
import torch
from torchvision import transforms

from berthing_zhoushan.DeepLab_utils.dataloaders.utils import *
from torchvision.utils import make_grid, save_image


def load_model():

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # load model
    file_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_file = os.path.join(file_path, 'DeepLab_utils/run/cityscapes_berthing_1/deeplab-resnet/model_best.pth.tar')
    model_s_time = time.time()
    model = DeepLab(num_classes=7,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)
    ckpt = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time - model_s_time
    print('model load time:', model_load_time)

    return model


def gain_img_seg(model, img):

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    s_time = time.time()
    img = img[..., ::-1]
    img = Image.fromarray(img)
    image = img.convert('RGB')
    target = img.convert('L')
    sample = {'image': image, 'label': target}
    tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

    model.eval()
    tensor_in = tensor_in.cuda()
    with torch.no_grad():
        output = model(tensor_in)

    semantic_ids_img = output[:3].detach().cpu().numpy()[0]

    semantic_id_img = torch.max(output[:3], 1)[1].detach().cpu().numpy()[0]

    grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                           3, normalize=False, range=(0, 255))

    u_time = time.time()
    img_time = u_time - s_time
    print("Infer time: {} ".format(img_time))

    return semantic_ids_img, grid_image


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # load model
    ckpt_file = './DeepLab_utils/run/cityscapes_berthing/deeplab-resnet/model_best.pth.tar'
    model_s_time = time.time()
    model = DeepLab(num_classes=7,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)
    ckpt = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time - model_s_time
    print("model load time is {}".format(model_load_time))

    # open a img
    img_file = '/media/dataset/cityscapes_berthing/Test_Imgs/010075.png'

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    s_time = time.time()
    image = Image.open(img_file).convert('RGB')
    target = Image.open(img_file).convert('L')
    sample = {'image': image, 'label': target}
    tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

    model.eval()
    tensor_in = tensor_in.cuda()
    with torch.no_grad():
        output = model(tensor_in)
    print(output[0].detach().cpu().numpy().shape)

    semantic_id_img = torch.max(output[:3], 1)[1].detach().cpu().numpy()[0]
    print(semantic_id_img[[0, 1], [1200, 100]])

    u_time = time.time()

    # save_output_grid_img
    # out_path = '/media/dataset/cityscapes_berthing/Test_Results'
    # grid_image = make_grid(decode_seg_map_sequence(results), 3, normalize=False, range=(0, 255))
    # save_image(grid_image, out_path + "/" + "{}_mask.png".format(img_file[-10:-4]))

    img_time = u_time - s_time
    print("image:{}, Infer time: {} ".format(img_file, img_time))


if __name__ == "__main__":
    main()
