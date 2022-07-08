from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torch
import torchvision.models as models
from utils import data_transform
from torchvision.models.resnet import BasicBlock
import os


def save_image(save_prediction_path, model, save_path, image_name):
    # save one middle layer of ResNet, target_layers ranges from 1 to 4

    # load image
    transform = data_transform['test']
    img = transform(image)
    img = img.unsqueeze(0)

    for idx, target_layers in enumerate([model.layer1[-1],
                                         model.layer2[-1],
                                         model.layer3[-1],
                                         model.layer4[-1]]):
        input_tensor = img

        # CAM
        target_layers = [target_layers]
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
        target_category = 0

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]

        img_rgb = img.squeeze(0).permute(1, 2, 0).numpy()
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        sub_dir = image_name.split('.')[0]
        save_name = 'heatmap-' + sub_dir + '-layer' + str(idx + 1) + '.jpg'
        sub_save_dir = os.path.join(save_path, sub_dir)
        make_dir(sub_save_dir)
        sub_save_path = os.path.join(sub_save_dir, save_name)
        Image.fromarray(visualization).save(sub_save_path)

    save_prediction(save_prediction_path, img)


def save_prediction(save_prediction_path, img_tensor):
    with open(save_prediction_path, 'a', encoding='utf-8') as f:
        score = torch.sigmoid(model(img_tensor.cuda()))
        score = str(score.cpu().data.numpy()[0][0])
        text = image_name + "\t" + score
        print(text)
        f.write(text)
        f.write('\r')


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    # loading trained ResNet
    model_name = 'sem_model'
    model_path = './save_models/' + model_name + '.pth'
    model_dict = torch.load(model_path)
    model = models.ResNet(BasicBlock, [1, 1, 1, 1], num_classes=1)
    model.load_state_dict(model_dict)
    model.eval()

    # roots of original images and visualizing saving path
    original_images_root = '.\\origin_image'
    save_root_dir = ".\\visualizing_model_images"
    make_dir(original_images_root)
    make_dir(save_root_dir)

    for sub_dir in os.listdir(original_images_root):
        sub_dir_path = os.path.join(original_images_root, sub_dir)
        image_list = os.listdir(sub_dir_path)
        save_dir = sub_dir
        save_path = os.path.join(save_root_dir, save_dir)
        log_path = os.path.join(save_path, 'Possibility.txt')
        make_dir(save_path)

        for image_name in image_list:
            p = os.path.join(sub_dir_path, image_name)
            image = Image.open(p)
            save_image(log_path, model, save_path, image_name)
