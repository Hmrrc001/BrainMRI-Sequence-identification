#  test image and put it into the corresponding folder, don't need labels
import os
import shutil
from monai.data import Dataset

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import CSVSaver
from monai.transforms import Compose, LoadImage, ScaleIntensityd, EnsureTyped
from EfficientNet import MyEfficientNet
from torch.nn.functional import interpolate


class MRIDataset(Dataset):

    def __init__(self, data, stack, transform):
        '''
        :param data: data list, only image
        :param stack: the number of image slice stack
        :param transform: convention transform
        '''
        self.data = data
        self.stack = stack
        self.transform = transform
        self.load = LoadImage()

        self._build_stack_img()

    def _build_stack_img(self):
        self.new_data = []
        for img in self.data:
            imgarr = self.load(img)
            imgname = img.split('/')[-1]
            img_channel = np.moveaxis(imgarr[0], -1, 0)   # move z-axis to first channel

            img_channel = torch.from_numpy(img_channel).unsqueeze(0).unsqueeze(0)
            img_resize = interpolate(
                input=img_channel,  # type: ignore
                size=(20, 128, 128),
            )
            img_resize = img_resize.squeeze()

            start_point = int((20 - self.stack) / 2)
            img_stack = img_resize[start_point: start_point + self.stack, ...]
            data_dict = {'img': img_stack, 'name': imgname}
            self.new_data.append(data_dict)
        return self.new_data

    # get data operation
    def __getitem__(self, index):
        data = self.new_data[index]
        augments_data = self.transform(data)
        return augments_data

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.new_data)


def main(loader, model_path, stack, classes, input_path, output_path):

    model = MyEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=stack,
                           num_classes=len(classes), pretrained=False, dropout_rate=0.0).to(device)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with torch.no_grad():
        count = 0
        for data in loader:
            images = data["img"].to(device)
            outputs = model(images).argmax(dim=1)
            class_folder = classes[outputs]

            # save img to class folder
            name = data['name'][0]
            src = os.path.join(input_path, name)
            dsc = os.path.join(output_path, class_folder, name)
            shutil.copy(src, dsc)

            count += 1
            print(count)


if __name__ == "__main__":
    from glob import glob
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description="test and split images")
    parser.add_argument('--dataroot', type=str, default="/media/sun/FJJ/guoting/data/MRISeq/0628split/FLAIR_tra")
    parser.add_argument("--output_path", type=str, default='output')
    parser.add_argument('--classes_csv', type=str, default='Results/EfficietNet/20220809_stack8_lr1e-05_epochs100/log/classes.csv')
    parser.add_argument('--model_path', type=str, default='Results/EfficietNet/20220809_stack8_lr1e-05_epochs100/Saved_models/best_model.pth')
    parser.add_argument('--stack', default=8, type=int, help='input channel')
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    args = parser.parse_args()

    # to set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get data
    imgs = glob(args.dataroot + '/*.gz')
    print(f'There are {len(imgs)} images')

    val_transforms = Compose(
        [
            ScaleIntensityd(keys='img'),
            EnsureTyped(keys=['img'])
        ]
    )
    ds = MRIDataset(data=imgs, stack=args.stack, transform=val_transforms)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, pin_memory=torch.cuda.is_available())

    # get classes from log
    class_file = pd.read_csv(args.classes_csv, index_col=0, header=None)
    classes = class_file.index.tolist()
    class_to_idx = class_file.to_dict()[1]
    print(class_to_idx)

    #  创建文件夹
    for i in classes:
        if not os.path.exists(os.path.join(output_path, i)):
            os.makedirs(os.path.join(output_path, i))

    # to save csv file and images in output_path
    main(loader, args.model_path, args.stack, classes, args.dataroot, output_path)
