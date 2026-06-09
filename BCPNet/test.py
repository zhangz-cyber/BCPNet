import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from net.BCPNet import Net
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()

parser.add_argument(
    '--testsize',
    type=int,
    default=384,
    help='testing size'
)

parser.add_argument(
    '--pth_path',
    type=str,
    default='./checkpoints/BCPNet/BCPNet.pth',
    help='path to model weight file'
)

parser.add_argument(
    '--save_root',
    type=str,
    default='./results/BCPNet',
    help='root dir to save prediction results'
)

opt = parser.parse_args()

# ===========================
# Load Model
# ===========================
model = Net().cuda()

state = torch.load(
    opt.pth_path,
    map_location='cpu'
)

model.load_state_dict(state)

model.cuda()
model.eval()

# ===========================
# Testing
# ===========================
for _data_name in [
    'CAMO',
    'CHAMELEON',
    'COD10K',
    'NC4K'
]:

    data_path = f'./data/TestDataset/{_data_name}/'

    save_path = os.path.join(
        opt.save_root,
        _data_name
    )

    os.makedirs(save_path, exist_ok=True)

    image_root = f'{data_path}/Imgs/'
    gt_root = f'{data_path}/GT/'

    test_loader = test_dataset(
        image_root=image_root,
        gt_root=gt_root,
        testsize=opt.testsize
    )

    for i in range(test_loader.size):

        image, gt, name = test_loader.load_data()

        gt = np.asarray(
            gt,
            np.float32
        )

        gt /= (gt.max() + 1e-8)

        image = image.cuda()

        with torch.no_grad():

            res = model(image)

            res = F.interpolate(
                res,
                size=gt.shape,
                mode='bilinear',
                align_corners=False
            )

            res = (
                res.sigmoid()
                .cpu()
                .numpy()
                .squeeze()
            )

            res = (
                res - res.min()
            ) / (
                res.max() - res.min() + 1e-8
            )

            imageio.imwrite(
                os.path.join(save_path, name),
                (res * 255).astype(np.uint8)
            )

    print(f'{_data_name} Done.')