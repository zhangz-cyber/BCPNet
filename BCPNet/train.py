import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

try:
    import swanlab
except ImportError:
    swanlab = None

from net.s2net import Net
from utils.tdataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter, poly_lr

try:
    from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
except ImportError as e:
    raise ImportError(
        "py_sod_metrics is required. Please install it by:\n"
        "pip install py_sod_metrics"
    ) from e


torch.manual_seed(2021)
np.random.seed(2021)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2021)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def structure_loss(pred, mask):
    if pred.shape[2:] != mask.shape[2:]:
        pred = F.interpolate(pred, size=mask.shape[2:], mode='bilinear', align_corners=False)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_loss(predict, target):
    if predict.shape[2:] != target.shape[2:]:
        predict = F.interpolate(predict, size=target.shape[2:], mode='bilinear', align_corners=False)

    smooth = 1
    p = 2
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    num = 2 * (predict * target).sum(dim=1) + smooth
    den = (predict.pow(p) + target.pow(p)).sum(dim=1) + smooth
    return (1 - num / den).mean()


def alignment_loss(text_feat, visual_feat):
    text_feat = F.normalize(text_feat, dim=-1)
    visual_feat = F.normalize(visual_feat, dim=-1)

    cos_sim = torch.sum(text_feat * visual_feat, dim=-1)
    loss = 1 - cos_sim.mean()

    return loss


def extract_main_prediction(outs):
    if torch.is_tensor(outs):
        return outs

    if not isinstance(outs, (list, tuple)):
        raise RuntimeError(
            f'Unexpected model output type: {type(outs)}. '
            f'Expected Tensor, list, or tuple.'
        )

    if len(outs) == 0:
        raise RuntimeError('Model returned an empty list/tuple.')

    pred_like = []
    output_info = []

    for idx, item in enumerate(outs):
        if torch.is_tensor(item):
            output_info.append((idx, tuple(item.shape)))
            if item.dim() == 4:
                pred_like.append((idx, item))
        else:
            output_info.append((idx, type(item).__name__))

    if len(pred_like) == 0:
        raise RuntimeError(
            f'No 4D prediction tensor found in model outputs. '
            f'Output info: {output_info}'
        )

    if len(outs) == 3 and len(pred_like) == 1:
        return pred_like[0][1]

    if len(outs) == 3 and len(pred_like) == 3:
        return outs[0]

    if len(outs) in [4, 5, 7]:
        if torch.is_tensor(outs[2]) and outs[2].dim() == 4:
            return outs[2]

    pred_like.sort(
        key=lambda x: x[1].shape[-2] * x[1].shape[-1],
        reverse=True
    )

    return pred_like[0][1]


def parse_train_outputs(outs, use_text):
    main_pred = extract_main_prediction(outs)
    side_preds = []
    visual_feat = None
    text_feat_proj = None

    if torch.is_tensor(outs):
        return main_pred, side_preds, visual_feat, text_feat_proj

    if not isinstance(outs, (list, tuple)):
        return main_pred, side_preds, visual_feat, text_feat_proj

    if (
        use_text
        and len(outs) == 3
        and torch.is_tensor(outs[0])
        and outs[0].dim() == 4
        and torch.is_tensor(outs[1])
        and outs[1].dim() == 2
        and torch.is_tensor(outs[2])
        and outs[2].dim() == 2
    ):
        visual_feat = outs[1]
        text_feat_proj = outs[2]
        return main_pred, side_preds, visual_feat, text_feat_proj

    if (
        len(outs) == 3
        and all(torch.is_tensor(x) and x.dim() == 4 for x in outs)
    ):
        side_preds = [outs[1], outs[2]]
        return main_pred, side_preds, visual_feat, text_feat_proj

    for item in outs:
        if torch.is_tensor(item) and item.dim() == 4 and item is not main_pred:
            if item.shape[1] == 1:
                side_preds.append(item)

    return main_pred, side_preds, visual_feat, text_feat_proj


def get_align_weight(opt, epoch):
    if opt.align_warmup_epoch <= 0:
        return opt.w_align

    warmup_ratio = min(1.0, float(epoch + 1) / float(opt.align_warmup_epoch))
    return opt.w_align * warmup_ratio


def to_uint8_mask(arr):
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.max() <= 1:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    return arr


def append_eval_csv(csv_path, row_dict):
    write_header = not os.path.exists(csv_path)

    fieldnames = [
        'epoch', 'dataset', 'Smeasure', 'wFmeasure', 'MAE',
        'adpEm', 'meanEm', 'maxEm', 'adpFm', 'meanFm', 'maxFm'
    ]

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row_dict)


def evaluate_dataset(model, data_name, opt):
    model.eval()

    data_path = os.path.join(opt.test_path, data_name)
    image_root = os.path.join(data_path, 'Imgs') + '/'
    gt_root = os.path.join(data_path, 'GT') + '/'

    if opt.use_text:
        text_root = os.path.join(opt.test_text_root, data_name)
        test_loader = test_dataset(
            image_root=image_root,
            gt_root=gt_root,
            testsize=opt.testsize,
            text_root=text_root,
            use_text=True,
            text_dim=opt.text_dim
        )
    else:
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    with torch.no_grad():
        for _ in range(test_loader.size):
            if opt.use_text:
                image, gt, _, text_feat = test_loader.load_data()
                text_feat = text_feat.cuda()
            else:
                image, gt, _ = test_loader.load_data()
                text_feat = None

            gt = to_uint8_mask(gt)
            image = image.cuda()

            outs = model(image, text_feat)
            res = extract_main_prediction(outs)

            res = F.interpolate(
                res,
                size=gt.shape,
                mode='bilinear',
                align_corners=False
            )

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred = (res * 255).astype(np.uint8)

            FM.step(pred=pred, gt=gt)
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            M.step(pred=pred, gt=gt)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": float(sm),
        "wFmeasure": float(wfm),
        "MAE": float(mae),
        "adpEm": float(em["adp"]),
        "meanEm": float(em["curve"].mean()),
        "maxEm": float(em["curve"].max()),
        "adpFm": float(fm["adp"]),
        "meanFm": float(fm["curve"].mean()),
        "maxFm": float(fm["curve"].max()),
    }

    return results


def evaluate_all(model, epoch, opt, log_file, csv_path):
    dataset_names = [x.strip() for x in opt.test_datasets.split(',') if x.strip()]
    all_results = {}

    print(f'\n===== Start Evaluation @ Epoch {epoch + 1} =====')
    log_file.write(f'\n===== Start Evaluation @ Epoch {epoch + 1} =====\n')
    log_file.flush()

    for data_name in dataset_names:
        results = evaluate_dataset(model, data_name, opt)
        all_results[data_name] = results

        msg = (
            f'{datetime.now()} Epoch [{epoch + 1:03d}/{opt.epoch:03d}] '
            f'[Test-{data_name}] '
            f'Sm: {results["Smeasure"]:.4f} | '
            f'wFm: {results["wFmeasure"]:.4f} | '
            f'MAE: {results["MAE"]:.4f} | '
            f'adpEm: {results["adpEm"]:.4f} | '
            f'meanEm: {results["meanEm"]:.4f} | '
            f'maxEm: {results["maxEm"]:.4f} | '
            f'adpFm: {results["adpFm"]:.4f} | '
            f'meanFm: {results["meanFm"]:.4f} | '
            f'maxFm: {results["maxFm"]:.4f}'
        )

        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

        append_eval_csv(csv_path, {
            'epoch': epoch + 1,
            'dataset': data_name,
            **results,
        })

        if opt.use_swanlab:
            swanlab.log({
                f"eval/{data_name}/Smeasure": results["Smeasure"],
                f"eval/{data_name}/wFmeasure": results["wFmeasure"],
                f"eval/{data_name}/MAE": results["MAE"],
                f"eval/{data_name}/adpEm": results["adpEm"],
                f"eval/{data_name}/meanEm": results["meanEm"],
                f"eval/{data_name}/maxEm": results["maxEm"],
                f"eval/{data_name}/adpFm": results["adpFm"],
                f"eval/{data_name}/meanFm": results["meanFm"],
                f"eval/{data_name}/maxFm": results["maxFm"],
                "eval/epoch": epoch + 1,
            }, step=epoch + 1)

    avg_results = {
        key: float(np.mean([res[key] for res in all_results.values()]))
        for key in next(iter(all_results.values())).keys()
    }

    avg_msg = (
        f'{datetime.now()} Epoch [{epoch + 1:03d}/{opt.epoch:03d}] [Test-AVG] '
        f'Sm: {avg_results["Smeasure"]:.4f} | '
        f'wFm: {avg_results["wFmeasure"]:.4f} | '
        f'MAE: {avg_results["MAE"]:.4f} | '
        f'adpEm: {avg_results["adpEm"]:.4f} | '
        f'meanEm: {avg_results["meanEm"]:.4f} | '
        f'maxEm: {avg_results["maxEm"]:.4f} | '
        f'adpFm: {avg_results["adpFm"]:.4f} | '
        f'meanFm: {avg_results["meanFm"]:.4f} | '
        f'maxFm: {avg_results["maxFm"]:.4f}'
    )

    print(avg_msg)
    log_file.write(avg_msg + '\n')
    log_file.flush()

    append_eval_csv(csv_path, {
        'epoch': epoch + 1,
        'dataset': 'AVG',
        **avg_results,
    })

    if opt.use_swanlab:
        swanlab.log({
            "eval_avg/Smeasure": avg_results["Smeasure"],
            "eval_avg/wFmeasure": avg_results["wFmeasure"],
            "eval_avg/MAE": avg_results["MAE"],
            "eval_avg/adpEm": avg_results["adpEm"],
            "eval_avg/meanEm": avg_results["meanEm"],
            "eval_avg/maxEm": avg_results["maxEm"],
            "eval_avg/adpFm": avg_results["adpFm"],
            "eval_avg/meanFm": avg_results["meanFm"],
            "eval_avg/maxFm": avg_results["maxFm"],
            "eval/epoch": epoch + 1,
        }, step=epoch + 1)

    return all_results, avg_results


def train(train_loader, model, optimizer, epoch, total_step, opt, log_file, global_step):
    model.train()

    loss_record1 = AvgMeter()
    loss_record2 = AvgMeter()
    loss_record3 = AvgMeter()
    loss_record_align = AvgMeter()

    align_weight = get_align_weight(opt, epoch)

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        if opt.use_text:
            images, gts, edges, text_feats = pack
            text_feats = Variable(text_feats).cuda()
        else:
            images, gts, edges = pack
            text_feats = None

        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()

        outs = model(images, text_feats)

        lateral_map_1, side_preds, visual_feat, text_feat_proj = parse_train_outputs(
            outs=outs,
            use_text=opt.use_text
        )

        loss1 = structure_loss(lateral_map_1, gts)

        loss2 = torch.tensor(0.0, device=images.device)
        loss3 = torch.tensor(0.0, device=images.device)

        if len(side_preds) >= 1 and opt.w_side2 > 0:
            loss2 = structure_loss(side_preds[0], gts)

        if len(side_preds) >= 2 and opt.w_side3 > 0:
            loss3 = structure_loss(side_preds[1], gts)

        loss_align = torch.tensor(0.0, device=images.device)
        if opt.use_text and visual_feat is not None and text_feat_proj is not None:
            loss_align = alignment_loss(text_feat_proj, visual_feat)

        loss = loss1 + opt.w_side2 * loss2 + opt.w_side3 * loss3

        if opt.use_text:
            loss = loss + align_weight * loss_align

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        global_step += 1

        loss_record1.update(loss1.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)

        if opt.use_text:
            loss_record_align.update(loss_align.data, opt.batchsize)

        if opt.use_swanlab:
            log_dict = {
                "train/loss_total": float(loss.item()),
                "train/loss1": float(loss1.item()),
                "train/loss_side2": float(loss2.item()),
                "train/loss_side3": float(loss3.item()),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/epoch": epoch + 1,
            }

            if opt.use_text:
                log_dict["train/loss_align"] = float(loss_align.item())
                log_dict["train/align_weight"] = float(align_weight)

            swanlab.log(log_dict, step=global_step)

        if i % 60 == 0 or i == total_step:
            if opt.use_text:
                msg = (
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    '[l1: {:.4f}] [side2: {:.4f}] [side3: {:.4f}] '
                    '[align: {:.4f}] [align_w: {:.6f}]'
                ).format(
                    datetime.now(), epoch + 1, opt.epoch, i, total_step,
                    loss_record1.avg,
                    loss_record2.avg,
                    loss_record3.avg,
                    loss_record_align.avg,
                    align_weight
                )
            else:
                msg = (
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    '[l1: {:.4f}] [side2: {:.4f}] [side3: {:.4f}]'
                ).format(
                    datetime.now(), epoch + 1, opt.epoch, i, total_step,
                    loss_record1.avg,
                    loss_record2.avg,
                    loss_record3.avg
                )

            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

    save_path = 'checkpoints/newnew/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch + 1) % 1 == 0 or (epoch + 1) == opt.epoch:
        ckpt_path = save_path + 'S2Net-%d.pth' % epoch
        torch.save(model.state_dict(), ckpt_path)

        msg = '[Saving Snapshot:] ' + ckpt_path
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    return global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ============================================================
    # Basic training settings
    # ============================================================
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # DINOv2 + token text 更吃显存，默认先用 4
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')

    # DINOv2 ViT-B/14 推荐使用 448 = 32 * 14
    parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
    parser.add_argument('--testsize', type=int, default=448, help='testing dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

    # ============================================================
    # Dataset settings
    # ============================================================
    parser.add_argument('--train_path', type=str, default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--test_path', type=str, default='./data/TestDataset', help='path to test datasets')
    parser.add_argument(
        '--test_datasets',
        type=str,
        default='CAMO,CHAMELEON,COD10K,NC4K',
        help='comma separated test datasets'
    )

    parser.add_argument('--train_save', type=str, default='S2Net_DINOv2_CGCOD_tokennew_text')

    # ============================================================
    # Loss settings
    # ============================================================
    parser.add_argument('--w_pr', type=float, default=1.5, help='weight for region prior loss')
    parser.add_argument('--w_align', type=float, default=0.005, help='weight for text-visual alignment loss')
    parser.add_argument('--align_warmup_epoch', type=int, default=5, help='warmup epochs for alignment loss')

    parser.add_argument('--w_side2', type=float, default=0.4, help='weight for side output 2')
    parser.add_argument('--w_side3', type=float, default=0.2, help='weight for side output 3')

    # ============================================================
    # Optimizer settings
    # ============================================================
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')

    # ============================================================
    # Backbone settings
    # ============================================================
    parser.add_argument(
        '--backbone',
        type=str,
        default='dinov2',
        choices=['res2net', 'swin', 'pvt', 'dinov2'],
        help='visual backbone'
    )

    parser.add_argument(
        '--swin_variant',
        type=str,
        default='swin_b_384_22k',
        choices=[
            'swin_b_384_22k',
            'swin_l_384_22k',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b5'
        ]
    )

    parser.add_argument(
        '--swin_ckpt',
        type=str,
        default='./models/swin_base_patch4_window12_384_22k.pth'
    )

    parser.add_argument(
        '--dinov2_variant',
        type=str,
        default='dinov2_vitb14',
        choices=[
            'dinov2_vits14',
            'dinov2_vitb14',
            'dinov2_vitl14',
            'dinov2_vitg14',
        ],
        help='DINOv2 backbone variant'
    )

    parser.add_argument(
        '--dinov2_ckpt',
        type=str,
        default='./models/dinov2/dinov2_vitb14_pretrain.pth',
        help='local DINOv2 checkpoint path'
    )

    parser.add_argument(
        '--dinov2_repo',
        type=str,
        default='./external/dinov2',
        help='local DINOv2 repo path'
    )

    parser.add_argument(
        '--dinov2_source',
        type=str,
        default='local',
        choices=['local', 'github'],
        help='load DINOv2 from local repo or GitHub torch hub'
    )

    # ============================================================
    # Text settings
    # ============================================================
    parser.add_argument('--use_text', dest='use_text', action='store_true', help='use text embeddings')
    parser.add_argument('--no_text', dest='use_text', action='store_false', help='disable text embeddings')
    parser.set_defaults(use_text=True)

    parser.add_argument('--text_dim', type=int, default=512, help='dimension of CLIP token embedding')

    parser.add_argument(
        '--train_text_root',
        type=str,
        default='/sda/home/shihaoyu/Projects/S2Net/multimodal_clip_tokennew/train',
        help='directory of training token-level text pt files'
    )

    parser.add_argument(
        '--test_text_root',
        type=str,
        default='/sda/home/shihaoyu/Projects/S2Net/multimodal_clip_tokennew',
        help='root directory of test token-level text pt folders'
    )

    parser.add_argument('--text_hidden_dim', type=int, default=256, help='hidden dim for text projection')
    parser.add_argument('--text_dropout', type=float, default=0.1, help='dropout for text projection')
    parser.add_argument('--text_scale', type=float, default=0.05, help='feature modulation scale for text FiLM')

    # ============================================================
    # SwanLab settings
    # ============================================================
    parser.add_argument('--use_swanlab', dest='use_swanlab', action='store_true', help='use SwanLab visualization')
    parser.add_argument('--no_swanlab', dest='use_swanlab', action='store_false', help='disable SwanLab visualization')
    parser.set_defaults(use_swanlab=True)

    parser.add_argument('--swanlab_project', type=str, default='S2Net-COD', help='SwanLab project name')
    parser.add_argument(
        '--swanlab_exp_name',
        type=str,
        default='S2Net_DINOv2_CGCOD_token_text',
        help='SwanLab experiment name'
    )
    parser.add_argument(
        '--swanlab_mode',
        type=str,
        default='offline',
        choices=['cloud', 'offline', 'disabled'],
        help='SwanLab running mode'
    )

    opt = parser.parse_args()

    if opt.backbone == 'swin':
        if opt.trainsize != 384 or opt.testsize != 384:
            raise ValueError('Using swin_*_384_22k requires --trainsize 384 --testsize 384')

    if opt.backbone == 'dinov2':
        if opt.trainsize % 14 != 0 or opt.testsize % 14 != 0:
            raise ValueError('Using DINOv2 ViT-*/14 requires --trainsize and --testsize divisible by 14.')

        if opt.dinov2_source == 'local' and not os.path.isdir(opt.dinov2_repo):
            raise FileNotFoundError(f'DINOv2 local repo not found: {opt.dinov2_repo}')

        if opt.dinov2_ckpt and not os.path.isfile(opt.dinov2_ckpt):
            raise FileNotFoundError(f'DINOv2 checkpoint not found: {opt.dinov2_ckpt}')

    if opt.use_swanlab and swanlab is None:
        raise ImportError('swanlab is not installed, but --use_swanlab was specified.')

    if opt.use_text:
        if not os.path.isdir(opt.train_text_root):
            raise FileNotFoundError(f'Train text root not found: {opt.train_text_root}')

        if not os.path.isdir(opt.test_text_root):
            raise FileNotFoundError(f'Test text root not found: {opt.test_text_root}')

    os.makedirs('log', exist_ok=True)

    txt_log_path = os.path.join('log', f'{opt.train_save}.txt')
    csv_log_path = os.path.join('log', f'{opt.train_save}_eval.csv')
    log_file = open(txt_log_path, 'a', encoding='utf-8')

    if opt.use_swanlab:
        swanlab.init(
            project=opt.swanlab_project,
            experiment_name=opt.swanlab_exp_name,
            config={
                "model": f"S2Net-{opt.backbone}",
                "backbone": opt.backbone,
                "swin_variant": opt.swin_variant,
                "swin_ckpt": opt.swin_ckpt,
                "dinov2_variant": opt.dinov2_variant,
                "dinov2_ckpt": opt.dinov2_ckpt,
                "dinov2_repo": opt.dinov2_repo,
                "dinov2_source": opt.dinov2_source,
                "lr": opt.lr,
                "epoch": opt.epoch,
                "batchsize": opt.batchsize,
                "trainsize": opt.trainsize,
                "testsize": opt.testsize,
                "train_path": opt.train_path,
                "test_path": opt.test_path,
                "test_datasets": opt.test_datasets,
                "w_pr": opt.w_pr,
                "w_align": opt.w_align,
                "align_warmup_epoch": opt.align_warmup_epoch,
                "w_side2": opt.w_side2,
                "w_side3": opt.w_side3,
                "optimizer": opt.optimizer,
                "weight_decay": opt.weight_decay,
                "use_text": opt.use_text,
                "train_text_root": opt.train_text_root,
                "test_text_root": opt.test_text_root,
                "text_dim": opt.text_dim,
                "text_hidden_dim": opt.text_hidden_dim,
                "text_dropout": opt.text_dropout,
                "text_scale": opt.text_scale,
            },
            logdir="swanlog",
            mode=opt.swanlab_mode,
        )

    model = Net(
        backbone=opt.backbone,
        swin_variant=opt.swin_variant,
        swin_ckpt=opt.swin_ckpt,
        dinov2_variant=opt.dinov2_variant,
        dinov2_ckpt=opt.dinov2_ckpt,
        dinov2_repo=opt.dinov2_repo,
        dinov2_source=opt.dinov2_source,
        use_text=opt.use_text,
        text_dim=opt.text_dim,
        text_hidden_dim=opt.text_hidden_dim,
        text_dropout=opt.text_dropout,
        text_scale=opt.text_scale
    ).cuda()

    if opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        edge_root=edge_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        text_root=opt.train_text_root if opt.use_text else None,
        use_text=opt.use_text,
        text_dim=opt.text_dim
    )

    total_step = len(train_loader)

    print("Start Training")
    print(f"Backbone: {opt.backbone}")

    if opt.backbone == 'swin':
        print(f"Swin variant: {opt.swin_variant}")
        print(f"Swin ckpt: {opt.swin_ckpt}")

    if opt.backbone == 'dinov2':
        print(f"DINOv2 variant: {opt.dinov2_variant}")
        print(f"DINOv2 ckpt: {opt.dinov2_ckpt}")
        print(f"DINOv2 repo: {opt.dinov2_repo}")
        print(f"DINOv2 source: {opt.dinov2_source}")

    print(f"Use text: {opt.use_text}")
    print(f"Optimizer: {opt.optimizer}")
    print(f"Weight decay: {opt.weight_decay}")

    if opt.use_text:
        print(f"Train text root: {opt.train_text_root}")
        print(f"Test text root: {opt.test_text_root}")
        print(f"w_align: {opt.w_align}")
        print(f"align_warmup_epoch: {opt.align_warmup_epoch}")

    print(f"w_side2: {opt.w_side2}")
    print(f"w_side3: {opt.w_side3}")

    log_file.write(f'\n===== Training Start @ {datetime.now()} =====\n')
    log_file.write(f'Backbone: {opt.backbone}\n')

    if opt.backbone == 'swin':
        log_file.write(f'Swin variant: {opt.swin_variant}\n')
        log_file.write(f'Swin ckpt: {opt.swin_ckpt}\n')

    if opt.backbone == 'dinov2':
        log_file.write(f'DINOv2 variant: {opt.dinov2_variant}\n')
        log_file.write(f'DINOv2 ckpt: {opt.dinov2_ckpt}\n')
        log_file.write(f'DINOv2 repo: {opt.dinov2_repo}\n')
        log_file.write(f'DINOv2 source: {opt.dinov2_source}\n')

    log_file.write(f'Use text: {opt.use_text}\n')
    log_file.write(f'Optimizer: {opt.optimizer}\n')
    log_file.write(f'Weight decay: {opt.weight_decay}\n')
    log_file.write(f'w_side2: {opt.w_side2}\n')
    log_file.write(f'w_side3: {opt.w_side3}\n')

    if opt.use_text:
        log_file.write(f'Train text root: {opt.train_text_root}\n')
        log_file.write(f'Test text root: {opt.test_text_root}\n')
        log_file.write(f'w_align: {opt.w_align}\n')
        log_file.write(f'align_warmup_epoch: {opt.align_warmup_epoch}\n')

    log_file.flush()

    global_step = 0

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)

        global_step = train(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            total_step=total_step,
            opt=opt,
            log_file=log_file,
            global_step=global_step
        )

        evaluate_all(
            model=model,
            epoch=epoch,
            opt=opt,
            log_file=log_file,
            csv_path=csv_log_path
        )

    log_file.write(f'===== Training End @ {datetime.now()}\n')
    log_file.flush()
    log_file.close()

    if opt.use_swanlab:
        swanlab.finish()