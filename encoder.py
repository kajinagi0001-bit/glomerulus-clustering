#!/usr/bin/env python
# Copyright (c) Meta Platforms,
# MIT License

import argparse
import math
import os
import random
import shutil
import time
import warnings
import sys
import wandb
import numpy as np
from pathlib import Path
from PIL import Image
import tools.builder
import tools.loader
import tools.folder
import tools.evaluation_index
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from termcolor import colored
from tools.randstainna import RandStainNA

from tools.bg_augmentation import (
    GlomerulusCopyPasteAug,
    GlomerulusFixedPositionRGBBgAug,
    GlomerulusFixedPositionBgImageAug,
    RandomBackgroundAugSelector,
    GlomerulusMoCoDataset,
    GlomerulusFixedDataset
)

import cv2
# ---------------------------
# argparse
# ---------------------------
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="MoCo Pretraining with ResNet50")
parser.add_argument('--exp', default=0, type=str)
parser.add_argument('--out-path', default=None, type=str, help='output root directory (REQUIRED)')
# boolean flags: enable/disable pair
parser.add_argument("--pretrain", action="store_true", default=True, help="load medical pretrain")
parser.add_argument('--pretrain_path',default='tools/mocov2_rn50_ep200.torch', type=str)
parser.add_argument("--data-dir",type=str,default='./dataset_1203_org/rat_PAS_crop_longside_1.3')
parser.add_argument("--no-pretrain", dest="pretrain", action="store_false")
parser.add_argument("-a", "--arch", default="resnet50", choices=model_names)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--start-epoch", default=0, type=int)
parser.add_argument("-b", "--batch-size", default=32, type=int)
parser.add_argument("--lr-conv", default=3e-5, type=float, dest="lr_conv")
parser.add_argument("--lr-fc", default=1e-2, type=float, dest="lr_fc")
parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=100, type=int)
parser.add_argument("--resume", default="", type=str, help="checkpoint path")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--gpu_num", default=None, type=int)
parser.add_argument("--train_mode" ,default="layer4" ,type=str, choices=['all','layer3+4','layer4'])
parser.add_argument("--moco-dim", default=128, type=int)
parser.add_argument("--moco-k", default=65536, type=int)
parser.add_argument("--moco-m", default=0.999, type=float)
parser.add_argument("--moco-t", default=0.07, type=float)
parser.add_argument("--mlp", action="store_true", default=True)
parser.add_argument("--no-mlp", dest="mlp", action="store_false")
parser.add_argument("--aug-plus", action="store_true")
parser.add_argument("--cos", action="store_true", default=True)
parser.add_argument("--no-cos", dest="cos", action="store_false")
parser.add_argument("--std-hyper", default=0.8, type=float, help="RandStainNA parameter")
parser.add_argument("--ent_epstart", default=30, type=int)
parser.add_argument("--ent_epend", default=50, type=int)
parser.add_argument("--lmax", default=0.2, type=float)
# parser = argparse.ArgumentParser(description="MoCo Pretraining with Glomerulus Copy-Paste")
parser.add_argument("--mask-dir", type=str, default='./dataset_1203_org/masks')
parser.add_argument("--bg-dir", type=str, default='./wsi_background_512')
parser.add_argument(
    "--bg-mode", 
    default='copy-paste-pb', 
    type=str, 
    choices=['copy-paste-pb', 'rgb_prob', 'wsi_pb', 'rgb_wsi_prob'],
    help='背景変換の手法を選択します'
)
parser.add_argument("--fade-width", type=float, default=8.0)
parser.add_argument('--bg-prob',default=0.3,type=float)

# ---------------------------
# utilities
# ---------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_worker_init_fn(base_seed: int):
    def _fn(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _fn

# ---------------------------
# main
# ---------------------------
def main():
    args = parser.parse_args()

    if not args.out_path:
        raise ValueError("--out-path を指定してください")
    os.makedirs(args.out_path, exist_ok=True)

    if args.seed is not None:
        set_global_seed(args.seed)
    else:
        cudnn.benchmark = True

    args.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(colored('device_name ----> ', 'cyan'), colored(args.device, 'cyan'))

    main_worker(args.device, args)

def main_worker(gpu, args):
    args.gpu = gpu
    print(f"Use GPU: {args.gpu} for training")

    # ----- model -----
    print(f"=> creating model '{args.arch}'")
    encoder = models.resnet50(weights=None)

    if args.pretrain:
        print(colored('you have choiced pretrain path.', 'red'))
        pretrained_model_path = args.pretrain_path
        state_dict = torch.load(pretrained_model_path, map_location=args.gpu)
        state_dict = {k: v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict, strict=False)
    else:
        print(colored('you have not choiced pretrain path.', 'blue'))

    model = tools.builder.MoCo(
        encoder,
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    )

    # freeze-all then enable only encoder_q.layer4 and encoder_q.fc
    for p in model.parameters():
        p.requires_grad = False
    if args.train_mode == 'all':
        for p in model.parameters():
            p.requires_grad = True
    elif args.train_mode == 'layer3+4':
        for name, p in model.encoder_q.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                p.requires_grad = True
    elif args.train_mode == 'layer4':
        for name, p in model.encoder_q.named_parameters():
            if 'layer4' in name or 'fc' in name:
                p.requires_grad = True

    model = model.to(args.gpu)

    # ----- loss & optimizer -----
    criterion = nn.CrossEntropyLoss().to(args.gpu)

    conv_params = []
    fc_params = []

    for name, p in model.encoder_q.named_parameters():
        if not p.requires_grad:
            continue

        if 'fc' in name:
            fc_params.append(p)
        else:
            conv_params.append(p)
    print(f"Conv layers={len(conv_params)},FC layers={len(fc_params)}")
    optimizer_params = []
    if len(conv_params) > 0:
        optimizer_params.append({'params': conv_params, 'lr': args.lr_conv})
    if len(fc_params) > 0:
        optimizer_params.append({'params': fc_params, 'lr': args.lr_fc})
    optimizer = torch.optim.SGD(
        optimizer_params,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # ----- resume -----
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=args.gpu)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            sys.exit(1)

    # ==========================================
    # データ拡張機能の動的切り替え
    # ==========================================
    train_augmentation = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2), 
        RandStainNA(
            yaml_file='tools/rat_PAS_org.yaml', 
            std_hyper=args.std_hyper,
            probability=0.8,
            distribution='normal',
            is_train=True
        ),
        transforms.ToTensor(),
    ])

    # 背景画像リストを事前に取得 (必要なモードでのみ使用)
    bg_dir = Path(args.bg_dir)
    bg_files = [str(p) for p in bg_dir.rglob("*") if p.suffix.lower() in {'.jpg', '.png', '.tif'}]

    print(colored(f"Selected Augmentation Mode: {args.bg_mode}", "magenta"))

    if args.bg_mode == 'copy-paste-pb':
        aug_logic = GlomerulusCopyPasteAug(bg_paths=bg_files, img_size=512, target_fade_width=args.fade_width, prob=args.bg_prob)
        train_dataset = GlomerulusMoCoDataset(
            img_root=args.data_dir, mask_root=args.mask_dir, cp_aug=aug_logic, post_transform=train_augmentation
        )

    elif args.bg_mode == 'rgb_prob':
        aug_logic = GlomerulusFixedPositionRGBBgAug(img_size=512, target_fade_width=7.0, prob=args.bg_prob)
        train_dataset = GlomerulusFixedDataset(
            img_root=args.data_dir, mask_root=args.mask_dir, aug_logic=aug_logic, transform=train_augmentation
        )

    elif args.bg_mode == 'wsi_pb':
        aug_logic = GlomerulusFixedPositionBgImageAug(bg_paths=bg_files, img_size=512, target_fade_width=10.0, prob=args.bg_prob)
        train_dataset = GlomerulusFixedDataset(
            img_root=args.data_dir, mask_root=args.mask_dir, aug_logic=aug_logic, transform=train_augmentation
        )

    elif args.bg_mode == 'rgb_wsi_prob':
        aug_rgb = GlomerulusFixedPositionRGBBgAug(img_size=512, target_fade_width=7.0, prob=1.0)
        aug_image = GlomerulusFixedPositionBgImageAug(bg_paths=bg_files, img_size=512, target_fade_width=7.0, prob=1.0)
        aug_logic = RandomBackgroundAugSelector(aug_rgb, aug_image, p_image=0.5, bg_prob=args.bg_prob)
        train_dataset = GlomerulusFixedDataset(
            img_root=args.data_dir, mask_root=args.mask_dir, aug_logic=aug_logic, transform=train_augmentation
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    generator = None
    worker_init_fn = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        worker_init_fn = make_worker_init_fn(args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    # ----- wandb -----
    try:
        wandb.init(
            project='pretrain_final',
            name=f'exp{args.exp}',
            config={
                **vars(args),
                "dataset_path": args.data_dir,
                "dataset_size": len(train_dataset),
            },
            settings=wandb.Settings(start_method="thread")
        )
    except Exception as e:
        print(colored(f"W&B init failed: {e}", "yellow"))

    # ----- train loop -----
    best_loss = math.inf
    warmup_epochs = 10

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args, warmup_epochs)

        avg_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        if wandb.run is not None:
            wandb.log({'train_loss': avg_loss, 'epoch': epoch})

        # save encoder weights
        enc_dir = os.path.join(args.out_path, f"exp{args.exp}", "encoder_weights")
        os.makedirs(enc_dir, exist_ok=True)
        torch.save(model.encoder_q.state_dict(), os.path.join(enc_dir, f"encoder_{epoch+1:04d}.pth.tar"))

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            torch.save(model.encoder_q.state_dict(), os.path.join(enc_dir, "encoder_best.pth.tar"))

        # save checkpoint
        ckpt_dir = os.path.join(args.out_path, f"exp{args.exp}", "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch+1:04d}.pth.tar")
        save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                filename=ckpt_path,
                best_dst=os.path.join(ckpt_dir, "model_best.pth.tar")
        )

    if wandb.run is not None:
        wandb.finish()

# ---------------------------
# train one epoch
# ---------------------------
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses], prefix=f"Epoch: [{epoch}]")

    model.train()

    end = time.time()
    lmax = args.lmax
    lamda_ent = lambda_cos(epoch, args.ent_epstart, args.ent_epend, lmax)
    use_ent = lamda_ent > 0.0

    # forward hook for features on encoder_q.layer4
    features = {}
    def save_features(module, input, output):
        features['value'] = output

    target_layer = dict(model.encoder_q.named_modules())['layer4']
    handle_f = target_layer.register_forward_hook(save_features)

    avg_losses = 0.0

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        # robust batch unpacking
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images[0] = images[0].to(args.gpu, non_blocking=True)
        images[1] = images[1].to(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        # forward
        output, target = model(im_q=images[0], im_k=images[1])
        contrastive_loss = criterion(output, target)

        local_loss = 0.0
        loss = contrastive_loss

        if use_ent:
            feats = features.get('value', None)
            if feats is not None and feats.requires_grad:
                grads = torch.autograd.grad(
                    contrastive_loss, feats,
                    retain_graph=True, create_graph=False, allow_unused=True
                )[0]
            else:
                grads = None

            if grads is not None:
                cam = compute_gradcam_from_saved(feats, grads)
                local_loss = entropy_loss(cam)
                loss = contrastive_loss + lamda_ent * local_loss

        loss.backward()
        optimizer.step()

        avg_losses += loss.item()
        losses.update(loss.item(), images[0].size(0))

        # per-batch logging of decomposed losses
        if wandb.run is not None:
            wandb.log({
                'loss_ce': float(contrastive_loss.item()),
                'loss_ent': float(local_loss) if use_ent else 0.0,
                'lambda_ent': float(lamda_ent),
                'loss_total': float(loss.item()),
                'iter': i,
                'epoch': epoch
            },step=epoch)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    handle_f.remove()
    return avg_losses / len(train_loader)

# ---------------------------
# checkpoint utils
# ---------------------------
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_dst=None):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    torch.save(state, filename)
    print(colored('-'*20, 'yellow'), colored('save checkpoint', 'yellow'), colored('-'*20, 'yellow'))
    if is_best and best_dst is not None:
        shutil.copyfile(filename, best_dst)

# ---------------------------
# meters
# ---------------------------
class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        print("\t".join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

# ---------------------------
# LR schedule
# ---------------------------
def adjust_learning_rate(optimizer, epoch, args, warmup_epochs=10):
    """Warmup + Cosine schedule"""
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:
            base_lr = args.lr_conv
        elif i == 1:
            base_lr = args.lr_fc
        
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
        
        print(colored(param_group["lr"], 'blue'))
        param_group["lr"] = lr


# ---------------------------
# CAM + entropy loss
# ---------------------------
def compute_gradcam_from_saved(features, gradients):
    alpha = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((alpha * features).sum(dim=1, keepdim=True))
    cam_min = cam.amin(dim=(2, 3), keepdim=True)
    cam_max = cam.amax(dim=(2, 3), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam.detach()

def entropy_loss(cam):
    cam = cam / (cam.sum(dim=(2,3), keepdim=True) + 1e-8)
    ent = -torch.sum(cam * torch.log(cam + 1e-8), dim=(2,3))
    return ent.mean()

def lambda_cos(epoch, e_start, e_end, lmax):
    if epoch < e_start: return 0.0
    if epoch > e_end:   return lmax
    t = (epoch - e_start) / (e_end - e_start)
    return lmax * 0.5 * (1 - math.cos(math.pi * t))

# ---------------------------
if __name__ == "__main__":
    main()
