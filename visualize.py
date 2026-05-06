#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
import gc
import os
import sys
import random
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, models

from tqdm import tqdm
from termcolor import colored
from pytorch_grad_cam import GradCAM
import seaborn as sns

# -----------------------------
# 引数
# -----------------------------
parser = argparse.ArgumentParser(description='Robust Cluster-prototype-driven Grad-CAM')
parser.add_argument('--exp', default='0', type=str)
parser.add_argument('--epoch', default=29, type=int, help='Default epoch. Will be overwritten if best_epoch_info.pkl exists.')
parser.add_argument('--out-path', default='result_sample', type=str)
parser.add_argument('--dataset-path', default='./dataset_1203_org/rat_PAS_crop_longside_1.3', type=str, help='Dataset root directory')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--model_name', default='resnet50', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--max_proto_images', default=0, type=int, help='重心計算に使う最大枚数 (0=全件)')
parser.add_argument('--trim_ratio', default=0.1, type=float, help='外れ値除去率 [0, 0.5)')
parser.add_argument('--top_k_visualize', default=100, type=int, help='各クラスタで可視化する枚数')
parser.add_argument('--target-clusters', default='top_bottom', type=str, 
                    help="'all': 全クラスタ, 'top10': 係数上位10のみ, 'top_bottom': 上位2+下位2")
parser.add_argument('--skip-existing', action='store_true')

# -----------------------------
# Utility
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cudnn.deterministic = True; cudnn.benchmark = False

def build_transform(img_size: int):
    tfm = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    tfm_vis = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    return tfm, tfm_vis

def build_model(name: str):
    if name.lower() != 'resnet50':
        raise ValueError('resnet50 のみ対応')
    model = models.resnet50(weights=None)
    model.fc = nn.Identity()
    return model

def load_checkpoint(model: nn.Module, exp_path: str, epoch: int, device: torch.device):
    candidates = [
        os.path.join(exp_path, 'checkpoint', f'checkpoint_{epoch+1:04d}.pth.tar'),
        os.path.join(exp_path, 'checkpoint', f'checkpoint_00{str(epoch).zfill(2)}.pth.tar'),
        os.path.join(exp_path, 'checkpoint', f'checkpoint_{epoch:04d}.pth.tar')
    ]
    resume_path = None
    for p in candidates:
        if os.path.isfile(p):
            resume_path = p
            break
    
    if not resume_path:
        raise FileNotFoundError(f"Checkpoint not found. Searched: {candidates}")

    print(f"=> loading checkpoint '{resume_path}'", flush=True)
    ckpt = torch.load(resume_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt)
    
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        for p in ['module.encoder_q.', 'encoder_q.', 'module.']:
            if new_k.startswith(p):
                new_k = new_k.replace(p, '')
        new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"=> loaded. missing={len(missing)}", flush=True)

@torch.inference_mode()
def extract_feature(model, device, img_pth, tfm):
    """
    1枚の画像から特徴量を抽出する関数
    """
    try:
        with Image.open(img_pth) as im:
            im = im.convert('RGB')
            # GPUへ転送
            x = tfm(im).unsqueeze(0).to(device, non_blocking=True)
            f = model(x)
        
        # すぐにCPUへ戻し、GPUメモリを解放する
        f_cpu = F.normalize(f, dim=-1).cpu() 
        
        # 明示的に削除
        del x, f
        return f_cpu
    except Exception as e:
        print(f"Error reading {img_pth}: {e}", flush=True)
        return None

# クラスタごとにプロトタイプベクトル(pv)計算
@torch.inference_mode()
def compute_cluster_prototype(model, device, img_paths, tfm, max_imgs, seed, trim_ratio):
    if len(img_paths) == 0: return None
    paths = img_paths
    
    # max_imgsでランダムサンプリング (重心計算用)
    if 0 < max_imgs < len(img_paths):
        rng = np.random.default_rng(seed)
        paths = list(rng.choice(img_paths, size=max_imgs, replace=False))
    
    # 特徴抽出
    feats = []
    valid_paths = []
    
    # プロトタイプ計算時はそこまで枚数が多くない想定だが、念のためバッチ処理風にする
    iterator = tqdm(paths, desc="Extracting features (Proto)", leave=False)
    
    for i, p in enumerate(iterator):
        f = extract_feature(model, device, p, tfm)
        if f is not None:
            feats.append(f)
            valid_paths.append(p)
        
        # 100枚ごとにガベージコレクション
        if (i + 1) % 100 == 0:
            gc.collect()

    if len(feats) == 0: return None
    
    # ここでスタックするが一瞬メモリを食う
    Fstack = torch.cat(feats, dim=0) # [N,D]

    # 仮の中心
    center = F.normalize(Fstack.mean(dim=0, keepdim=True), dim=1)
    cos_sim = (Fstack @ center.t()).squeeze(1)
    
    # Trim処理
    trim_ratio = float(np.clip(trim_ratio, 0.0, 0.49))
    k = max(1, int(len(cos_sim) * (1.0 - trim_ratio)))
    keep_idx = torch.topk(cos_sim, k=k, largest=True).indices
    
    trimmed = Fstack[keep_idx]
    proto = F.normalize(trimmed.mean(dim=0, keepdim=True), dim=1)
    
    # クリーンアップ
    del Fstack, center, cos_sim, keep_idx, trimmed, feats
    gc.collect()
    torch.cuda.empty_cache()
    
    return proto, valid_paths

class ProtoDotTarget:
    def __init__(self, proto_vec: torch.Tensor): self.u = proto_vec
    def __call__(self, model_out: torch.Tensor):
        if model_out.dim() == 1: model_out = model_out.unsqueeze(0)
        u = self.u if self.u.dim() == 2 else self.u.unsqueeze(0)
        u = F.normalize(u.to(model_out.device), dim=-1)
        f = F.normalize(model_out, dim=-1)
        return (f * u).sum(dim=1)

def save_gradcam_comparison(model, cam, device, u, img_pth, tfm, tfm_vis, save_pth):
    im = Image.open(img_pth).convert('RGB')
    x = tfm(im).unsqueeze(0).to(device)
    x_vis = tfm_vis(im).cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    
    # 1. Proto-GradCAM
    target_proto = ProtoDotTarget(u.to(device))
    grayscale_cam_proto = cam(input_tensor=x, targets=[target_proto])[0]
    
    # 2. Standard Grad-CAM
    grayscale_cam_std = cam(input_tensor=x, targets=None)[0]
    
    def process_cam(grayscale_cam, vis_img):
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        cam_image = heatmap * 0.5 + vis_img * 0.5
        return cam_image / np.max(cam_image)

    vis_proto = process_cam(grayscale_cam_proto, x_vis)
    vis_std = process_cam(grayscale_cam_std, x_vis)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(vis_proto)
    ax[0].set_title(f'Centroid-Guided\n(Cluster Feature)')
    ax[0].axis('off')

    ax[1].imshow(vis_std)
    ax[1].set_title('Standard Grad-CAM\n(Dominant Feature)')
    ax[1].axis('off')
    
    ax[2].imshow(x_vis)
    ax[2].set_title('Original')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_pth, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 個別の可視化関数内でもメモリ解放
    del x, x_vis, grayscale_cam_proto, grayscale_cam_std, vis_proto, vis_std
    gc.collect()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parser.parse_args()
    seed_everything(args.seed)

    # Best Epoch読み込み
    best_epoch_info_path = os.path.join(args.out_path, f'exp{args.exp}', 'best_epoch_info.pkl')
    if os.path.isfile(best_epoch_info_path):
        print(colored(f"=> Found best_epoch_info at {best_epoch_info_path}", "cyan"))
        try:
            with open(best_epoch_info_path, 'rb') as f:
                info = pickle.load(f)
            best_epoch = info.get('best_epoch')
            if best_epoch is not None:
                print(colored(f"=> Automatically switching epoch from {args.epoch} to Best Epoch: {best_epoch}", "green"))
                args.epoch = best_epoch
        except Exception as e:
            print(colored(f"Warning: Failed to load best_epoch_info: {e}", "yellow"))
    else:
        print(colored(f"=> best_epoch_info not found. Using specified epoch: {args.epoch}", "yellow"))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(colored(f'Using device: {device}', 'cyan'), flush=True)

    # パス設定
    exp_root = os.path.join(args.out_path, f'exp{args.exp}')
    epoch_root = os.path.join(exp_root, f'epoch{args.epoch}')
    
    if not os.path.exists(args.dataset_path):
        print(colored(f"Error: Dataset path does not exist: {args.dataset_path}", "red"))
        return

    # CSV読み込み
    csv_path = os.path.join(epoch_root, f'result_epoch{args.epoch}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Result CSV not found: {csv_path}")
    
    print(f"=> Loading clustering result: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    
    # ターゲットクラスタ設定
    if args.target_clusters == 'all':
        target_cluster_ids = sorted(df['class'].unique())
        print(f"=> Target: ALL clusters ({len(target_cluster_ids)})", flush=True)
    elif args.target_clusters == 'top10':
        top10_csv = os.path.join(epoch_root, f'top_10_clusters_epoch{args.epoch}.csv')
        if os.path.exists(top10_csv):
            df_top10 = pd.read_csv(top10_csv)
            target_cluster_ids = sorted(df_top10['Cluster_Index'].unique())
            print(colored(f"=> Target: Top 10 Clusters {target_cluster_ids}", "green"), flush=True)
        else:
            print(colored(f"Warning: {top10_csv} not found. Fallback to ALL clusters.", "yellow"), flush=True)
            target_cluster_ids = sorted(df['class'].unique())
    elif args.target_clusters == 'top_bottom':
        tb_csv = os.path.join(epoch_root, f'top_bottom_clusters_epoch{args.epoch}.csv')
        if os.path.exists(tb_csv):
            df_tb = pd.read_csv(tb_csv)
            target_cluster_ids = sorted(df_tb['Cluster_Index'].unique())
            print(colored(f"=> Target: Top2 & Bottom2 Clusters {target_cluster_ids}", "green"), flush=True)
        else:
            print(colored(f"Warning: {tb_csv} not found. Fallback to ALL clusters.", "yellow"), flush=True)
            target_cluster_ids = sorted(df['class'].unique())
    else:
        try:
            target_cluster_ids = [int(x) for x in args.target_clusters.split(',')]
            print(f"=> Target: Specified clusters {target_cluster_ids}", flush=True)
        except:
            print(colored("Error: Invalid target-clusters argument.", "red"), flush=True)
            return

    # モデル準備
    tfm, tfm_vis = build_transform(args.image_size)
    model = build_model(args.model_name)
    load_checkpoint(model, exp_root, args.epoch, device)
    model.to(device).eval()
    
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers) 

    out_root = os.path.join(epoch_root, f'gradcam_proto_trim_{args.trim_ratio}')
    os.makedirs(out_root, exist_ok=True)

    # インデックス作成
    print(colored(f"=> Building image path index from {args.dataset_path}...", "cyan"), flush=True)
    image_path_map = {}
    file_count = 0
    for root, _, files in os.walk(args.dataset_path):
        for f in files:
            image_path_map[f] = os.path.join(root, f)
            file_count += 1
            if file_count % 5000 == 0:
                print(f"   scanned {file_count} files...", end='\r', flush=True)
    print(f"\n=> Index built. Found {file_count} images.", flush=True)

    u_list = []
    cid_list = []

    # --- クラスタ処理ループ ---
    for cid in target_cluster_ids:
        save_dir = os.path.join(out_root, str(cid))
        done_marker = os.path.join(save_dir, "done.txt")
        
        # 復旧スキップ
        if args.skip_existing and os.path.exists(done_marker):
            print(colored(f"Skipping Cluster {cid} (Already processed: found done.txt)", "yellow"), flush=True)
            continue

        cluster_df = df[df['class'] == cid]
        img_names = cluster_df['image_name'].tolist()
        
        if len(img_names) == 0: continue

        full_paths = []
        missing_count = 0
        for img_name in img_names:
            if img_name in image_path_map:
                full_paths.append(image_path_map[img_name])
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(colored(f"Warning: {missing_count} images missing for cluster {cid}", "yellow"), flush=True)
        if len(full_paths) == 0:
            continue

        print(colored(f"Processing Cluster {cid} ({len(full_paths)} images)...", "green"), flush=True)

        # 1. プロトタイプ計算
        # ここでは max_proto_images で重心計算に使う枚数を制限してもよいが、
        # スコアリング（比較）は全画像に対して行う
        proto, valid_paths = compute_cluster_prototype(
            model, device, full_paths, tfm, 
            args.max_proto_images, args.seed, args.trim_ratio
        )
        if proto is None: continue
        
        u_list.append(proto.cpu().numpy().flatten())
        cid_list.append(cid)
        
        # 2. 全画像のスコアリング（バッチ分割処理）
        # 全画像リストを作成
        score_candidates = valid_paths # ここはランダムサンプリングせず全件
        
        u_gpu = proto.to(device)
        path_score_pairs = []
        
        print(colored(f"   -> Scoring all {len(score_candidates)} images in chunks...", "cyan"))

        # ▼▼▼【重要修正】全件をチャンク分割して処理し、都度メモリ解放する ▼▼▼
        chunk_size = 100 # 100枚ずつ処理
        
        # 進行状況バー
        pbar = tqdm(total=len(score_candidates), desc=f"Scoring Cluster {cid}", leave=False)
        
        # チャンクループ
        for i in range(0, len(score_candidates), chunk_size):
            chunk_paths = score_candidates[i : i + chunk_size]
            
            # チャンク内の画像を処理
            for p in chunk_paths:
                f = extract_feature(model, device, p, tfm)
                if f is not None:
                    # スコア計算 (GPUで行い、結果をすぐCPUのfloatへ)
                    f = f.to(device)
                    score = (f * u_gpu).sum().item() 
                    path_score_pairs.append((p, score))
                    
                    # 不要になったTensorを削除
                    del f
            
            pbar.update(len(chunk_paths))
            
            # ★チャンクごとの強制掃除★
            # これが重要です。OSにメモリを返却させます。
            gc.collect()
            torch.cuda.empty_cache()
            
        pbar.close()
        # ▲▲▲ ここまで ▲▲▲
        
        # 全件の計算が終わったらソート
        path_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_k = path_score_pairs[:args.top_k_visualize]
        
        # 可視化実行
        os.makedirs(save_dir, exist_ok=True)
        for i, (p, score) in enumerate(top_k):
            try:
                base = os.path.basename(p)
                save_name = os.path.join(save_dir, f"{score:.3f}_{base}")
                save_gradcam_comparison(model, cam, device, u_gpu, p, tfm, tfm_vis, save_name)
                
                # 可視化中も定期的に掃除 (20枚ごと)
                if i % 20 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Viz failed for {p}: {e}", flush=True)
                
        with open(done_marker, 'w') as f:
            f.write("done")
            
        # 次のクラスタへの切り替え前に大掃除
        del proto, u_gpu, path_score_pairs, top_k, valid_paths, full_paths, score_candidates
        gc.collect()
        torch.cuda.empty_cache()
        print(colored(f"Cluster {cid} Done. Memory cleaned.", "cyan"), flush=True)

    # 角度計算
    if len(u_list) > 1:
        U = np.stack(u_list)
        cos_sim = np.clip(U @ U.T, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_sim))
        
        df_ang = pd.DataFrame(angles, index=cid_list, columns=cid_list)
        df_ang.to_csv(os.path.join(epoch_root, f"prototype_angles.csv"))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_ang, cmap='viridis')
        plt.title('Prototype Angle Matrix')
        plt.savefig(os.path.join(epoch_root, f"prototype_angles.png"))
        plt.close()

    print(colored("All Done.", "cyan"), flush=True)

if __name__ == "__main__":
    main()