import cv2
import random
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from termcolor import colored

class GlomerulusCopyPasteAug:
    def __init__(self, bg_paths, img_size=512, target_fade_width=10.0,prob=1.0):
        self.bg_paths = bg_paths
        self.img_size = img_size
        self.prob = prob
        self.target_fade_width = target_fade_width

    def _perturb_mask(self, mask, delta):
        if delta == 0: return mask
        kernel = np.ones((3, 3), np.uint8)
        if delta > 0:
            return cv2.dilate(mask, kernel, iterations=delta)
        else:
            return cv2.erode(mask, kernel, iterations=abs(delta))

    def _get_alpha(self, mask, fade_width):
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        alpha = np.clip(dist_map / fade_width, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        return alpha[..., None].astype(np.float32)

    def __call__(self, fg_img_np, mask_512_np):
        if random.random() > self.prob:
            img_rgb = cv2.cvtColor(fg_img_np,cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        """
        1つの前景とマスクから、Copy-Paste合成画像を1枚生成する
        """
        fg_img_rgb = cv2.cvtColor(fg_img_np, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = fg_img_rgb.shape[:2]

        # 1. BBox取得
        mask_full = cv2.resize(mask_512_np, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(mask_full)

        # 2. 余白を持たせて等倍クロップ
        margin = 20
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(W_orig, x + w + margin), min(H_orig, y + h + margin)
        
        fg_crop = fg_img_rgb[y1:y2, x1:x2].copy()
        mask_crop = mask_full[y1:y2, x1:x2].copy()

        # 3. 条件付きリサイズ (512px制限)
        h_c, w_c = fg_crop.shape[:2]
        if h_c > self.img_size or w_c > self.img_size:
            scale_to_fit = self.img_size / max(h_c, w_c)
            fg_crop = cv2.resize(fg_crop, None, fx=scale_to_fit, fy=scale_to_fit, interpolation=cv2.INTER_LINEAR)
            mask_crop = cv2.resize(mask_crop, None, fx=scale_to_fit, fy=scale_to_fit, interpolation=cv2.INTER_NEAREST)

        # 4. ランダムスケーリング (70% ~ 100%)
        scale = random.uniform(0.7, 1.0)
        fg_resized = cv2.resize(fg_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # 5. マスク摂動
        delta = random.randint(-4, 4)
        mask_perturbed = self._perturb_mask(mask_resized, delta)

        new_h, new_w = fg_resized.shape[:2]
        bg_path = random.choice(self.bg_paths)
        bg = cv2.imread(bg_path)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        
        if random.random() < 0.5: bg = cv2.flip(bg, 1)
        if random.random() < 0.5: bg = cv2.flip(bg, 0)
        k = random.randint(0, 3)
        if k > 0: bg = np.rot90(bg, k)
        
        bg = cv2.resize(bg, (self.img_size, self.img_size))
        bg = np.ascontiguousarray(bg) # メモリ配置最適化

        # 6. 背景選択と配置
        x_off = random.randint(0, self.img_size - new_w)
        y_off = random.randint(0, self.img_size - new_h)

        fg_canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        mask_canvas = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        fg_canvas[y_off:y_off+new_h, x_off:x_off+new_w] = fg_resized
        mask_canvas[y_off:y_off+new_h, x_off:x_off+new_w] = mask_perturbed

        # 7. 合成
        fg_sharp = fg_canvas.astype(np.float32)
        # エッジ専用のブラー（カーネルサイズ 5x5 は最小限の修正として適当）
        fg_blur = cv2.GaussianBlur(fg_sharp, (5, 5), 0)
        alpha = self._get_alpha(mask_canvas, self.target_fade_width)
        edge_weight = 4.0 * alpha * (1.0 - alpha)
        # 境界部のみをぼかした前景を作成
        fg_combined = (fg_sharp * (1.0 - edge_weight) + fg_blur * edge_weight)

        # 8. 背景との最終合成
        bg_float = bg.astype(np.float32)
        combined = (fg_combined * alpha + bg_float * (1.0 - alpha))
        
        # 最後に一括で uint8 にクリップして変換（精度の安定）
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # PILに変換して戻す（後続のtorchvision transformsのため）
        return Image.fromarray(combined)

class GlomerulusFixedPositionBgImageAug:
    def __init__(self, bg_paths, img_size=512, target_fade_width=10.0,prob=1.0):
        self.img_size = img_size
        self.target_fade_width = target_fade_width
        self.prob = prob
        self.bg_paths = bg_paths if bg_paths else []

    def _get_alpha(self, mask, fade_width):
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        alpha = np.clip(dist_map / fade_width, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        return alpha[..., None].astype(np.float32)

    def __call__(self, img_np, mask_512_np):
        """
        img_np: 元画像 (BGR)
        mask_512_np: マスク (512x512固定)
        """
        if random.random() > self.prob:
            img_rgb = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        # --- 前景準備 ---
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_512_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # --- 背景準備 (画像フォルダからランダムに取得して変換) ---
        if not self.bg_paths:
            bg = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            bg_path = random.choice(self.bg_paths)
            bg = cv2.imread(bg_path)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            
            # ランダム反転
            if random.random() < 0.5: bg = cv2.flip(bg, 1) # 左右
            if random.random() < 0.5: bg = cv2.flip(bg, 0) # 上下
            # ランダム回転
            k = random.randint(0, 3)
            if k > 0: bg = np.rot90(bg, k)
            
            bg = cv2.resize(bg, (self.img_size, self.img_size))
            bg = np.ascontiguousarray(bg)

        # --- エッジ限定ブラーの準備 ---
        fg_sharp = img_res.astype(np.float32)
        fg_blur = cv2.GaussianBlur(fg_sharp, (5, 5), 0)
        
        # アルファチャンネル計算
        alpha = self._get_alpha(mask_res, self.target_fade_width)
        
        # エッジ重み計算: 境界部のみを特定してぼかす
        edge_weight = 4.0 * alpha * (1.0 - alpha)
        fg_combined = (fg_sharp * (1.0 - edge_weight) + fg_blur * edge_weight)

        # --- 最終合成 ---
        bg_float = bg.astype(np.float32)
        combined = (fg_combined * alpha + bg_float * (1.0 - alpha))
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        return Image.fromarray(combined)

class GlomerulusFixedPositionRGBBgAug:
    def __init__(self, img_size=512, target_fade_width=10.0,prob=1.0):
        self.img_size = img_size
        self.target_fade_width = target_fade_width
        self.prob = prob

    def _get_alpha(self, mask, fade_width):
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        alpha = np.clip(dist_map / fade_width, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        return alpha[..., None].astype(np.float32)

    def __call__(self, img_np, mask_512_np):
        """
        img_np: 元画像 (BGR)
        mask_512_np: マスク (512x512固定)
        """
        if random.random() > self.prob:
            img_rgb = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        
        # 1. 512x512にリサイズ (位置固定のためサイズを統一)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_512_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 2. ランダムRGB背景の生成
        # 完全にランダムな単色を選択し、それに空間的なノイズを加える
        base_color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
        bg = np.full((self.img_size, self.img_size, 3), base_color, dtype=np.uint8)
        
        # 空間的ノイズ (モデルが背景の「平坦さ」を学習のヒントにしないため)
        noise = np.random.randint(-30, 31, (self.img_size, self.img_size, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 3. エッジ限定ブラーの準備
        fg_sharp = img_res.astype(np.float32)
        fg_blur = cv2.GaussianBlur(fg_sharp, (5, 5), 0)
        
        # 4. アルファチャンネル計算 (内側フェード)
        alpha = self._get_alpha(mask_res, self.target_fade_width)
        
        # エッジ重み計算: 境界部のみを特定してぼかす
        edge_weight = 4.0 * alpha * (1.0 - alpha)
        fg_combined = (fg_sharp * (1.0 - edge_weight) + fg_blur * edge_weight)

        # 5. 合成
        bg_float = bg.astype(np.float32)
        combined = (fg_combined * alpha + bg_float * (1.0 - alpha))
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        return Image.fromarray(combined)

class RandomBackgroundAugSelector:
    """RGB背景と組織画像背景をランダムに切り替えるラッパー"""
    def __init__(self, aug_rgb, aug_image, p_image=0.5,bg_prob=1.0):
        self.aug_rgb = aug_rgb
        self.aug_image = aug_image
        self.p_image = p_image
        self.bg_prob=bg_prob

    def __call__(self, img_np, mask_512_np):
        # 50%の確率で組織画像、残りの50%でRGB背景を選択
        if random.random() > self.bg_prob:
            # Augmentationをスキップする場合、BGR->RGB変換し、PIL Image化して返す
            # (後続のtransformsがPILを期待しているため)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        if random.random() < self.p_image:
            return self.aug_image(img_np, mask_512_np)
        else:
            return self.aug_rgb(img_np, mask_512_np)

class GlomerulusMoCoDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, cp_aug, post_transform=None):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.cp_aug = cp_aug
        self.post_transform = post_transform

        # 画像とマスクのペアを構築
        self.pairs = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        all_imgs = [p for p in self.img_root.rglob("*") if p.suffix.lower() in valid_exts]
        
        for img_p in all_imgs:
            # マスクのパスを推定 (例: masks/フォルダ名/ファイル名.png)
            mask_p = self.mask_root / img_p.parent.name / (img_p.stem + ".png")
            if mask_p.exists():
                self.pairs.append((str(img_p), str(mask_p)))
        
        print(f"Dataset initialized: {len(self.pairs)} glomerulus pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_p, mask_p = self.pairs[index]
        fg_img = cv2.imread(img_p)
        mask_512 = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        # MoCo用に2つの異なる合成ビューを作成
        view1 = self.cp_aug(fg_img, mask_512)
        view2 = self.cp_aug(fg_img, mask_512)

        if self.post_transform:
            view1 = self.post_transform(view1)
            view2 = self.post_transform(view2)

        return [view1, view2], 0 # 0はダミーラベル

class GlomerulusFixedDataset(torch.utils.data.Dataset):
    """固定位置版 (rgb_prob, wsi_pb, rgb_wsi_prob) のためのデータセット"""
    def __init__(self, img_root, mask_root, aug_logic, transform=None):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.aug_logic = aug_logic
        self.transform = transform
        
        self.pairs = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        all_imgs = [p for p in self.img_root.rglob("*") if p.suffix.lower() in valid_exts]
        
        for img_p in all_imgs:
            sample_folder_name = img_p.parent.name 
            mask_p = self.mask_root / sample_folder_name / (img_p.stem + ".png")
            if mask_p.exists():
                self.pairs.append((str(img_p), str(mask_p)))
                
        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {img_root}")
        print(colored(f"Dataset loaded: {len(self.pairs)} pairs found.", "green"))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        view1 = self.aug_logic(img, mask)
        view2 = self.aug_logic(img, mask)

        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)
        return [view1, view2], 0

    