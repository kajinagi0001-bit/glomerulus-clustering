import argparse
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import umap
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# サーバー環境でのプロット用設定（GUIなし対応）
plt.switch_backend('Agg')

from tqdm import tqdm
from termcolor import colored
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
# プロジェクト内ツールの読み込み
try:
    import tools.loader
    import tools.folder
except ImportError:
    print(colored("Warning: tools package not found. Adjust imports as necessary.", "yellow"))

parser = argparse.ArgumentParser(description='Evaluation with GMM and Logistic Regression')
parser.add_argument('--exp', default='0', type=str, help='number of experiment')
parser.add_argument('--n-comp', default=24, type=int, help='number of UMAP components')
parser.add_argument('--out-path', default='result_sample', type=str, help='out path')
parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint")
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--allepoch', default=20, type=int, help='number of epoch')
parser.add_argument('--data-dir',type=str,default='./dataset_1203_org/rat_PAS_crop_longside_1.3')

def main():
    args = parser.parse_args()
    print(colored(str(args), 'cyan'))

    if args.seed is not None:
        seed_worker(args)

    if torch.cuda.is_available():
        args.gpu_device = torch.device(f'cuda:{args.gpu}')
        print(colored('device_name ----> ', 'white'), colored(args.gpu_device, 'white'))
    else:
        print(colored('Can not use GPU. Exiting.', 'red'))
        sys.exit(1)

    print(colored('=> Now start construction', 'cyan'))
    
    # 各エポックの評価結果を保存するリスト
    epoch_results = []
    
    # ラスト11エポックを確認
    start_epoch = max(0, args.allepoch - 11)
    
    for a in range(start_epoch, args.allepoch, 1):
        try:
            result = main_worker(args.gpu_device, args, a)
            if result:
                epoch_results.append(result)
        except Exception as e:
            print(colored(f"Error evaluating epoch {a}: {e}", "red"))
            import traceback
            traceback.print_exc()

    if not epoch_results:
        print("No results collected.")
        return

    # 最適エポックを選択
    best_epoch = select_best_epoch(epoch_results, args)
    print(colored(f'=> Best epoch selected: {best_epoch}', 'green'))
    
    # 最適エポックの詳細情報を取得
    best_epoch_result = next((r for r in epoch_results if r['epoch'] == best_epoch), None)
    
    # 最適エポックの情報をファイルに保存
    best_epoch_info = {
        'best_epoch': best_epoch,
        'best_num_clusters': best_epoch_result['num_clusters'] if best_epoch_result else None,
        'epoch_results': epoch_results
    }
    
    exp_dir = os.path.join(args.out_path, f'exp{args.exp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, 'best_epoch_info.pkl'), 'wb') as f:
        pickle.dump(best_epoch_info, f)

def main_worker(device, args, epoch_idx):
    # モデル構築
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Identity()

    # チェックポイントパスの構築
    ckpt_path = os.path.join(args.out_path, f'exp{args.exp}', 'checkpoint', f'checkpoint_{epoch_idx+1:04d}.pth.tar')
    
    # ファイル名フォーマットが異なる場合のフォールバック（元のコードのフォーマット）
    if not os.path.isfile(ckpt_path):
         ckpt_path = os.path.join(args.out_path, f'exp{args.exp}', 'checkpoint', f'checkpoint_00{str(epoch_idx).zfill(2)}.pth.tar')

    if os.path.isfile(ckpt_path):
        print(f"=> loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            # encoder_q. prefixの除去
            if k.startswith('encoder_q.'):
                new_k = k.replace('encoder_q.', '')
                new_state_dict[new_k] = v
            elif k.startswith('module.encoder_q.'):
                new_k = k.replace('module.encoder_q.', '')
                new_state_dict[new_k] = v
            else:
                 new_state_dict[k] = v
                 
        model.load_state_dict(new_state_dict, strict=False)
        print(f"=> loaded checkpoint '{ckpt_path}' (epoch {checkpoint.get('epoch', epoch_idx)})")
    else:
        print(f"=> no checkpoint found at '{ckpt_path}'")
        return None

    torch.cuda.set_device(args.gpu)
    model = model.to(device)
    model.eval()

    # データセット設定
    test_dir = args.data_dir
    if not os.path.exists(test_dir):
        print(colored(f"Dataset dir {test_dir} not found.", "red"))
        return None

    test_aug = [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]

    try:
        test_dataset = tools.folder.NestedImageDataset(
            test_dir, transforms.Compose(test_aug)
        )
        print('NestedImageDataset')
    except AttributeError:
         test_dataset = ImageFolder(test_dir, transforms.Compose(test_aug))
         print('ImageFolder')

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=seed_worker(args),
        generator=generator
    )
    
    columns = ['patient', 'image_name', 'type', 'class']
    result = pd.DataFrame(columns=columns)
    encoder_feature = []
    images_path_list = []

    print(colored(f'=> start feature extraction (Epoch {epoch_idx})', 'cyan'))

    for i, batch in enumerate(tqdm(test_loader, desc='Processing', colour='yellow')):
        # datasetの実装により戻り値が異なる場合の対応
        if len(batch) == 3:
            image, target, images_path = batch
        else:
            image, target = batch
            # パスが取得できない場合はダミーを使用するか、Datasetの実装を確認してください
            # ここではDatasetがpathを返すと仮定
            images_path = [f"dummy_{i}_{j}.png" for j in range(len(image))]
            print(images_path)

        clustering_image = image.to(device)
        with torch.no_grad():
            features = model(clustering_image)

        features = features.cpu().numpy() # batchごとnumpyへ
        
        for j in range(features.shape[0]):
            encoder_feature.append(features[j])
            images_path_list.append(images_path[j])

    # パス作成
    path_dir = os.path.join(args.out_path, f'exp{args.exp}', 'path')
    os.makedirs(path_dir, exist_ok=True)

    encoder_feature = np.array(encoder_feature).astype(np.float64)
    images_path_list = np.array(images_path_list)
    
    # 保存
    np.save(os.path.join(path_dir, f'encoder_feature_epoch{epoch_idx}.npy'), encoder_feature)
    np.save(os.path.join(path_dir, 'images_path_list.npy'), images_path_list)
    
    print(colored('=> start dimensionality reduction', 'cyan'))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # サンプル数が少ない場合の安全策
        n_neighbors = min(30, len(encoder_feature) - 1)
        umap_reducer = umap.UMAP(n_components=args.n_comp, n_neighbors=n_neighbors, min_dist=0.1, metric='euclidean', random_state=args.seed)
        reduced_features = umap_reducer.fit_transform(encoder_feature).astype(np.float64)

        # UMAPモデルの保存
        with open(os.path.join(path_dir, f'umap_model_epoch{epoch_idx}.pkl'), 'wb') as f:
            pickle.dump(umap_reducer, f)
        
        print(colored("Dimension reduction completed. Shape:", 'white'), reduced_features.shape)

        np.save(os.path.join(path_dir, f'reduced_features_{args.n_comp}d_epoch{epoch_idx}.npy'), reduced_features)

    # GMM BIC Selection
    n_components_range = range(1, min(30, len(reduced_features)))
    BIC = []
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=args.seed, reg_covar=1e-3)
        gmm.fit(reduced_features)
        BIC.append(gmm.bic(reduced_features))

    best_BIC_idx = np.argmin(BIC)
    best_BIC_clusters = n_components_range[best_BIC_idx]
    print(f'Best number of clusters (BIC): {best_BIC_clusters}')

    gmm = GaussianMixture(n_components=best_BIC_clusters, random_state=args.seed)
    gmm.fit(reduced_features)
    with open(os.path.join(path_dir, f'gmm_model_epoch{epoch_idx}.pkl'), 'wb') as f:
        pickle.dump(gmm, f)
    labels = gmm.predict(reduced_features)
    
    # 結果の集計
    result_list = []
    for i in range(len(images_path_list)):
        out_id = labels[i]
        basename = os.path.basename(images_path_list[i])
        # ディレクトリ構造依存: root/diabetes/human_id/image.png を想定
        dirname = os.path.dirname(images_path_list[i])
        human_id = os.path.basename(dirname)
        diabetes = os.path.basename(os.path.dirname(dirname))

        result_list.append({
            'patient': human_id,
            'image_name': basename,
            'type': diabetes,
            'class': int(out_id)
        })

    result = pd.DataFrame(result_list)
    
    epoch_dir = os.path.join(args.out_path, f'exp{args.exp}', f'epoch{epoch_idx}')
    os.makedirs(epoch_dir, exist_ok=True)
    result.to_csv(os.path.join(epoch_dir, f'result_epoch{epoch_idx}.csv'), index=False)

    # 追加: 各クラスタの代表ベクトルを確率分布として可視化
    plot_cluster_center_distributions(encoder_feature, labels, epoch_dir, epoch_idx, feature_name="original")
    plot_cluster_center_distributions(reduced_features, labels, epoch_dir, epoch_idx, feature_name="reduced")
    # クラスごとの集計
    out_class = np.sort(result['class'].unique())
    
    # Pivot tableの方が高速・安全
    pivot_df = result.pivot_table(index=['patient', 'type'], columns='class', values='image_name', aggfunc='count', fill_value=0)
    # 正規化（行ごとの合計で割る）
    pivot_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0).reset_index()
    
    # 列名の整理
    pivot_norm.columns = [str(c) if isinstance(c, int) else c for c in pivot_norm.columns]
    
    # typeを0/1に変換
    pivot_norm['type_label'] = pivot_norm['type'].apply(lambda x: 1 if 'diabetes' in str(x) and 'not' not in str(x) else 0)
    
    # typeカラムを文字列として統一（not_diabetes/diabetes）
    pivot_norm['type_str'] = pivot_norm['type_label'].apply(lambda x: 'diabetes' if x==1 else 'not_diabetes')
    
    # 保存用に整理
    count_result = pivot_norm.drop(columns=['type']).rename(columns={'patient': 'name', 'type_str': 'type', 'type_label': 'type_bin'})
    # type_bin (0/1) を type として使うか、文字列を使うかは元のコードに合わせる
    # 元コードは type=0/1 にしているので合わせる
    count_result['type'] = count_result['type_bin']
    count_result = count_result.drop(columns=['type_bin'])
    
    count_result.to_csv(os.path.join(epoch_dir, f'count_result_epoch{epoch_idx}.csv'), index=False)

    # --- main_worker内の「ロジスティック回帰」セクションを以下に差し替え ---

    # ロジスティック回帰
    print(colored('=> start Logistic Regression.', 'cyan'))
    class_cols = [str(c) for c in out_class]
    X = count_result[class_cols]
    Y = count_result["type"]

    if len(np.unique(Y)) < 2:
        print(colored("Error: Only one class present in Y.", "red"))
        return None

    # --- 追加: analysis_epoch.csv の作成 (上のプログラムの再現) ---
    analysis_tmp = []
    for cls in out_class:
        label = str(cls)
        c1 = ((result['type'] == 'diabetes') & (result['class'] == cls)).sum()
        c0 = ((result['type'] == 'not_diabetes') & (result['class'] == cls)).sum()
        analysis_tmp.append([label, c1, c0, c1 + c0])
    df_sum = pd.DataFrame(analysis_tmp, columns=['labels', 'diabetes', 'not_diabetes', 'diabetes + not_diabetes'])
    df_sum.to_csv(os.path.join(epoch_dir, f'analysis_epoch{epoch_idx}.csv'), index=False)

    # ロジスティック回帰の実行
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    test_acc_all = []
    all_acc_all = [] # 追加: 全データ精度用
    all_coefficients = []

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        lr = LogisticRegression(fit_intercept=True, random_state=0, max_iter=1000)
        lr.fit(X_train, Y_train)
        threshold = Y_train.mean()

        # テスト精度の計算
        preds_test = (lr.predict_proba(X_test)[:, 1] > threshold).astype(int)
        test_acc_all.append(accuracy_score(Y_test, preds_test))

        # 全データ精度の計算 (上のプログラムの再現)
        preds_all = (lr.predict_proba(X)[:, 1] > threshold).astype(int)
        all_acc_all.append(accuracy_score(Y, preds_all))

        all_coefficients.append(lr.coef_[0])

    # 係数関連のCSV保存 (上のプログラムの再現)
    mean_coefficients = np.mean(all_coefficients, axis=0)
    indices = np.argsort(mean_coefficients)[::-1]
    
    # coefficients_epoch.csv
    pd.DataFrame({'Cluster_Index': indices, 'Coefficient_Value': mean_coefficients[indices]}).to_csv(
        os.path.join(epoch_dir, f'coefficients_epoch{epoch_idx}.csv'), index=False)
    
    # top_10_clusters.csv
    top10_idx = indices[:10]
    pd.DataFrame({'Cluster_Index': top10_idx, 'Coefficient_Value': mean_coefficients[top10_idx]}).to_csv(
        os.path.join(epoch_dir, f'top_10_clusters_epoch{epoch_idx}.csv'), index=False)

    # top_bottom_clusters.csv
    combined_idx = np.concatenate([indices[:2], indices[-2:]])
    pd.DataFrame({'Cluster_Index': combined_idx, 'Coefficient_Value': mean_coefficients[combined_idx]}).to_csv(
        os.path.join(epoch_dir, f'top_bottom_clusters_epoch{epoch_idx}.csv'), index=False)

    # --- 追加: Age-wise 解析 (上のプログラムの再現) ---
    try:
        pas_csv_path = os.path.join('dataset', 'pas_sample_index.csv') # パスは環境に合わせて調整
        if os.path.exists(pas_csv_path):
            pas_df = pd.read_csv(pas_csv_path)
            pas_df['sample_number'] = pas_df['sample_number'].astype(str)
            merged = pd.merge(count_result, pas_df[['sample_number', 'age']], left_on='name', right_on='sample_number', how='left')
            
            valid_ages, age_coef_rows = [], []
            for age_val in sorted(merged['age'].dropna().unique()):
                sub = merged[merged['age'] == age_val]
                if sub['type'].nunique() >= 2:
                    lr_age = LogisticRegression(fit_intercept=True, random_state=0)
                    lr_age.fit(sub[class_cols], sub['type'])
                    age_coef_rows.append(lr_age.coef_[0])
                    valid_ages.append(age_val)
            
            if valid_ages:
                heat_df = pd.DataFrame(age_coef_rows, index=valid_ages, columns=class_cols)
                heat_df.to_csv(os.path.join(epoch_dir, f'age_cluster_coefficients_epoch{epoch_idx}.csv'))
                plt.figure(figsize=(10, 8))
                sns.heatmap(heat_df, cmap='coolwarm', center=0)
                plt.savefig(os.path.join(epoch_dir, f'age_cluster_coefficients_epoch{epoch_idx}.png'))
                plt.close()
    except Exception as e:
        print(f"Age-wise analysis failed: {e}")

    # 結果辞書の更新
    logistic_results = {
        'epoch': epoch_idx,
        'test_accuracy': np.mean(test_acc_all),
        'test_accuracy_std': np.std(test_acc_all),
        'all_accuracy': np.mean(all_acc_all), # 追加
        'mean_coefficients': mean_coefficients,
        'coefficient_sum_abs': np.sum(np.abs(mean_coefficients)),
        'num_clusters': len(out_class)
    }

    # 可視化（UMAP）
    # ▼▼▼【変更箇所】引数を追加して呼び出し（カラーマップ処理のため） ▼▼▼
    generate_visualizations(
        reduced_features, 
        labels, 
        top10_idx, 
        combined_idx, # 追加
        epoch_dir, 
        epoch_idx, 
        args.n_comp, 
        args.seed,
        images_path_list, # 追加: 画像パスリスト
        result # 追加: 結果DataFrame（クラス参照用）
    )
    
    return logistic_results

def generate_visualizations(features, labels, top10_idx, combined_top_bottom, out_dir, epoch, n_comp, seed, images_path_list, result_df):
    """
    2つ目のプログラムのロジックに基づいて詳細な可視化を行う
    """
    print(colored("start of Umap visualization with custom colormap", 'cyan'))

    # 1. カラーマップの定義 (2つ目のプログラムより)
    color_map = [
    # 暖色系 (赤から黄色まで)
    '#FF0000',    # 鮮やかな赤
    '#FF4500',    # オレンジレッド
    '#FF6347',    # トマト色
    '#FF8C00',    # ダークオレンジ
    '#FFA500',    # オレンジ
    '#FFD700',    # ゴールド
    '#FFFF00',    # 黄色
    '#FFA07A',    # ライトサーモン
    
    # 寒色系 (緑から青)
    '#00FF00',    # 緑
    '#00FF7F',    # スプリンググリーン
    '#00CED1',    # ダークターコイズ
    '#00BFFF',    # ディープスカイブルー
    '#1E90FF',    # ドッジャーブルー
    '#0000FF',    # 青
    '#4169E1',    # ロイヤルブルー
    '#87CEEB',    # スカイブルー
    
    # 紫系
    '#8A2BE2',    # ブルーバイオレット
    '#9400D3',    # ダークバイオレット
    '#BA55D3',    # 中程度の蘭
    '#FF00FF',    # マゼンタ
    '#FF1493',    # ディープピンク
    '#DA70D6',    # オーキッド
    
    # アース系 (茶色系)
    '#8B4513',    # 鞍茶色
    '#A0522D',    # シエナ
    '#CD853F',    # ペルー
    '#D2691E',    # チョコレート
    '#B8860B',    # ダークゴールドロッド
    
    # グレースケール
    '#000000',    # 黒
    '#808080',    # グレー
    '#FFFFFF'     # 白
    ]

    # クラスタ番号と色のマッピング情報を保存
    unique_cluster_ids = np.unique(labels)
    cluster_color_mapping = []
    for cls_id in sorted(unique_cluster_ids):
        color = color_map[int(cls_id) % len(color_map)]
        cluster_color_mapping.append({'cluster_id': int(cls_id), 'color_hex': color})
    
    mapping_df = pd.DataFrame(cluster_color_mapping)
    mapping_df.to_csv(os.path.join(out_dir, f'cluster_color_mapping_epoch{epoch}.csv'), index=False)
    print(colored(f'Cluster color mapping saved to cluster_color_mapping_epoch{epoch}.csv', 'green'))

    # 2. 各データポイントへの色割り当て
    color_list_top10 = []
    color_list_top_bottom = []
    color_list_reduce = []
    
    # 処理の高速化のためにDictionary化
    # 画像名 -> クラスID のマッピングを作成
    img_to_class = dict(zip(result_df['image_name'], result_df['class']))

    # カラーリストの作成
    # 2つ目のプログラムのロジックを再現（画像パス順に色を格納）
    for b_p in images_path_list:
        image_name = os.path.basename(b_p)
        
        # クラスIDを取得
        cls_id = img_to_class.get(image_name)
        if cls_id is None:
            continue # 万が一見つからない場合

        # カラーマップから色を取得（クラスIDがcolor_mapの長さを超える場合は循環させる安全策を追加）
        color = color_map[cls_id % len(color_map)]
        
        # 全データ用リストに追加
        color_list_reduce.append(color)
        
        # Top 10用
        if cls_id in top10_idx:
            color_list_top10.append(color)
            
        # Top/Bottom用
        if cls_id in combined_top_bottom:
            color_list_top_bottom.append(color)

    # 3. UMAP可視化 (2次元へ圧縮してプロット)

    # --- (A) Top 10 Clusters ---
    mask_top10 = np.isin(labels, top10_idx)
    if mask_top10.sum() > 0:
        feat_top10 = features[mask_top10]
        labels_top10 = labels[mask_top10]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # 2次元への圧縮を再度実行
            umap_reducer_3 = umap.UMAP(n_components=2, n_neighbors=100, min_dist=1.0, metric='euclidean', random_state=seed)
            embedding_top10 = umap_reducer_3.fit_transform(feat_top10)

        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding_top10[:, 0], embedding_top10[:, 1], c=color_list_top10, s=10)
        
        # # 凡例の作成（Top 10クラスタのみ）
        # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
        #                              markerfacecolor=color_map[int(cls_id) % len(color_map)], 
        #                              markersize=8, label=f'Cluster {int(cls_id)}')
        #                   for cls_id in sorted(top10_idx)]
        # plt.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)
        
        plt.grid()
        plt.title(f'Umap Visualization top10 2048d_umap{n_comp}d_epoch{epoch}\n(See cluster_color_mapping_epoch{epoch}.csv for full mapping)')
        plt.savefig(os.path.join(out_dir, f'umap_visualization_top10_umap{n_comp}d_epoch{epoch}.png'), dpi=100, bbox_inches='tight')
        plt.close()

    # --- (B) Top & Bottom Clusters ---
    mask_top_bottom = np.isin(labels, combined_top_bottom)
    if mask_top_bottom.sum() > 0:
        feat_top_bottom = features[mask_top_bottom]
        labels_top_bottom = labels[mask_top_bottom]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            umap_reducer_4 = umap.UMAP(n_components=2, n_neighbors=100, min_dist=1.0, metric='euclidean', random_state=seed)
            embedding_top_bottom = umap_reducer_4.fit_transform(feat_top_bottom)

        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding_top_bottom[:, 0], embedding_top_bottom[:, 1], c=color_list_top_bottom, s=10)
        
        # 凡例の作成（Top & Bottom クラスタのみ）
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[int(cls_id) % len(color_map)], 
                                     markersize=8, label=f'Cluster {int(cls_id)}')
                          for cls_id in sorted(combined_top_bottom)]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.grid()
        plt.title(f'Umap Visualization top_bottom 2048d_umap{n_comp}d_epoch{epoch}\n(See cluster_color_mapping_epoch{epoch}.csv for full mapping)')
        plt.savefig(os.path.join(out_dir, f'umap_visualization_top_bottom_umap{n_comp}d_epoch{epoch}.png'), dpi=100, bbox_inches='tight')
        plt.close()

    # --- (C) All Clusters (Reduce) ---
    # 元の特徴量(reduced_features)全体を2次元に圧縮
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        umap_reducer_5 = umap.UMAP(n_components=2, n_neighbors=100, min_dist=1.0, metric='euclidean', random_state=seed)
        embedding_reduce = umap_reducer_5.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(embedding_reduce[:, 0], embedding_reduce[:, 1], c=color_list_reduce, s=10)
    plt.grid()
    plt.title(f'Umap Visualization All {len(unique_cluster_ids)} Clusters - 2048d_umap{n_comp}d_epoch{epoch}\n(Cluster color mapping: See cluster_color_mapping_epoch{epoch}.csv)')
    plt.savefig(os.path.join(out_dir, f'umap_visualization_reduce_umap{n_comp}d_epoch{epoch}.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print(colored('==> Saving Graph Completed', 'cyan'))


def plot_cluster_center_distributions(encoder_feature, labels, out_dir, epoch, feature_name="original"):
    """各クラスタの代表ベクトルを確率分布化して可視化する"""
    unique_labels = np.unique(labels)
    all_rows = []

    for cls in unique_labels:
        mask = labels == cls
        if mask.sum() == 0:
            continue

        center = encoder_feature[mask].mean(axis=0)
        center = center.astype(np.float64)
        center = center - np.max(center)
        exp_center = np.exp(center)
        if exp_center.sum() == 0:
            continue
        prob = exp_center / exp_center.sum()

        dims = np.arange(len(prob))
        plt.figure(figsize=(10, 4))
        plt.plot(dims, prob, marker='o', linestyle='-', markersize=2)
        plt.ylim(0, 0.6)
        feature_label = "Original" if feature_name == "original" else "Reduced"
        plt.title(f'Cluster {cls} {feature_label} Vector Probability Distribution (epoch {epoch})')
        plt.xlabel('Dimension index')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cluster_center_distribution_{feature_name}_epoch{epoch}_cluster{cls}.png'))
        plt.close()

        all_rows.extend([
            {'cluster': int(cls), 'dimension': int(d), 'probability': float(p)}
            for d, p in enumerate(prob)
        ])

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(out_dir, f'cluster_center_distributions_{feature_name}_epoch{epoch}.csv'),
            index=False
        )


def select_best_epoch(epoch_results, args):
    if not epoch_results:
        return None
    
    # スコア計算ロジック
    best_score = -float('inf')
    best_epoch = None
    
    eval_data = []

    for result in epoch_results:
        test_acc = result['test_accuracy']
        test_acc_std = result['test_accuracy_std']
        coeff_sum = result['coefficient_sum_abs']
        
        stability = 1.0 / (1.0 + test_acc_std)
        # 正規化は全結果が揃ってからの方が良いが、ここでは簡易的に計算
        score = test_acc * 0.4 + stability * 0.3 + (coeff_sum * 0.01) * 0.3 # 係数和のスケール調整が必要かも
        
        eval_data.append({
            'epoch': result['epoch'],
            'score': score,
            'acc': test_acc,
            'std': test_acc_std
        })
        
        if test_acc > best_score:
            best_score = test_acc
            best_epoch = result['epoch']
            
    # CSV保存
    pd.DataFrame(eval_data).to_csv(os.path.join(args.out_path, f'exp{args.exp}', 'epoch_evaluation.csv'), index=False)
    
    return best_epoch

def seed_worker(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    main()