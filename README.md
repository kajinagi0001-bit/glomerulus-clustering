# glomerulus-clustering
糸球体画像の特徴学習，クラスタリング，可視化を行うプロジェクト
[ディレクトリ構成]
encoder.py   エンコーダ(ResNet50)の対照学習(MoCov2)を行うプログラム
clustering.py   学習済みエンコーダから特徴を抽出し，次元削減(Umap: 4096次元→24次元)，クラスタリング(GMM)を行うプログラム
visualize.py   学習済みエンコーダから，各画像の注目領域を可視化する
tools  モデルやAugmentationのツール群

入力想定:
dataset
-- rat
   -- diabetes
      -- kidney_1
        -- glo_1.jpg
   -- not-diabetes
-- human

[使用方法]

