# 環境
- CMake >= 3.18
- python 3.11
- PyTorch 2.6.0
- CUDA 12.2
- NVIDIA CUDA Toolkit

# ビルド
必要なパッケージをインストールした仮想環境を有効化しておく。

ルートディレクトリで、以下のコマンドを実行
```
cmake -B build \
  -D CMAKE_CUDA_COMPILER=<nvccへのパス> \
  -D CMAKE_CUDA_ARCHITECTURES=<対象のGPUのCompute Capability> \
  -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -D Python3/ROOT_DIR="$CONDA_PREFIX" \
  .
```

その後、以下のコマンドを実行
```
cmake --build build --parallel 8
```

# 設定ファイルの作成
シミュレーションのための設定ファイルを作成する必要があります。
```
cd ui
python main.py
```
で設定ファイル作成用のUIが起動します。

手動で作成しても大丈夫です。
configs/以下にあるスキーマ（schema.json）や設定ファイルの例を参考にしてください。

# シミュレーションの実行
```
./build/MD_MLP <configファイルへのパス>
```
でシミュレーションが実行されます。

出力は、設定ファイルの/meta/nameプロパティで設定した名前のフォルダ内に出力されます。
また、標準出力には現在のステップ数と残り時間が出力されます。
出力の間隔は/steps/simulation/log_stepプロパティで変更できます。

また、実行時引数で設定ファイルの特定の値を変更できます。
```
./build/MD_MLP <configファイルへのパス> --<キー名>=<値>
```
のようにします。例えば、
```
./build/MD_MLP <configファイルへのパス> --/common_settings/atoms/n_atoms=1000
```
などとすれば、設定ファイルの値に関わらず粒子数=1000でシミュレーションを行います。
