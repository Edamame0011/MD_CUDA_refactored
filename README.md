# 環境
- CMake >= 3.18
- python 3.11
- PyTorch 2.6.0
- CUDA 12.2
- NVIDIA CUDA Toolkit

# ビルド
必要なパッケージをインストールした仮想環境を有効化しておきます。

ルートディレクトリで、以下のコマンドを実行します。
```
cmake -S . -B build \
  -D CMAKE_CUDA_COMPILER=<nvccへのパス> \
  -D CMAKE_CUDA_ARCHITECTURES=<対象のGPUのCompute Capability> \
  -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -D Python3_ROOT_DIR="$CONDA_PREFIX"
```

続いて、以下のコマンドでビルドします。
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
./build/MD_MLP <設定ファイルのパス>
```
でシミュレーションが実行されます。

出力は、設定ファイルの/meta/nameプロパティで設定した名前のフォルダ内に出力されます。
また、標準出力には現在のステップ数と残り時間が出力されます。
出力の間隔は/steps/simulation/log_stepプロパティで変更できます。

また、コマンドライン引数を指定することで、設定ファイル内の値を実行時に上書きできます。
```
./build/MD_MLP <設定ファイルのパス> --key=value
```
例えば、以下のコマンドでは、設定ファイルの /common_settings/atoms/n_atoms の値を上書きし、粒子数を1000に設定してシミュレーションを実行します。
```
./build/MD_MLP <設定ファイルのパス> --/common_settings/atoms/n_atoms=1000
```
複数の値を上書きする事もできます。
```
./build/MD_MLP <設定ファイルのパス> --key1=value1 --key2=value2 --key3=value3
```

# Third-Party Libraries
This project uses the following third-party library:

- **[nlohmann/json](https://github.com/nlohmann/json)**  
  Copyright (c) 2013–2026 Niels Lohmann  
  Licensed under the [MIT License](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT).
