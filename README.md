## configの書き方
```
{
    "meta": {
        "name": <シミュレーションの名前>, 
        "unit": <ユニット ("lj" or "metal")>, 
        "seed": <初期速度の設定に用いる乱数シード>
    }, 
    "common_settings": {
        "atoms": {
            "mode": <初期配置をどのように作成するか ("generate_binary_lj" or "from_file")>, 
            // modeに応じた設定
        }, 
        "interactions": {
            "cell_list": <cell linked-listを用いるか否か (bool)>,
            "sort": <cell linked-listのインデックスに基づいてソートを行うか否か (bool)>, 
            "neighbour_list": {
                "cutoff": <カットオフ距離>, 
                "margin": <カットオフ距離からのマージン>
            }, 
            "potentials": {
                "type": <用いるポテンシャルの種類>, 
                // typeに応じた設定
            }
        }
    }, 
    "steps": [
        {
            "name": <シミュレーションステップの名前>, 
            "simulation": {
                "dt": <タイムステップ>, 
                "simulation_time": <シミュレーション時間 (単位系はmeta/unitで設定)>, 
                "ensemble": {
                    "type": <アンサンブルの種類 ("NVE" or "NVT")>, 
                    <typeに応じた設定>
                }, 
                "use_graph": <CUDA Graphsを用いるか否か (bool)>
            }, 
            "observer": {
                "type": <出力の種類>, 
                // typeに応じた設定
            }
        }, 

        // 以下は続けて行うシミュレーションの設定 (optional)
        {
            "name": <シミュレーションステップの名前>, 
            "simulation": {
                "dt": <タイムステップ>, 
                "simulation_time": <シミュレーション時間 (単位系はmeta/unitで設定)>, 
                "ensemble": {
                    "type": <アンサンブルの種類 ("NVE" or "NVT")>, 
                    // typeに応じた設定
                }, 
                "use_graph": <CUDA Graphsを用いるか否か (bool)>
            }, 
            "observer": {
                "type": <出力の種類>, 
                // typeに応じた設定
            },
            "save_last_structure (optional)": {
                "path": 保存先のパス
                "is_unwrap": pbcの補正を展開するか否か (bool)
            }
            "step"(optional): <前のシミュレーションのステップ数を引き継ぐかリセットするか ("reset"ならリセット)>
        }, 

        // 必要であれば他のシミュレーション設定
    ] 
}
```

### 共通設定 (common_settings)
#### atoms  
modeの値によって、必要なパラメータが異なります。

|modeの値|パラメータ名|型|説明|
|---|---|---|---|
|generate_binary_lj (バイナリLJ作成)|n_atoms|int|粒子数|
||density|float|数密度|
||ratio|float[2]|粒子Aと粒子Bの比率|
|from_file (外部ファイル読み込み)|format|string|ファイルフォーマット (現在は"xyz"のみ)|
||path|string|読み込むファイルのパス|

#### interactions/potentials  
typeの値によって、必要なパラメータが異なります。
|typeの値|パラメータ名|型|説明|
|---|---|---|---|
|lennard_jones|numbers|int[]|粒子種毎の原子番号 ( $s$ 要素 )|
||sigma|float[]|LJパラメータ $\sigma$ ( $s^2$ 要素)|
||epsilon|float[]|LJパラメータ $\epsilon$ ( $s^2$ 要素)|
||cutoff|float[]|カットオフ距離 ( $s^2$ 要素)|
|NNP (TorchScript形式)|max_edges|int|系全体のエッジ数の上限|
||model_path|string|モデルのパス|
||cutoff|float|カットオフ距離|
|NNP_aoti (AOT Inductor形式)|max_edges|int|系全体のエッジ数の上限|
||model_path|string|モデルのパス|
||cutoff|float|カットオフ距離|
|NNP_force_aoti (力/エネルギー分離)|max_edges|int|系全体のエッジ数の上限|
||force_model_path|string|力推論モデルのパス|
||energy_model_path|string|ポテンシャル推論モデルのパス|
||cutoff|float|カットオフ距離|

### シミュレーションステップ  
stepsは配列になっており、複数のシミュレーションを連続して実行できます。

#### simulation/ensemble
##### NVEの場合
- temperature (float): 初期温度

##### NVTの場合
- thermostat (string): 熱浴の種類
- 各熱浴毎のパラメータ:


    |パラメータ|Nose-Hoover|Bussi|Langevin|
    |---|---|---|---|
    |tau|○|○|○|
    |temperature|○|○|○|
    |seed|-|○|○|
    |scheduler|○|○|○|


schedulerの設定値
- "constant": 一定温度でシミュレーションを行います。
- "linear" : 毎ステップ線形に温度を変化させます。
    - rate_per_unit_time (float): 単位時間あたりの温度変化量（昇温は +、降温は -）

#### observer (出力設定)
##### 時間・エネルギー・温度を出力
- linear: 線形出力
  - interval (int): 出力間隔 (ステップ数)
- log: ログスケール出力
  - divisions (int): $10^1$ ステップ毎の分割数

##### トラジェクトリの出力


|typeの値|パラメータ名|型|説明|
|linear_export_trajectory (線形保存)|output_path|string|出力先ファイルパス|
||is_unwrap|bool|PBCを展開するか否か|
||interval|int|出力間隔（ステップ数）|
|log_export_trajectory (ログスケール保存)|output_path|string|出力先ファイルパス|
||is_unwrap|bool|PBCを展開するか否か|
||divisions|int| $10^1$ ステップ毎の分割数|
|log_plus_stride_export_trajectory (ログ＋線形間隔保存 ※バグの可能性あり)|output_path|string|出力先ディレクトリ|
||is_unwrap|bool|PBCを展開するか否か|
||divisions|int| $10^1$ ステップ毎の分割数|
||num_trajectory|int|保存するトラジェクトリの数|
||stride|int|空けるステップ数|

##### その他
- target_temperature_export (特定温度での構造保存)
  - target_temperatures (float[]): 出力したい温度のリスト（必ず高温から降順で指定）
  - output_path (string): 保存先フォルダ（直下に {temp}.xyz として保存）
  - initial_temperature (float): 初期温度
  - cooling_rate_per_step (float): 1ステップあたりの冷却速度（絶対値で指定）
  - is_unwrap (bool): PBCを展開するか否か

## Dependencies
This project uses the following third-party libraries:

* [nlohmann/json](https://github.com/nlohmann/json) - JSON for Modern C++ (MIT License)
