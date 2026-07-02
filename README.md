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
            "sort": <cell linked-listのインデックスに基づいてソートを行うか否か (bool)>
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
            "step"(optional): <前のシミュレーションのステップを引き継ぐかリセットするか ("reset"ならリセット)>
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
||ratio|配列|粒子Aと粒子Bの比率|
|from_file (外部ファイル読み込み)|format|string|ファイルフォーマット (現在は"xyz"のみ)|
||path|string|読み込むファイルのパス|


### common_settings/atoms
#### generate_binary_lj: バイナリljユニットを作成
- n_atoms: 粒子数
- density: 数密度
- ratio: 粒子Aと粒子Bの割合 (2要素の配列)

#### from_file: ファイルから初期状態を読み込み
- format: ファイルのフォーマット (現在xyzフォーマットにのみ対応)
- path: ファイルへのパス

### common_settings/interactions/potentials
#### lennard_jones: レナード-ジョーンズポテンシャル
- numbers: 粒子種毎の原子番号 (2要素の配列)
- sigma: ljパラメータσ (4要素の配列)
- epsilon: ljパラメータε (4要素の配列)
- cutoff: カットオフ距離 (4要素の配列)

#### NNP: NNP (TorchScript形式)
- cutoff: カットオフ距離
- max_edges: 系全体のエッジ数の上限
- model_path: モデルのパス

#### NNP_aoti: NNP (aoti形式)
- cutoff: カットオフ距離
- max_edges: 系全体のエッジ数の上限
- model_path: モデルのパス

#### NNP_force_aoti: 力のみを推論するモデルと、エネルギーを推論するモデルが分かれている場合 (aoti形式)
- cutoff: カットオフ距離
- max_edges: 系全体のエッジ数の上限
- force_model_path: 力を推論するモデルのパス
- energy_model_path: エネルギーを推論するモデルのパス

### steps/simulation/ensemble
#### NVE: NVEシミュレーション
- temperature: 初期温度

#### NVT: NVTシミュレーション
- thermostat: 熱浴の種類 ("Nose-Hoover" or "Bussi" or "Langevin")  


<Nose-Hooverの場合>  
- tau: tauの値
- temperature: 初期温度
- scheduler: 温度変化 (後述)


<Bussiの場合>  
- tau: tauの値
- seed: 乱数シード
- temperature: 初期温度
- scheduler: 温度変化 (後述)


<Langevinの場合>
- tau: tauの値
- seed: 乱数シード
- temperature: 初期温度
- scheduler: 温度変化 (後述)


<schedulerの設定>
- "scheduler": "constant"  
  一定温度でシミュレーション

- "scheduler": "linear"  
  毎ステップ線形に温度を変えながらシミュレーション  
  rate_per_unit_time: 単位時間毎の温度変化 (上げていく場合は+、下げていく場合は-)
  
### steps/observer
#### linear: 線形スケールで時間・運動エネルギー・ポテンシャル・全エネルギー・温度を出力
- interval: 出力間隔 (ステップ)

#### log: ログスケールで時間・運動エネルギー・ポテンシャル・全エネルギー・温度を出力
- divisions: $10^1$ ステップ毎の分割数

#### target_temperature_export: 温度を変えながらシミュレーションを行う際に、特定温度でのトラジェクトリをエクスポート
- target_temperatures: 出力する温度 (配列、必ず高い順にする)
- output_path: 出力するフォルダへのパス（設定したフォルダ直下に、{temp}.xyzで保存されます。）
- is_unwrap: pbcの展開をするか否か (bool)
- initial_temperature: 初期温度
- cooling_rate_per_step: 1ステップあたりの冷却速度 (現在は冷却のみに対応。NVTの設定とは違って絶対値で記述)

#### linear_export_trajectory: 線形スケールでトラジェクトリを保存
- output_path: 出力先ディレクトリ
- is_unwrap: pbcを展開するか否か (bool)
- interval: 出力間隔 (ステップ)

#### log_export_trajectory: ログスケールでトラジェクトリを保存
- output_path: 出力先ディレクトリ
- is_unwrap: pbcを展開するか否か (bool)
- divisions: $10^1$ ステップ毎の分割数

#### log_plus_stride_export_trajectory: ログスケール + 線形間隔をあけてトラジェクトリを保存 (何かバグってるかもしれないです。)
- num_trajectory: 保存するトラジェクトリの数
- stride: 何ステップ開けるか
- output_path: 出力先ディレクトリ
- is_unwrap: pbcを展開するか否か (bool)
- divisions: $10^1$ ステップ毎の分割数

## Dependencies
This project uses the following third-party libraries:

* [nlohmann/json](https://github.com/nlohmann/json) - JSON for Modern C++ (MIT License)
