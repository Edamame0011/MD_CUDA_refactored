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
            <modeに応じた設定>
        }, 
        "interactions": {
            "cell_list": <cell linked-listを用いるか否か ("true" or "false")>,
            "sort": <cell linked-listのインデックスに基づいてソートを行うか否か ("true" or "false")>
            "neighbour_list": {
                "cutoff": <カットオフ距離>, 
                "margin": <カットオフ距離からのマージン>
            }, 
            "potentials": {
                "type": <用いるポテンシャルの種類>, 
                <typeに応じた設定>
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
                "use_graph": <CUDA Graphsを用いるか否か ("True" or "False")>
            }, 
            "observer": {
                "type": <出力の種類>, 
                <typeに応じた設定>
            }
        }, 

        {
            "name": <シミュレーションステップの名前>, 
            "simulation": {
                "dt": <タイムステップ>, 
                "simulation_time": <シミュレーション時間 (単位系はmeta/unitで設定)>, 
                "ensemble": {
                    "type": <アンサンブルの種類 ("NVE" or "NVT")>, 
                    <typeに応じた設定>
                }, 
                "use_graph": <CUDA Graphsを用いるか否か ("True" or "False")>
            }, 
            "observer": {
                "type": <出力の種類>, 
                <typeに応じた設定>
            }, 
            "step": <前のシミュレーションのステップを引き継ぐかリセットするか ("reset"ならリセット)>
        }, 

        <必要であれば他のシミュレーション設定>
    ] 
}
```

### common_settings/atoms
- type
  generate_binary_lj: バイナリljユニットを作成
  設定項目
-- "numbers": 粒子種毎の原子番号 (2要素の配列)
-- "sigma": ljパラメータσ (4要素の配列)
-- "epsilon": ljパラメータε (4要素の配列)
-- "cutoff": カットオフ距離 (4要素の配列) 

## Dependencies
This project uses the following third-party libraries:

* [nlohmann/json](https://github.com/nlohmann/json) - JSON for Modern C++ (MIT License)
