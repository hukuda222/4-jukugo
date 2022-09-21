# 4-jukugo
自然言語から四字熟語を生成します。lightning-transformers+hydraで実験管理する練習として作りました。

<img src="https://user-images.githubusercontent.com/21185928/191546689-224d0381-61f9-4a29-a4e7-9bf00a876432.png" width="60%">

## 実行コマンド

### install
```
poetry install
```

### 学習
```
poetry run python src/train.py
```

### 推論時
```
poetry run python src/inference.py
```

### streamlit
```
poetry shell
streamlit run src/app.py
```
