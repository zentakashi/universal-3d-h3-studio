# Universal 3D × MiniMax H3 Studio

任意の 3D モデル（`.glb` / `.obj`）や参照画像・動画・音声から、**ワンストップで MiniMax H3（Hailuo 3.0）SD 動画（音声付き）を生成**できる汎用スタジオ Web UI ＋ Google Colab L4（24GB VRAM）バックエンド一式です。

## ディレクトリ構成

```text
universal-3d-h3-studio/
├── frontend/
│   └── studio.html                      # 単一ファイル完結型 汎用 Studio Web UI
├── server/
│   └── studio_server.py                 # FastAPI + ComfyUI ブリッジサーバー
├── colab/
│   └── universal_3d_h3_colab_l4.ipynb   # Google Colab L4 ワンストップ起動ノートブック
└── vendor/three/                        # Three.js r170 (オフライン対応)
```

## 主な機能

1. **汎用 3D ローダー & プリセット**
   - 任意の `.glb` / `.obj` をドラッグ＆ドロップまたはファイル選択で読み込み、バウンディングボックスを自動計算してセンタリング・自動スケーリング。
   - 3D モデルが手元にない場合でも、プリセットシーン（`都市・建築` / `プロダクト` / `ジオメトリ`）や、参照素材のみの `Ref2VA` / `FL2VA` / `T2VA` モードでそのまま利用可能。
2. **汎用カメラワーク & カスタム・キーフレーム**
   - モデルサイズに応じて距離を自動調整する 5 種類のプリセットカメラワーク（`360° オービット` / `ドリーイン` / `ローアングル上昇` / `クレーン俯瞰` / `横トラッキング`）。
   - `+ 現在視点を追加` で任意の視点列を登録し、滑らかに補間再生・録画できる **Custom Keyframe Camera Sequencer**。
   - 4 種類のライティングプリセット（`夜景ネオン` / `スタジオ` / `夕暮れ` / `デイライト`）。
3. **マルチモーダル参照スロット管理**
   - 参照画像 `<Picture 1..9>`：3D ビューのワンクリックキャプチャ、またはローカル画像アップロード。
   - 参照動画 `<Video 1..3>`：3D カメラワークの 5 秒・24fps SD 自動録画（WebM）、またはローカル動画アップロード（サーバー側で自動的に `ffmpeg` 24fps SD 変換）。
   - 参照音声 `<Audio 1..3>`：WAV / MP3 アップロード。
4. **MiniMax H3（Ref2VA / FL2VA / T2VA）SD 最適化生成**
   - 解像度：`832×480`（標準 SD 16:9）または `640×360`（高速 SD 16:9）。
   - フレーム数：`17k+5` 準拠（`124` / `107` / `141` フレーム、24fps）。
   - `pruned` 量子化モデル（CUDA 13: `int8_convrot` / CUDA 12: `fp8_scaled`）＋ 8-step Turbo LoRA ＋ `BasicGuider (CFG=1.0)` により Colab L4 (24GB) で安定動作。
   - UI 上の **「L4 ランタイム終了」ボタン** およびノートブック最終セルから `google.colab.runtime.unassign()` を安全に呼び出し可能。

## Google Colab L4 での起動手順

1. `colab/universal_3d_h3_colab_l4.ipynb` を Google Colab にアップロードし、ランタイムを **NVIDIA L4 GPU** に設定します。
2. Cell 1（環境確認・モデル自動DL）→ Cell 2（ComfyUI + `studio_server.py` 起動）を順に実行します。
3. 表示された Colab プロキシ URL を開くと `studio.html` がそのまま立ち上がり、3D 操作・参照素材登録・MiniMax H3 生成を 1 画面で完結できます。
4. 作業終了後は UI 上の **「L4 ランタイム終了」** または最終セルを実行して、L4 ランタイムを必ず解放してください。
