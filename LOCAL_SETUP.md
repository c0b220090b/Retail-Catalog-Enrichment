# Local Model Setup (Fork Notes)

このファイルは NVIDIA公式Blueprint [Retail-Catalog-Enrichment](https://github.com/NVIDIA-AI-Blueprints/Retail-Catalog-Enrichment)
を、手元のRTX GPU（VRAM 約45GB）1枚でローカルOSSモデルに差し替えて動かすための実行手順メモです。
`README.md`（upstream本体）は変更せず、このファイルに差分運用の手順を集約しています。

## ゴール

- 既定のNVIDIA NIM（Nemotron 3 Nano Omni / Nemotron 3.5 Lightning）をvLLMで立てたローカルOSSモデルに差し替える
- FLUX / TRELLIS はいったんクラウドAPI経由で使い、後で余力があれば差し替える
- 45GB VRAM 1枚に収まる構成で、VLM解析〜画像生成までの一連の流れを手元で動かす

## ハードウェア / ソフトウェア前提

| 項目 | 内容 |
|---|---|
| GPU | RTX系 1枚（VRAM ~45GB） |
| Docker | 28.0+ |
| Python | 3.11+ |
| その他 | NVIDIA Container Toolkit、`uv`パッケージマネージャ |

## モデル差し替えマッピング

| 役割 | upstream既定 (config.yamlの値) | 差し替え先 | 提供方法 | 目安VRAM |
|---|---|---|---|---|
| VLM（画像解析） | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Qwen2.5-VL-7B-Instruct | vLLM (ローカル) | ~6GB (量子化) / ~16GB (fp16) |
| LLM（プロンプト企画） | `nvidia/nemotron-3.5-lightning` | Llama 3.1 8B Instruct | vLLM (ローカル) | ~6GB (量子化) |
| Embedding（ポリシー照合） | `nvidia/nv-embedqa-e5-v5` | そのまま維持 | 元のNIM/クラウド | ~1GB未満 |
| 画像生成 | FLUX Kontext Dev | 当面は元モデルをクラウドAPI経由 | NVIDIA hosted NIM | - |
| 3D生成 | Microsoft TRELLIS | 当面は元モデルをクラウドAPI経由 | NVIDIA hosted NIM | - |

> FLUX / TRELLIS はコードが `/v1/infer` という専用APIを前提にしており、別モデルへの丸ごと差し替えはラッパーの書き直しが必要。
> まずはクラウド経由で動作確認し、余力があれば同系統モデルの量子化版に差し替える。

---

## セットアップ手順

### フェーズ0：環境確認

```bash
nvidia-smi          # ドライバ・VRAM確認
docker --version    # 28.0+
python3 --version   # 3.11+
docker info | grep -i nvidia   # NVIDIA Container Toolkit有効化確認
```

### フェーズ1：依存関係インストール

このリポジトリのルートで:

```bash
cp .env.example .env

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

`.env` に以下を設定：

```bash
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # FLUX/TRELLISのクラウド利用に使用
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx                                            # モデルDL用（Qwen/Llamaは基本トークン不要）
```

### フェーズ2：LLM/VLMをvLLMでローカル起動

別ターミナルを2つ用意。

```bash
pip install vllm

# ターミナルA：VLM（画像解析役）
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8001 --gpu-memory-utilization 0.3

# ターミナルB：LLM（プロンプト企画役）
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8002 --gpu-memory-utilization 0.3
```

`--gpu-memory-utilization` で各プロセスの使用VRAM比率を制限し、45GB内に2モデル同時に収める。
起動後、以下でヘルスチェック：

```bash
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
```

### フェーズ3：`shared/config/config.yaml` を書き換え

upstream既定の値（Docker Compose用のサービス名 `nim-vlm` 等）から、ローカルのvLLMエンドポイントに書き換える：

```yaml
vlm:
  url: "http://localhost:8001/v1"
  model: "Qwen/Qwen2.5-VL-7B-Instruct"

llm:
  url: "http://localhost:8002/v1"
  model: "meta-llama/Llama-3.1-8B-Instruct"

# 当面は元のNIM/クラウド設定のまま
embeddings:
  url: "http://localhost:8005/v1"
  model: "nvidia/nv-embedqa-e5-v5"

flux:
  url: "https://integrate.api.nvidia.com/v1/infer"   # クラウド経由、要検証

trellis:
  url: "https://integrate.api.nvidia.com/v1/infer"   # クラウド経由、要検証
```

> `flux` / `trellis` のクラウドエンドポイント指定が実際に機能するかは、`docs/DOCKER.md` を確認しつつ検証が必要。
> 動かない場合はこのフェーズでは画像生成/3D生成をスキップし、VLM/LLM経路のみ確認する。

### フェーズ4：バックエンド単体で疎通確認

```bash
uvicorn --app-dir src backend.main:app --host 0.0.0.0 --port 8000 --reload
```

別ターミナルから:

```bash
curl -X POST http://localhost:8000/vlm/analyze \
  -F "image=@sample_product.jpg"
```

レスポンスが返れば、モデル差し替えの核心部分は成功。

### フェーズ5：フロントエンドで通し確認

```bash
cd src/ui
pnpm install
pnpm dev
```

`http://localhost:3000` で画像アップロード→VLM解析→カテゴリ分類までをUIから確認。

### フェーズ6：画像生成（FLUX）を接続

- まずはクラウドNIM経由で動作確認
- 慣れてきたら FLUX.1-schnell（Apache 2.0、非商用縛りなし）の量子化版（NF4/fp8）へ差し替えを検討
  - この場合 `/v1/infer` 互換のラッパーAPIを自前で実装する必要あり

### フェーズ7：3D生成（TRELLIS）を接続

最後に着手。急がず、時間があるときに。

---

## トラブルシューティングメモ

| 症状 | 確認ポイント |
|---|---|
| vLLM起動時にOOM | `--gpu-memory-utilization` を下げる、`--max-model-len` を制限する |
| VLM/LLM同時起動でVRAM不足 | どちらか一方を先に落としてから他方を起動、または量子化版に切り替え |
| config.yaml変更が反映されない | バックエンド再起動が必要な場合あり（`--reload`でも設定ファイル変更は反映されないことがある） |
| FLUX/TRELLISクラウドエンドポイントが繋がらない | `docs/DOCKER.md` の想定形式と一致しているか確認。合わない場合は該当機能を一時的に無効化 |

## 進捗ログ

- [ ] フェーズ0: 環境確認
- [ ] フェーズ1: 依存関係インストール
- [ ] フェーズ2: vLLMでVLM/LLM起動
- [ ] フェーズ3: config.yaml書き換え
- [ ] フェーズ4: バックエンド疎通確認
- [ ] フェーズ5: フロントエンド通し確認
- [ ] フェーズ6: FLUX接続
- [ ] フェーズ7: TRELLIS接続

## upstreamとの同期

このリポジトリは `upstream` として NVIDIA公式リポジトリを参照しています。
最新化する場合：

```bash
git fetch upstream
git merge upstream/main
```

`README.md` 本体には手を入れず、差分は `LOCAL_SETUP.md`（このファイル）と `shared/config/config.yaml` の変更に閉じているため、
upstream追従時のコンフリクトは基本的に発生しません。
