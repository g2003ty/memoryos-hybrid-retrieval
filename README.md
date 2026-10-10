# MemoryOS Hybrid Retrieval

本目录整理了页面级稠密检索、BM25 和倒数排名融合（RRF）实验的可复现实验脚本。脚本只读取用户自行准备的 LoCoMo 数据和 MemoryOS snapshot；本目录不包含数据集、snapshot、候选缓存或实验 JSON。

## 目录结构

- `retrieval/`：页面级 dense + BM25 + RRF 的仅检索脚本。
- `evaluation/`：页面级检索评估、原始 prompt 端到端评估，以及作为本地导入依赖的 `run_reliability_memoryos_eval.py` 和 `memory_verifier.py`。
- `memoryos_core/`：MemoryOS 的记忆模块和评估辅助代码。
- `configs/`：dense、BM25 和 RRF 配置。
- `docs/reproducibility.md`：数据准备与运行顺序。

## 安全说明

`memoryos_core/utils.py` 从 `OPENAI_API_KEY` 环境变量读取 API key。请在运行前设置该变量，不要把真实 key 写入脚本、配置或版本库。

静态检查未发现核心脚本中的 Windows 用户目录或 `/root/` 等硬编码绝对路径。

## 数据获取

请通过 LoCoMo 官方发布渠道获取 `locomo10.json`，并通过 MemoryOS 项目或实验发布物获取与该数据对应的预构建 snapshot。将二者放在本地工作区，不要将它们复制进本仓库。snapshot 根目录应包含每个样本的既有 snapshot 文件和 manifest。

## 检索参数

- 稠密检索：all-MiniLM-L6-v2，归一化嵌入，前 50 个候选。
- BM25：k1=1.5，b=0.75，前 50 个候选。
- 混合检索：RRF k=60，最终候选预算 25。
- BM25 文档字段：用户输入、助手回复、时间戳。

## 安装

```powershell
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

评测脚本分布在不同子目录，运行前请把它们加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "$PWD\evaluation;$PWD\memoryos_core;$PWD\retrieval"
```

## API key

```powershell
$env:OPENAI_API_KEY = "<your-key>"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
```

不要在脚本、配置、JSON 或 README 中写入真实 key。

## 运行示例

以下命令中的路径均需替换为本地数据和 snapshot 路径。第一步不调用 LLM，生成候选缓存；第二步使用原始 prompt 做端到端评估。

```powershell
python retrieval\run_page_level_hybrid_retrieval.py `
  --dataset C:\path\to\locomo10.json `
  --storage-root C:\path\to\e2e_storage `
  --limit 100 `
  --output-dir .\outputs\page_level_hybrid_retrieval_100

python evaluation\run_hybrid_retrieval_baseline_e2e.py `
  --baseline-result C:\path\to\baseline.json `
  --candidate-cache .\outputs\page_level_hybrid_retrieval_100\candidate_cache\dense_bm25_hybrid `
  --dataset C:\path\to\locomo10.json `
  --storage-root C:\path\to\e2e_storage `
  --limit 100 `
  --output .\outputs\hybrid_retrieval_baseline_e2e_100.json
```

`run_v2_relevance_only_e2e.py` 依赖原始 baseline/candidate cache。运行前请按上面的 API key 说明设置 `OPENAI_API_KEY`。

## 许可

本整理目录采用 MIT License。第三方模型、数据集、MemoryOS 原始代码及其依赖仍受各自许可证和使用条款约束。