# MemoryOS Hybrid Retrieval

鏈洰褰曟暣鐞嗕簡椤甸潰绾х瀵嗘绱€丅M25 鍜屽€掓暟鎺掑悕铻嶅悎锛圧RF锛夊疄楠岀殑鍙鐜板疄楠岃剼鏈€傝剼鏈彧璇诲彇鐢ㄦ埛鑷鍑嗗鐨?LoCoMo 鏁版嵁鍜?MemoryOS snapshot锛涙湰鐩綍涓嶅寘鍚暟鎹泦銆乻napshot銆佸€欓€夌紦瀛樻垨瀹為獙 JSON銆?
## 鐩綍缁撴瀯

- `retrieval/`锛氶〉闈㈢骇 dense + BM25 + RRF 鐨勪粎妫€绱㈣剼鏈€?- `evaluation/`锛氶〉闈㈢骇妫€绱㈣瘎浼般€佸師濮?prompt 绔埌绔瘎浼帮紝浠ュ強浣滀负鏈湴瀵煎叆渚濊禆鐨?`run_reliability_memoryos_eval.py` 鍜?`memory_verifier.py`銆?- `memoryos_core/`锛歁emoryOS 鐨勮蹇嗘ā鍧楀拰璇勪及杈呭姪浠ｇ爜銆?- `configs/`锛歞ense銆丅M25 鍜?RRF 閰嶇疆銆?- `docs/reproducibility.md`锛氭暟鎹噯澶囦笌杩愯椤哄簭銆?
## 瀹夊叏璇存槑

`memoryos_core/utils.py` 浠?`OPENAI_API_KEY` 鐜鍙橀噺璇诲彇 API key銆傝鍦ㄨ繍琛屽墠璁剧疆璇ュ彉閲忥紝涓嶈鎶婄湡瀹?key 鍐欏叆鑴氭湰銆侀厤缃垨鐗堟湰搴撱€?
闈欐€佹鏌ユ湭鍙戠幇鏍稿績鑴氭湰涓殑 Windows 鐢ㄦ埛鐩綍鎴?`/root/` 绛夌‖缂栫爜缁濆璺緞銆?
## 鏁版嵁鑾峰彇

璇烽€氳繃 LoCoMo 瀹樻柟鍙戝竷娓犻亾鑾峰彇 `locomo10.json`锛屽苟閫氳繃 MemoryOS 椤圭洰鎴栧疄楠屽彂甯冪墿鑾峰彇涓庤鏁版嵁瀵瑰簲鐨勯鏋勫缓 snapshot銆傚皢浜岃€呮斁鍦ㄦ湰鍦板伐浣滃尯锛屼笉瑕佸皢瀹冧滑澶嶅埗杩涙湰浠撳簱銆俿napshot 鏍圭洰褰曞簲鍖呭惈姣忎釜鏍锋湰鐨勬棦鏈?snapshot 鏂囦欢鍜?manifest銆?
## 瀹夎

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

璇勬祴鑴氭湰鍒嗗竷鍦ㄤ笉鍚屽瓙鐩綍锛岃繍琛屽墠璇锋妸瀹冧滑鍔犲叆 `PYTHONPATH`锛屼娇鏃т唬鐮佷腑鐨勫悓鍚嶆湰鍦板鍏ヤ繚鎸佸彲鐢細

```powershell
$env:PYTHONPATH = "$PWD\evaluation;$PWD\memoryos_core;$PWD\retrieval"
```

## API key

```powershell
$env:OPENAI_API_KEY = "<your-key>"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"  # 濡備娇鐢ㄥ吋瀹规湇鍔★紝鍙浛鎹?```

涓嶈鍦ㄨ剼鏈€侀厤缃€丣SON 鎴?README 涓啓鍏ョ湡瀹?key銆?
## 杩愯绀轰緥

浠ヤ笅鍛戒护涓殑璺緞鍧囬渶鏇挎崲涓烘湰鍦版暟鎹拰 snapshot 璺緞銆傜涓€姝ヤ笉璋冪敤 LLM锛岀敓鎴愬€欓€夌紦瀛橈紱绗簩姝ヤ娇鐢ㄥ師濮?prompt 鍋氱鍒扮璇勪及銆?
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

`run_v2_relevance_only_e2e.py` 渚濊禆鍘熷 baseline/candidate cache銆傝繍琛屽墠璇锋寜涓婇潰鐨?API key 璇存槑璁剧疆 `OPENAI_API_KEY`銆?
## 璁稿彲

鏈暣鐞嗙洰褰曢噰鐢?MIT License銆傜涓夋柟妯″瀷銆佹暟鎹泦銆丮emoryOS 鍘熷浠ｇ爜鍙婂叾渚濊禆浠嶅彈鍚勮嚜璁稿彲璇佸拰浣跨敤鏉℃绾︽潫銆?
