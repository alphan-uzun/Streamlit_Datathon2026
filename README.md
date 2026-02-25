# STAPLE EN->HU Streamlit Repro Bundle

GitHub repo purpose:
- **Presentation/demo bundle** for the STAPLE EN->HU project
- Runs the Streamlit dashboards on a new computer **without retraining**
- Includes the data/result JSON files used by the dashboards

Quick start (Anaconda Prompt):
```cmd
cd /d C:\path\to\streamlit_repro_bundle
conda create -n staple_streamlit python=3.10 -y
conda activate staple_streamlit
pip install -r requirements-streamlit.txt
python -m streamlit run streamlit_cumulative_gains_dashboard.py
```

This folder is a **clean bundle** of the Streamlit dashboards and the **dev/test data + result files** they use.

It is intended for:
- presenting the project (`streamlit_cumulative_gains_dashboard.py`)
- browsing results (`streamlit_best_model_dashboard.py`)
- reproducing the **best valid model pipeline settings** (with the required checkpoints/artifacts available)

## What Is Included

- Streamlit apps
  - `streamlit_cumulative_gains_dashboard.py` (presentation-focused)
  - `streamlit_best_model_dashboard.py` (analysis-focused)
- Dashboard dependencies (local modules used by the apps)
  - `data_io.py`, `metrics.py`, `textnorm.py`, `compare_all_metrics.py`
- Dev/test datasets used by the dashboards
  - `staple-2020/en_hu/dev...gold/aws...`
  - `staple-2020/en_hu/test...gold/aws...`
- Candidate pools and reranked outputs used in the dashboards
- Official scorer summaries used in the dashboards
- DeepL back-translation JSON files used in the error-analysis pages
- Utility scripts for reranking / scoring / exporting

See `MANIFEST.txt` for the exact bundled files.

## What Is NOT Included (Large Artifacts)

These are **not** copied into this bundle because they are large model checkpoints/artifacts:

- LoRA generator adapter (used for generation)
  - `..\artifacts\nllb_lora_en_hu_v34e3_recall_long`
- Embedding/encoder checkpoint (used by rerankers)
  - `..\artifacts\geo_filter_v31_manual\geo_filter_best.pt`
- Best frequency-aware reranker checkpoint
  - `..\artifacts\relevance_regressor_v27a_v34e3_big\ranker_best.pt`
- Optional embedding cache (recommended for speed)
  - `..\artifacts\cache\geo_mpnet_embeddings.pt`

If your artifacts live elsewhere, update the paths in the commands below.

## Environment (CMD)

Examples below are for **Windows Command Prompt (`cmd.exe`)**.

Activate your environment first (example):

```cmd
conda activate staple2020_clean
cd /d C:\Users\aalti\Desktop\datathon\STAPLE_2\streamlit_repro_bundle
```

## Install Runtime Dependencies (if needed)

Minimum for Streamlit dashboards:

```cmd
pip install streamlit altair pandas
```

Optional (presentation asset export script):

```cmd
pip install vl-convert-python dataframe-image graphviz
```

If schema PNG export fails, also install the Graphviz binary (`dot`) and add it to PATH:

```cmd
winget install Graphviz.Graphviz
```

Verify:

```cmd
where dot
```

## Run The Dashboards

### 1) Presentation flow dashboard (recommended for presenting)

```cmd
python -m streamlit run streamlit_cumulative_gains_dashboard.py
```

### 2) Analysis dashboard (fuller diagnostics)

```cmd
python -m streamlit run streamlit_best_model_dashboard.py
```

## Reproduce The Best Valid Model Settings (Using Existing Checkpoints)

This reproduces the **best valid frequency-aware reranker + AWS-first-guess** pipeline on the bundled `v34e3_big` candidate pools.

### A) Dev split rerank (soft)

```cmd
python rerank_relevance_regressor_noleak.py ^
  --encoder-geo-model ..\artifacts\geo_filter_v31_manual\geo_filter_best.pt ^
  --ranker-ckpt ..\artifacts\relevance_regressor_v27a_v34e3_big\ranker_best.pt ^
  --prompts-file staple-2020\en_hu\dev.en_hu.2020-02-20.gold.txt ^
  --aws-file staple-2020\en_hu\dev.en_hu.aws_baseline.pred.txt ^
  --candidates-json candidates_dev_v34e3_big.json ^
  --out-json reranked_dev_v34e3_big_relevance_v27a.json ^
  --embedding-cache ..\artifacts\cache\geo_mpnet_embeddings.pt ^
  --eval-gold staple-2020\en_hu\dev.en_hu.2020-02-20.gold.txt
```

### B) Dev split AWS-first-guess anchor

```cmd
python aws_anchor_diagnostics.py ^
  --gold staple-2020\en_hu\dev.en_hu.2020-02-20.gold.txt ^
  --aws staple-2020\en_hu\dev.en_hu.aws_baseline.pred.txt ^
  --reranked-json reranked_dev_v34e3_big_relevance_v27a.json ^
  --out-json reranked_dev_v34e3_big_relevance_v27a_aws_anchor.json ^
  --k 5
```

### C) Compare raw vs reranked on dev

```cmd
python compare_all_metrics.py ^
  --gold staple-2020\en_hu\dev.en_hu.2020-02-20.gold.txt ^
  --aws staple-2020\en_hu\dev.en_hu.aws_baseline.pred.txt ^
  --candidates-json candidates_dev_v34e3_big.json ^
  --reranked-json reranked_dev_v34e3_big_relevance_v27a_aws_anchor.json ^
  --out-json compare_all_dev_v34e3_big_relevance_v27a_aws_anchor_report.json
```

### D) Test split rerank (soft)

```cmd
python rerank_relevance_regressor_noleak.py ^
  --encoder-geo-model ..\artifacts\geo_filter_v31_manual\geo_filter_best.pt ^
  --ranker-ckpt ..\artifacts\relevance_regressor_v27a_v34e3_big\ranker_best.pt ^
  --prompts-file staple-2020\en_hu\test.en_hu.2020-02-20.gold.txt ^
  --aws-file staple-2020\en_hu\test.en_hu.aws_baseline.pred.txt ^
  --candidates-json candidates_test_v34e3_big.json ^
  --out-json reranked_test_v34e3_big_relevance_v27a.json ^
  --embedding-cache ..\artifacts\cache\geo_mpnet_embeddings.pt ^
  --eval-gold staple-2020\en_hu\test.en_hu.2020-02-20.gold.txt
```

### E) Test split AWS-first-guess anchor

```cmd
python aws_anchor_diagnostics.py ^
  --gold staple-2020\en_hu\test.en_hu.2020-02-20.gold.txt ^
  --aws staple-2020\en_hu\test.en_hu.aws_baseline.pred.txt ^
  --reranked-json reranked_test_v34e3_big_relevance_v27a.json ^
  --out-json reranked_test_v34e3_big_relevance_v27a_aws_anchor.json ^
  --k 5
```

### F) Compare raw vs reranked on test

```cmd
python compare_all_metrics.py ^
  --gold staple-2020\en_hu\test.en_hu.2020-02-20.gold.txt ^
  --aws staple-2020\en_hu\test.en_hu.aws_baseline.pred.txt ^
  --candidates-json candidates_test_v34e3_big.json ^
  --reranked-json reranked_test_v34e3_big_relevance_v27a_aws_anchor.json ^
  --out-json compare_all_test_v34e3_big_relevance_v27a_aws_anchor_report.json
```

## (Optional) Reproduce XXL Candidate-Pool Impact (No Retraining)

If you already have the same checkpoints and want to test larger candidate pools:

### Generate dev-only XXL candidates (slow)

```cmd
python generate_candidates.py ^
  --base-model facebook/nllb-200-distilled-600M ^
  --adapter-dir ..\artifacts\nllb_lora_en_hu_v34e3_recall_long ^
  --batch-size 4 ^
  --beam-size 32 ^
  --sample-n 192 ^
  --max-new-tokens 64 ^
  --progress-every-batches 5 ^
  --dev-prompts staple-2020\en_hu\dev.en_hu.2020-02-20.gold.txt ^
  --out-dev candidates_dev_v34e3_xxl.json ^
  --out-dev-meta candidates_dev_v34e3_xxl.meta.json
```

Then rerank + anchor with the same commands as above, replacing:
- `candidates_dev_v34e3_big.json` -> `candidates_dev_v34e3_xxl.json`
- output filenames accordingly

## Official Duolingo/STAPLE Scorer (Optional)

### 1) Clone scorer repo inside this bundle (once)

```cmd
mkdir external
cd external
git clone https://github.com/duolingo/duolingo-sharedtask-2020.git
cd ..
```

### 2) Export a reranked JSON to `.pred.txt` and score it

Example (`test`, anchored, `k=5`):

```cmd
python export_staple_pred.py ^
  --input-json reranked_test_v34e3_big_relevance_v27a_aws_anchor.json ^
  --k 5 ^
  --prompt-file staple-2020\en_hu\test.en_hu.2020-02-20.gold.txt ^
  --out-pred submissions\test_v34e3_big_relevance_v27a_aws_anchor_k5.pred.txt ^
  --score-goldfile staple-2020\en_hu\test.en_hu.2020-02-20.gold.txt
```

## DeepL Error Analysis Files (Optional)

The bundle already includes saved DeepL back-translation JSONs used by the error-analysis pages.

If you want to regenerate them, set your key in `cmd.exe`:

```cmd
set DEEPL_API_KEY=YOUR_KEY_HERE
```

Then run the DeepL scripts/inline snippets from your project notes (not bundled as a single script here because they were used ad hoc for specific prompt slices).

## Presentation Asset Export (PNG/SVG/CSV)

This uses the **same visual style** (Altair/Graphviz-based, not Matplotlib).

```cmd
python export_presentation_assets.py --split test --k 5 --out-dir presentation_exports
```

If it prints missing dependency messages, install the suggested packages (see above).

## Notes

- The bundle includes **data and result files used by the Streamlit dashboards**, not all training artifacts.
- For exact reproducibility of the best model outputs, use the same checkpoints listed in the “NOT Included” section.
- The dashboards show **presentation-friendly labels** (no internal version IDs), while filenames still keep the original experiment naming.
#   S t r e a m l i t _ D a t a t h o n 2 0 2 6  
 