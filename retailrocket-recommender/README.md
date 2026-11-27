# OPeRA 2025 Online Shopping Session Recommender

This repository now powers a session-based and content-based product recommendation demo using **OPeRA**, a real-user online-shopping behaviour dataset collected in 2025 by the Northeastern University Human-centered AI Lab.

> **Verified observation window:** 24 March 2025 to 8 May 2025. Dates are read directly from the Parquet action timestamps.

The GitHub repository name still contains `retailrocket-recommender` for continuity with the existing project link, but the live dashboard now uses OPeRA 2025 data. It no longer presents the old RetailRocket 2015 dataset as current.

## Dataset and provenance

- Dataset: [NEU-HAI / OPeRA](https://huggingface.co/datasets/NEU-HAI/OPeRA)
- Paper: [OPeRA: A Dataset of Observation, Persona, Rationale, and Action](https://arxiv.org/abs/2506.05606)
- Citation: Wang et al. (2025). *OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation*. arXiv:2506.05606.
- Licence: CC BY 4.0.

The full action dataset contains 28,904 observed browser actions across 692 shopping sessions and 51 research participants. The live data pipeline extracted 652 product ASINs with product metadata from action-associated product records.

## What the application demonstrates

- Daily activity across click, scroll, navigation, input and tab-activation actions.
- Product-link actions and purchase-intent control clicks as separate activity types.
- Product metadata and product-interest exploration by ASIN.
- Session co-visitation recommendations based on products associated with earlier actions in a session.
- Product-to-product recommendations using TF-IDF title features and cosine similarity.
- Maximal Marginal Relevance (MMR) reranking to balance similarity and diversity.
- A session explorer that displays the product-associated history and next-product suggestions.

## Important interpretation limits

- OPeRA is a small observational research sample, not a complete retailer clickstream or market-wide visitor dataset.
- A `click_type=purchase` record refers to a purchase-intent control click (such as checkout, buy now or add to cart). It is **not** proof of a completed order.
- Product metadata is collected from products attached to action records; it should not be interpreted as full catalogue exposure or transaction history.
- The current chronological session holdout contains only **4 eligible next-product targets** for this product-linked subset. That is too few to make reliable Recall@K, NDCG@K or conversion-performance claims. The dashboard deliberately does not show zero-filled accuracy scores.
- The recommendation explorer is an algorithm demonstration. It does not establish business lift or an online A/B test result.

## Recommendation approach

1. **Session co-visitation:** count products that appear together in product-associated actions across training sessions. Items already in the current session history are excluded.
2. **Content-based fallback:** vectorize product titles with TF-IDF and rank candidates using cosine similarity when session history is too sparse for co-visitation.
3. **MMR reranking:** use product-title similarity to reduce repetitive suggestions and diversify the top results.

## Automated build

The GitHub Actions workflow `.github/workflows/refresh-opera-2025.yml` downloads the public Parquet shards from Hugging Face, extracts the required fields, rebuilds the recommendation payload and publishes the static dashboard data. Raw Parquet files are temporary and are not committed to this repository.

Run locally:

```powershell
cd retailrocket-recommender
pip install -r requirements.txt
python scripts/build_opera_recommender.py
python -m http.server 8000
```

Open `http://localhost:8000` after running the script.

## Dashboard outputs

- `data.js` — compact browser dashboard payload.
- `data.json` — readable JSON copy of the dashboard payload.
- `reports/opera_2025_summary.json` — source range, counts, product data, recommendation outputs and evaluation limitations.
- `reports/metrics.json` — explicit status explaining why accuracy metrics are not reported for this sample.

All observations retain their original 2025 timestamps. No dates are shifted or synthetically generated.
