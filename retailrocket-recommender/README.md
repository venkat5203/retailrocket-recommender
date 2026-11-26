# Current Retail Catalogue Recommendation System

This project refreshes a product catalogue snapshot using the Open Food Facts API and builds transparent content-based product recommendations.

## Important scope change
The previous Retailrocket clickstream data is from 2015. It is historical and should not be described as current. This project now uses product catalogue records reported as modified on or after 2026-01-01 UTC.

The new recommendation method is content-based, not collaborative filtering. It uses product names, brands, categories, quantity, ingredients and nutrition grade. It does not claim to model real 2026 users, views, baskets, purchases, revenue or conversion.

## Source
- Open Food Facts: https://world.openfoodfacts.org/
- Structured search API: https://world.openfoodfacts.org/api/v2/search
- API docs: https://openfoodfacts.github.io/documentation/docs/Product-Opener/v2/search/get-search/
- Database licence: Open Database License (ODbL) 1.0; attribute Open Food Facts contributors and follow ODbL share-alike requirements.
- Products are community-contributed catalogue entries; modification date does not mean the item was sold or purchased on that date.

The API documents sort_by=last_modified_t and pagination. To reduce load, each refresh fetches one page of up to 100 of the most recently modified products. The collector sends an identifying User-Agent, tries the primary Open Food Facts Search API host and a second documented host, retries HTTP 429/503 responses with bounded delays, and filters every record by last_modified_t >= 2026-01-01. This is a bounded recent sample, not a complete export of Open Food Facts. GitHub Actions stores the last successful dashboard snapshot and reuses it during a temporary API outage, only while that snapshot is no more than 45 days old. The refresh status report distinguishes a fresh snapshot from a cached fallback.

## Run on Windows
From the retailrocket-recommender folder:
1. python -m venv .venv
2. .venv\Scripts\activate
3. pip install -r requirements.txt
4. python scripts/build_current_recommender.py
5. Open index.html, or deploy this folder to Vercel.

The generated data.js and web/data.js are used by the two static HTML entry points. The raw API response is saved under data/raw/ and should not be committed.

## Recommendation algorithm
1. Normalize current product records.
2. Construct TF-IDF text features from name, brand, category, quantity, ingredients and nutrition grade.
3. Calculate cosine similarity between products.
4. Apply Maximal Marginal Relevance (MMR, lambda 0.72) to balance similarity and diversity.
5. Export top-10 related products for each catalogue product.

Similarity scores are ranking signals, not probabilities. Because there is no 2026 behavioural dataset, do not report Recall@K, NDCG@K, conversion lift or purchase prediction as real-world measured results.

## Refresh
Run python scripts/build_current_recommender.py to fetch a new snapshot and rebuild recommendations. A weekly GitHub Actions workflow and manual workflow dispatch are included for reproducible refreshes. The workflow commits generated dashboard data, a bounded one-page source sample for auditability, and refresh-status reports; it does not download or commit the full Open Food Facts database.

## Limitations
- A product catalogue is not a transaction or clickstream dataset.
- Product fields can be incomplete, community-entered or stale.
- The snapshot is limited to one most-recently-modified API result page (up to 100 records) per refresh. It is not statistically representative of the full database.
- API sorting does not guarantee every recently updated product is represented in a small snapshot.
- Respect Open Food Facts API usage rules and ODbL obligations.
