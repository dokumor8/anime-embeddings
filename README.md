# Anime similarity score

Finding similar anime based on embedding distance.

## How it works:

### Preparation

1. Get anime review dataset - https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews
2. Compute review embeddings using `BAAI/bge-base-en-v1.5` - https://huggingface.co/BAAI/bge-base-en-v1.5
   1. Filter to ~4000 most popular titles.
   2. For shows with many reviews, use only the first 128.
3. Compute synopsis embeddings for the same shows.
4. Store them in two numpy matrices.
5. (Actually done in runtime) Get the weighted average

### Runtime

1. Select an anime.
2. Get its embeddings for reviews and for the synopsis.
3. Get 
4. Take the weighted average where the synopsis has weight 0.1 and the reviews have weight 0.9.
5. Average the numpy matrices 
