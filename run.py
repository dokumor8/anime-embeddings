from FlagEmbedding import FlagModel
sentences_1 = ["action movie", "explosions are so good, bam, bam, this is fun"]
sentences_2 = ["super cool movie", "I'm a good guy, so I like very cools movies"]
model = FlagModel("./model", use_fp16=True)
# Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1, convert_to_numpy=True)
embeddings_2 = model.encode(sentences_2, convert_to_numpy=True)
embeddings_1_mean = embeddings_1.mean(axis=0)
embeddings_2_mean = embeddings_2.mean(axis=0)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
similarity_mean = embeddings_1_mean @ embeddings_2_mean.T
print(similarity_mean)

# # for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# # corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
# queries = ['query_1', 'query_2']
# passages = ["样例文档-1", "样例文档-2"]
# q_embeddings = model.encode_queries(queries)
# p_embeddings = model.encode(passages)
# scores = q_embeddings @ p_embeddings.T
