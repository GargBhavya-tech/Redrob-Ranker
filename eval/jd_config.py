"""
jd_config.py
------------
All JD-specific knowledge lives here, distilled from job_description.docx.

The whole point of the challenge is reasoning about the GAP between what the JD
*says* (keywords) and what it *means* (production retrieval/ranking experience at
a product company, with seniority and availability). Every constant below is a
direct, defensible reading of a sentence in the JD -- keep it that way so you can
explain each one in the Stage-5 interview.
"""

# A compact natural-language version of the JD used as the query for semantic
# similarity. We deliberately phrase it the way a STRONG candidate would describe
# their own work, so plain-language Tier-5 profiles (who never say "RAG") still
# match on the *meaning*.
JD_QUERY = (
    "Senior AI engineer who owns the intelligence layer of a product: ranking, "
    "retrieval and matching systems deployed to real users at scale. Production "
    "experience with embeddings-based retrieval (sentence-transformers, BGE, E5, "
    "OpenAI embeddings), vector databases and hybrid search (FAISS, Pinecone, "
    "Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch), and designing evaluation "
    "frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, "
    "A/B testing). Has shipped at least one end-to-end search, ranking or "
    "recommendation system to real users. Strong Python and software engineering. "
    "Pragmatic shipper at a product company, not a pure researcher, not "
    "framework-only, not consulting-only. Based in or willing to relocate to "
    "Pune or Noida, India, and currently active and available to talk."
)

# --- Skill / evidence vocabularies (lowercase, substring-matched) -------------
# "Core" = the retrieval/ranking/eval competencies the JD says you ABSOLUTELY need.
CORE_RETRIEVAL = {
    "rag", "retrieval", "retrieval-augmented", "embedding", "embeddings", "vector",
    "vector database", "vector search", "pinecone", "weaviate", "qdrant", "milvus",
    "faiss", "elasticsearch", "opensearch", "bm25", "semantic search",
    "sentence-transformers", "bge", "e5", "learning to rank", "learning-to-rank",
    "ltr", "lambdamart", "ranking", "re-ranking", "reranking", "recommendation",
    "recommender", "ndcg", "mrr", "information retrieval", "hybrid search",
    "dense retrieval", "nearest neighbor", "ann", "hnsw",
}

# "General ML" = nice signal but the JD warns it is NOT sufficient on its own.
ML_GENERAL = {
    "nlp", "llm", "large language model", "fine-tuning", "fine-tune", "lora",
    "qlora", "peft", "transformers", "pytorch", "tensorflow", "xgboost",
    "lightgbm", "machine learning", "deep learning", "hugging face", "huggingface",
    "mlops", "feature engineering", "model deployment",
}

# Evidence that the candidate actually BUILT a system (used to detect real
# production work in career descriptions, regardless of skill keywords).
BUILD_EVIDENCE = {
    "recommendation", "recommender", "search", "ranking", "retrieval", "relevance",
    "matching system", "personalization", "personalized", "embeddings", "vector",
    "a/b test", "ndcg", "recommend", "ranker",
}

# --- Disqualifiers / down-weights (explicit in the JD) ------------------------
# "People who have ONLY worked at consulting firms ... in their entire career."
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "mphasis", "hcl", "ltimindtree", "tech mahindra",
}

# "primary expertise is computer vision, speech, or robotics" without NLP/IR
OFF_DOMAIN = {
    "computer vision", "image classification", "object detection", "opencv",
    "yolo", "speech recognition", "asr", "tts", "robotics", "ros", "slam",
    "autonomous", "lidar",
}

# Pure-research signal (academic/research-only, no production)
RESEARCH_ONLY = {
    "research scientist", "phd researcher", "postdoc", "research fellow",
    "academic", "publication", "paper", "thesis",
}

# Titles that, on their own, indicate a NON-engineering role (keyword-stuffer trap)
NON_TECH_TITLE_HINTS = {
    "hr manager", "marketing manager", "sales executive", "accountant",
    "business analyst", "operations manager", "customer support", "content writer",
    "graphic designer", "project manager", "civil engineer", "mechanical engineer",
    "recruiter", "office manager",
}

TECH_TITLE_HINTS = {
    "engineer", "developer", "scientist", "ml", "ai", "machine learning",
    "data", "research", "architect", "nlp",
}

# Target locations (JD: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR, Bangalore welcome)
TARGET_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
    "gurgaon", "gurugram", "delhi ncr", "ncr",
}

# JD experience band ("5-9 years ... a range, not a requirement")
YOE_IDEAL_LOW = 5.0
YOE_IDEAL_HIGH = 9.0
YOE_HARD_LOW = 3.0   # below this the JD's "senior" bar is very hard to meet

# --- Scoring weights for the structured component (sum need not be 1; the final
#     fusion is rank-based via RRF, so these only set the *ordering* of the
#     structured signal). Tunable; documented for the interview. -------------
W_RETRIEVAL_EVIDENCE = 3.0   # strongest single signal: built retrieval/ranking
W_CORE_SKILLS        = 1.5
W_SENIORITY          = 1.5
W_PRODUCT_COMPANY    = 1.5
W_YOE_FIT            = 1.0
W_ML_GENERAL         = 0.5
W_LOCATION           = 1.0

# Penalties (subtracted from structured score)
P_CONSULTING_ONLY    = 4.0
P_OFF_DOMAIN_ONLY    = 4.0
P_RESEARCH_ONLY      = 3.0
P_KEYWORD_STUFFER    = 6.0   # non-tech title + AI skills + no build evidence
P_TITLE_CHASER       = 1.5   # avg tenure < 18 months across history

# Behavioral multiplier bounds (applied multiplicatively to fused relevance)
BEHAVIOR_MIN = 0.55
BEHAVIOR_MAX = 1.10
