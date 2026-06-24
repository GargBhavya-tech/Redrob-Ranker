"""
Job Description configuration for the Redrob Hackathon ranker.

Encodes the "Senior AI Engineer — Founding Team" JD (Redrob AI, Pune/Noida)
as structured targets for the scoring pipeline.
"""

# Free-text JD used for semantic embedding (condensed from job_description.docx,
# focused on the technical/role substance rather than culture/admin sections).
JD_TEXT = """
Senior AI Engineer, Founding Team, at an AI-native talent intelligence
platform (Series A). Owns the intelligence layer of the product: ranking,
retrieval, and matching systems for candidate-job matching at scale.

Core responsibilities: audit and improve an existing BM25 + rule-based
ranking system; ship a v2 ranking system using embeddings, hybrid retrieval
(lexical + dense vector + structured), and LLM-based re-ranking; build
evaluation infrastructure including offline benchmarks (NDCG, MRR, MAP,
recall@k), online A/B testing, and recruiter feedback loops; drive long-term
architecture for candidate-JD matching at scale.

Must-have experience: production experience with embeddings-based retrieval
systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real
users, including embedding drift, index refresh, and retrieval-quality
regression; production experience with vector databases or hybrid search
infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, FAISS); strong Python with high code quality; hands-on
evaluation framework design for ranking systems (NDCG, MRR, MAP,
offline-to-online correlation, A/B test interpretation).

Nice-to-have: LLM fine-tuning (LoRA, QLoRA, PEFT); learning-to-rank models
(XGBoost-based or neural); HR-tech / recruiting tech / marketplace product
background; distributed systems or large-scale inference optimization;
open-source contributions in AI/ML.

Seniority: 5-9 years total experience, with 4-5+ years in applied ML/AI
roles at product companies (not pure consulting/services). Has shipped at
least one end-to-end ranking, search, or recommendation system to real users
at meaningful scale. Strong opinions on retrieval (hybrid vs dense),
evaluation (offline vs online), and LLM integration (fine-tune vs prompt),
defensible with reference to systems actually built.

Location: Pune or Noida, India preferred; open to candidates in Hyderabad,
Mumbai, Delhi NCR willing to relocate. No visa sponsorship outside India.
"""


# Expanded query used ONLY for the Stage-2 semantic similarity step. Mixes
# plain-language descriptions (so "plain-language Tier-5" candidates who never
# use buzzwords still match) with the buzzword synonyms (so keyword profiles
# match too). Measured on the human anchor: expanded query lifts tier<->rank
# correlation 0.73 -> 0.78 before recent-role weighting, 0.82 combined.
JD_SEMANTIC_QUERY = (
    "senior ai engineer building production ranking retrieval and search systems, "
    "recommendation systems, semantic search, learning to rank, embeddings based "
    "retrieval, vector search, hybrid search, relevance, evaluation with ndcg mrr "
    "map, shipped to real users at meaningful scale. "
    "faiss pinecone weaviate qdrant milvus elasticsearch opensearch bm25 bge e5 "
    "sentence-transformers rag dense retrieval reranking nearest neighbor hnsw recsys "
    "collaborative filtering matching pipeline candidate ranking click-through "
    "relevance labeling a/b testing offline online evaluation correlation"
)

# --- Structured targets used for the rule-based / structural signals ---

# Title tokens that indicate strong alignment with the AI Engineer / ML /
# Search / Retrieval role family. Used for S_title (Jaccard-style lexical
# match), since dense embeddings can conflate hierarchically different titles.
TARGET_TITLE_TOKENS = {
    "ai", "ml", "machine", "learning", "engineer", "scientist",
    "search", "retrieval", "ranking", "recommendation", "nlp",
    "llm", "applied", "research", "data", "founding",
}

# "Core" AI/ML skills directly named as must-haves in the JD. Heavy weight.
CORE_SKILLS = {
    "embeddings", "sentence-transformers", "sentence transformers",
    "openai embeddings", "bge", "e5", "vector database", "vector databases",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "hybrid search", "hybrid retrieval",
    "ndcg", "mrr", "map", "learning to rank", "learning-to-rank",
    "ltr", "xgboost", "lightgbm", "a/b testing", "ab testing",
    "retrieval", "ranking", "semantic search", "rag",
    "retrieval augmented generation", "bm25", "re-ranking", "reranking",
}

# Nice-to-have / secondary skills.
SECONDARY_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "llm fine-tuning", "distributed systems", "kubernetes",
    "fastapi", "python", "pytorch", "tensorflow", "transformers",
    "nlp", "deep learning", "recommendation systems", "spark",
}

# Generic "AI keyword stuffer" terms — high density of these alone, without
# title/career alignment, is the trap the JD explicitly warns about.
AI_KEYWORD_TERMS = CORE_SKILLS | SECONDARY_SKILLS | {
    "ai", "artificial intelligence", "machine learning", "chatgpt",
    "langchain", "llamaindex", "prompt engineering", "generative ai",
    "gpt", "claude", "gemini",
}

# Required years of experience (lower bound of JD's 5-9 yr band).
REQUIRED_YOE = 5.0

# Target locations (Indian metros emphasized by the JD).
TARGET_LOCATIONS = {
    "pune": 1.0,
    "noida": 1.0,
    "hyderabad": 0.8,
    "mumbai": 0.8,
    "delhi": 0.8,
    "delhi ncr": 0.8,
    "ncr": 0.8,
    "gurugram": 0.8,
    "gurgaon": 0.8,
    "bangalore": 0.6,
    "bengaluru": 0.6,
}

# Job titles considered "non-fit" regardless of skill list — used in the
# keyword-stuffer trap check (JD explicitly calls out e.g. "Marketing
# Manager" with a perfect AI skill list as a non-fit).
NON_FIT_TITLE_KEYWORDS = {
    "marketing", "sales", "hr", "human resources", "recruiter",
    "graphic designer", "content writer", "business analyst",
    "accountant", "finance", "operations manager", "office manager",
    "customer support", "customer service", "designer",
}

# Pure-services companies the JD explicitly is wary of as sole background.
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "ltimindtree", "l&t infotech",
}


# --- Added for score_evidence (ported from eval harness) ----------------------
# Regex patterns over career-description prose that signal the candidate actually
# BUILT a retrieval/ranking/recommendation system (vs. merely listing keywords).
BUILD_EVIDENCE_PATTERNS = [
    r"\bbuilt\b.{0,50}(search|ranking|recommendation|retrieval|matching|relevance)",
    r"(shipped|rebuilt|migrated).{0,50}(search|ranking|retrieval|embedding|recommendation)",
    r"(owned|designed|architected).{0,50}(ranking|retrieval|relevance|search|recommendation)",
    r"(rag|dense retrieval|hybrid retrieval|hybrid search).{0,60}(production|serving|queries|users|scale)",
    r"learning[- ]to[- ]rank|lambdamart|\bndcg\b",
    r"(bm25|bge|faiss|hnsw|pinecone|weaviate|qdrant|milvus|opensearch|elasticsearch).{0,60}(retriev|rank|embed|search|serving|production)",
    r"recommendation system",
    r"led the team .{0,40}(retrieval|ranking|search|embedding)",
]

OFF_DOMAIN_TERMS = [
    "computer vision", "image classification", "object detection", "opencv",
    "yolo", "speech recognition", "asr", "robotics", "lidar",
]

RESEARCH_ONLY_TERMS = [
    "research scientist", "phd researcher", "postdoc", "research fellow",
    "academic", "publication", "thesis",
]
