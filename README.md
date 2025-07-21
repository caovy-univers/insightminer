# InsightMiner: Academic Document QA and Content Retrieval Using Data Science Techniques

**Objective**:

To build a retrieval-based QA system that parses academic PDFs, embeds the content, and retrieves relevant segments in response to user queries.

**Goals**:

- Extract and chunk text from PDFs
- Generate and index text embeddings (using FAISS)
- Perform document search using similarity scores
- Evaluate retrieval precision or answer relevance
- Optional: summarization using extractive or LLM-based methods
- Datasets: [ArXiv Academic Metadata](https://www.kaggle.com/datasets/Cornell-University/arxiv), such as titles, authors, abstracts, categories

InsightMiner Project: Step-by-Step Plan

Step 1️⃣ Preprocess and Combine Text Method:
pandas string operations

Combining title + abstract → combined_text

Why?
 You want one clean text field per paper that fully represents its content for embedding and search. Abstract alone might miss some keywords from the title.

Step 2️⃣ Generate Text Embeddings
Method/Technique:
Sentence Embedding using Sentence Transformers

Model example: all-MiniLM-L6-v2

Why?
 Embeddings convert text into numerical vectors that capture meaning. This allows you to compare research papers and user queries based on semantic similarity, rather than keyword matching.

Step 3️⃣ Build Vector Index
Method/Technique:
FAISS (Facebook AI Similarity Search)

Why?
 You can't search embeddings directly using pandas — FAISS allows fast, scalable similarity searches across thousands of vectors, critical for finding relevant papers in real time.

Step 4️⃣ Build Document Retrieval Function
Method:
Vector search query → Top-k results

Cosine similarity or L2 distance

Why?
 You need a way to retrieve the most relevant papers given a user question. This is the core search functionality that powers your academic assistant.

Step 5️⃣ Add Category-Based Filtering
Method/Technique:
pandas filtering using df['categories']

Streamlit UI dropdown

Why?
 Letting users filter by research area (e.g., only AI papers) improves usability and control, especially with a large dataset.

Step 6️⃣ Implement Keyword Extraction
Method/Technique:
KeyBERT or YAKE

Extract keywords from combined_text

Why?
 Keyword highlights make search results easier to understand and scan quickly. This adds a valuable feature beyond just showing plain abstracts.

Step 7️⃣ Add Topic Modeling
Method/Technique:
BERTopic (embedding-based topic clustering)

LDA (alternative)

Why?
 To automatically group similar papers together by topic, helping users explore large sets of research papers more efficiently.

Step 8️⃣ Build User Interface
Method/Technique:
Streamlit Web App

Why?
 Allows non-technical users to:
Enter search queries

Select categories

View results with authors, keywords, topics

It makes the system usable beyond scripts and notebooks.


Step 9️⃣ Evaluate and Test
Method/Technique:
Manual result checking

Cosine similarity score analysis

t-SNE/UMAP visualization (for embeddings or topics)

Why?
 To validate that the system is returning relevant, meaningful results — this step ensures quality before presentation or deployment.
