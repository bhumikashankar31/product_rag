# Product Catalogue RAG Assistant

## Overview

This project is a simple RAG-based question answering assistant built using a product catalogue in Markdown format.

The user can ask questions about the products, and the system retrieves the most relevant product information and passes it to Gemini to generate the answer.

## How It Works

The pipeline is:

Markdown catalogue
→ Product extraction
→ Product-level chunks
→ Embeddings
→ Cosine similarity
→ Top-3 relevant chunks
→ Gemini
→ Final answer

### 1. Product Extraction

The Markdown file is parsed using its headings.

- `##` represents a product category.
- `###` represents a product.
- The product description and features are stored along with the product name and category.

Each product is then converted into one chunk.

### 2. Embeddings

I used `all-MiniLM-L6-v2` from Sentence Transformers to convert each product chunk into a vector.

The user's question is also converted into a vector using the same model.

### 3. Retrieval

Cosine similarity is calculated between the user's query embedding and all product embeddings.

The 3 chunks with the highest similarity scores are selected as the relevant context.

I chose Top-3 as a simple balance between providing enough context and avoiding unnecessary information.

### 4. Generation

The retrieved chunks and the user's question are provided to Gemini.

The prompt instructs Gemini to answer using the provided context and not invent information when the answer is not available.

## Why I Chose This Approach

- I used product-level chunks because the catalogue already has a clear product structure.
- I used `all-MiniLM-L6-v2` because it is lightweight and suitable for a small retrieval task.
- I used cosine similarity to compare the query and product embeddings.
- I did not use a vector database because the current catalogue is very small and the embeddings can be handled in memory.

## Technologies Used

- Python
- Sentence Transformers
- Scikit-learn
- NumPy
- Google Gemini API
