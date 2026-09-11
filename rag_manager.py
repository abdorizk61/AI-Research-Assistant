"""
rag_manager.py
===============

Step 2 of the AI Research Assistant project: the RAG backbone.

Scope of this file:
    - Document chunking (LangChain's RecursiveCharacterTextSplitter)
    - A local ChromaDB vector store
    - Local embeddings via Ollama (no OpenAI / cloud calls)
    - Similarity-based retrieval

Explicitly OUT of scope for this step (coming later):
    - Prompt construction for the LLM
    - Calling the Ollama chat/generation model
    - The "I don't have enough information" fallback logic
      (that will live in the generation step, using the *scores*
      returned here to decide when to bail out)

Design notes:
    - We use ChromaDB's native `chromadb.utils.embedding_functions`
      OllamaEmbeddingFunction instead of `langchain-ollama`'s wrapper.
      This keeps our dependency footprint smaller (no need to pull in
      the full LangChain vectorstore integration) while still using
      LangChain purely for its excellent text splitters, as requested.
    - `PersistentClient` is used so the indexed knowledge base survives
      across app restarts. Switch to `chromadb.EphemeralClient()` for a
      throwaway in-memory store during quick testing.
    - Every chunk stored gets a metadata dict with the source URL and
      a chunk index, so the report/answer step can later cite sources.
"""

import uuid
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGManager")


class RAGManager:
    """
    Encapsulates the whole retrieval side of the RAG pipeline:
    chunking -> embedding -> storing -> similarity search.

    Usage:
        rag = RAGManager()
        rag.add_documents(scraped_text, source_url="https://example.com")
        results = rag.retrieve_context("What is X?", k=3)
    """

    def __init__(
        self,
        collection_name: str = "research_assistant_kb",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        use_persistent_storage: bool = True,
    ):
        """
        Args:
            collection_name: Name of the ChromaDB collection to use/create.
            persist_directory: Where to store the DB on disk (ignored if
                use_persistent_storage=False).
            embedding_model: Local Ollama embedding model tag. Must already
                be pulled, e.g. `ollama pull nomic-embed-text`.
            ollama_base_url: Base URL of the local Ollama server.
            chunk_size: Max characters per chunk (tune per embedding model).
            chunk_overlap: Character overlap between consecutive chunks,
                helps preserve context across chunk boundaries.
            use_persistent_storage: True -> data survives restarts.
                False -> ephemeral, in-memory only (handy for unit tests).
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # --- 1. Text splitter (chunking) ---
        # RecursiveCharacterTextSplitter tries to split on paragraph/sentence
        # boundaries first, falling back to smaller separators, so chunks
        # stay semantically coherent rather than cutting mid-sentence.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # --- 2. Embedding function (local, via Ollama) ---
        try:
            self.embedding_function = OllamaEmbeddingFunction(
                url=ollama_base_url,
                model_name=embedding_model,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embedding function: {e}")
            raise RuntimeError(
                f"Could not connect to Ollama embeddings at {ollama_base_url}. "
                f"Make sure Ollama is running and '{embedding_model}' is pulled "
                f"(`ollama pull {embedding_model}`)."
            ) from e

        # --- 3. Chroma client (persistent by default) ---
        if use_persistent_storage:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        # --- 4. Collection (created if it doesn't exist yet) ---
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "AI Research Assistant knowledge base"},
        )

        logger.info(
            f"RAGManager ready — collection='{self.collection_name}', "
            f"embedding_model='{self.embedding_model}', "
            f"existing_chunks={self.collection.count()}"
        )

    # ------------------------------------------------------------------
    # CORE FUNCTION 1: Chunk + store documents
    # ------------------------------------------------------------------
    def add_documents(self, text: str, source_url: str = "manual_input") -> int:
        """
        Splits `text` into chunks and stores them in ChromaDB, tagging
        each chunk with the source URL as metadata.

        Args:
            text: The raw scraped/typed text to index.
            source_url: Where this text came from (used later for
                citations and for letting users see what was indexed).

        Returns:
            The number of chunks that were added.

        Raises:
            ValueError: if `text` is empty/whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot add empty document text to the knowledge base.")

        chunks = self.text_splitter.split_text(text)

        if not chunks:
            logger.warning(f"Text splitter produced 0 chunks for source: {source_url}")
            return 0

        # Chroma requires: unique ids, list of documents, list of metadatas
        ids = [f"{uuid.uuid4()}" for _ in chunks]
        metadatas: List[Dict[str, Any]] = [
            {
                "source": source_url,
                "chunk_index": i,
                "chunk_count": len(chunks),
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Added {len(chunks)} chunk(s) from source: {source_url}")
        return len(chunks)

    # ------------------------------------------------------------------
    # CORE FUNCTION 2: Similarity retrieval
    # ------------------------------------------------------------------
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs similarity search over the vector store and returns the
        top-k most relevant chunks, along with their metadata and
        distance score (lower distance = more similar).

        Args:
            query: The user's natural-language question.
            k: How many chunks to retrieve.

        Returns:
            A list of dicts shaped like:
                {
                    "text": "<chunk content>",
                    "source": "<source_url>",
                    "distance": 0.1234,   # lower = more similar
                }
            Returns an empty list if the knowledge base is empty or the
            query is invalid. NOTE: deciding whether these results are
            "good enough" (the confidence threshold behind the "I don't
            have enough information" fallback) will be handled in the
            generation step — this function only returns raw results.
        """
        if not query or not query.strip():
            logger.warning("retrieve_context called with an empty query.")
            return []

        if self.collection.count() == 0:
            logger.warning("retrieve_context called but the knowledge base is empty.")
            return []

        # Never request more results than exist in the collection.
        safe_k = min(k, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=safe_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted_results.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "distance": dist,
                }
            )

        return formatted_results

    # ------------------------------------------------------------------
    # Small utility helpers (not required, but handy for the UI)
    # ------------------------------------------------------------------
    def get_indexed_source_count(self) -> int:
        """Total number of chunks currently stored (useful for UI status)."""
        return self.collection.count()

    def reset_knowledge_base(self) -> None:
        """Deletes and recreates the collection. Use with caution."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        logger.info("Knowledge base has been reset.")


# ----------------------------------------------------------------------
# Quick manual test (only runs if you execute this file directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rag = RAGManager(use_persistent_storage=False)  # ephemeral for a quick smoke test

    sample_text = (
        "Renewable energy sources like solar and wind power are becoming "
        "increasingly cost-competitive with fossil fuels. Battery storage "
        "technology is a key bottleneck for grid-scale adoption. "
        "Recent advances in solid-state batteries promise higher energy "
        "density and improved safety compared to traditional lithium-ion cells."
    )

    added = rag.add_documents(sample_text, source_url="https://example.com/energy-article")
    print(f"Added {added} chunks.")

    hits = rag.retrieve_context("What is limiting renewable energy adoption?", k=2)
    for i, hit in enumerate(hits, 1):
        print(f"\n--- Result {i} (distance={hit['distance']:.4f}) ---")
        print(f"Source: {hit['source']}")
        print(hit["text"])
