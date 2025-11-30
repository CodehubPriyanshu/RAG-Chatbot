"""
RAG Chatbot Core Logic
Handles data loading, embedding creation, retrieval, and answer generation.
"""

import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare(json_path: str = "transactions.json") -> Tuple[List[Dict], List[str]]:
    """
    Load transaction data from JSON and convert to descriptive text strings.
    
    Args:
        json_path: Path to the transactions JSON file
        
    Returns:
        Tuple of (raw_data, text_descriptions)
        - raw_data: List of original transaction dictionaries
        - text_descriptions: List of formatted transaction strings
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        text_descriptions = []
        for transaction in raw_data:
            # Format: "On 2024-01-12, Amit purchased a Laptop for 55000."
            text = f"On {transaction['date']}, {transaction['customer']} purchased a {transaction['product']} for {transaction['amount']}."
            text_descriptions.append(text)
        
        return raw_data, text_descriptions
    except FileNotFoundError:
        logger.error(f"Transaction data file not found: {json_path}")
        return [], []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {json_path}: {e}")
        return [], []
    except Exception as e:
        logger.error(f"Unexpected error loading transaction data: {e}")
        return [], []

def create_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Create embeddings for transaction texts using SentenceTransformer.
    
    Args:
        texts: List of text descriptions to embed
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        Tuple of (model, embeddings)
        - model: The loaded SentenceTransformer model
        - embeddings: numpy array of embeddings (n_samples, embedding_dim)
    """
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return model, embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings with model {model_name}: {e}")
        raise

def retrieve_transactions(query: str, model: SentenceTransformer, embeddings: np.ndarray, 
                         texts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Retrieve top-k most relevant transactions using cosine similarity.
    
    Args:
        query: User's question/query string
        model: SentenceTransformer model for encoding queries
        embeddings: Pre-computed embeddings for all transactions
        texts: List of transaction text descriptions
        top_k: Number of top results to return
        
    Returns:
        List of tuples (text, similarity_score) sorted by relevance
    """
    try:
        # Validate inputs
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to retrieve_transactions")
            return []
            
        if len(embeddings) == 0 or len(texts) == 0:
            logger.warning("Empty embeddings or texts provided to retrieve_transactions")
            return []
        
        # Encode the query
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
        
        # Return top-k texts with their similarity scores
        results = [(texts[idx], float(similarities[idx])) for idx in top_indices]
        return results
    except Exception as e:
        logger.error(f"Error retrieving transactions: {e}")
        return []

def generate_answer(query: str, context: List[Tuple[str, float]], raw_data: List[Dict]) -> str:
    """
    Generate an answer based on the query and retrieved context.
    Handles specific query types like total spending and customer history.
    
    Args:
        query: User's question
        context: List of (text, similarity_score) tuples from retrieval
        raw_data: Original transaction data for calculations
        
    Returns:
        Generated answer string
    """
    try:
        if not query or not isinstance(query, str):
            return "Invalid query provided."
            
        query_lower = query.lower()
        
        # Extract customer name if mentioned
        customer_name = None
        for transaction in raw_data:
            if transaction['customer'].lower() in query_lower:
                customer_name = transaction['customer']
                break
        
        # Handle total spending queries
        if "total spending" in query_lower or "total spent" in query_lower or "total amount" in query_lower:
            if customer_name:
                total = sum(t['amount'] for t in raw_data if t['customer'] == customer_name)
                transactions = [t for t in raw_data if t['customer'] == customer_name]
                return f"{customer_name}'s total spending is ₹{total}. They made {len(transactions)} transaction(s):\n" + \
                       "\n".join([f"- {t['product']} for ₹{t['amount']} on {t['date']}" for t in transactions])
            else:
                total = sum(t['amount'] for t in raw_data)
                return f"Total spending across all customers is ₹{total}."
        
        # Handle purchase history queries
        if "purchase history" in query_lower or "purchases" in query_lower or "bought" in query_lower:
            if customer_name:
                transactions = [t for t in raw_data if t['customer'] == customer_name]
                if transactions:
                    result = f"{customer_name}'s purchase history:\n"
                    for t in transactions:
                        result += f"- On {t['date']}, purchased {t['product']} for ₹{t['amount']}\n"
                    return result.strip()
                else:
                    return f"No purchases found for {customer_name}."
            else:
                return "Please specify a customer name to view their purchase history."
        
        # Handle date/month filtering queries
        if "february" in query_lower or "feb" in query_lower:
            feb_transactions = [t for t in raw_data if "2024-02" in t['date']]
            if feb_transactions:
                result = "February 2024 transactions:\n"
                for t in feb_transactions:
                    result += f"- {t['customer']} purchased {t['product']} for ₹{t['amount']} on {t['date']}\n"
                return result.strip()
            else:
                return "No transactions found for February 2024."
        
        if "january" in query_lower or "jan" in query_lower:
            jan_transactions = [t for t in raw_data if "2024-01" in t['date']]
            if jan_transactions:
                result = "January 2024 transactions:\n"
                for t in jan_transactions:
                    result += f"- {t['customer']} purchased {t['product']} for ₹{t['amount']} on {t['date']}\n"
                return result.strip()
            else:
                return "No transactions found for January 2024."
        
        if "march" in query_lower or "mar" in query_lower:
            mar_transactions = [t for t in raw_data if "2024-03" in t['date']]
            if mar_transactions:
                result = "March 2024 transactions:\n"
                for t in mar_transactions:
                    result += f"- {t['customer']} purchased {t['product']} for ₹{t['amount']} on {t['date']}\n"
                return result.strip()
            else:
                return "No transactions found for March 2024."
        
        # Default: Use retrieved context to generate answer
        if context:
            result = "Based on the transaction data:\n\n"
            for text, score in context:
                result += f"- {text}\n"
            return result.strip()
        else:
            return "I couldn't find relevant information to answer your question. Please try rephrasing."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "An error occurred while generating the answer. Please try again."

def initialize_rag_system(json_path: str = "transactions.json") -> Tuple[SentenceTransformer, np.ndarray, List[str], List[Dict]]:
    """
    Initialize the complete RAG system by loading data and creating embeddings.
    
    Args:
        json_path: Path to the transactions JSON file
        
    Returns:
        Tuple of (model, embeddings, texts, raw_data)
        - model: SentenceTransformer model
        - embeddings: Transaction embeddings
        - texts: Transaction text descriptions
        - raw_data: Original transaction data
    """
    try:
        raw_data, texts = load_and_prepare(json_path)
        if not raw_data:
            raise ValueError("Failed to load transaction data")
            
        model, embeddings = create_embeddings(texts)
        return model, embeddings, texts, raw_data
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        raise