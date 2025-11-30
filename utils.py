"""
Utility functions for the RAG Chatbot project.
Helper functions for data processing and formatting.
"""

import json
import logging
from typing import List, Dict
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_currency(amount: int) -> str:
    """
    Format amount as Indian Rupee currency string.
    
    Args:
        amount: Numeric amount
        
    Returns:
        Formatted currency string (e.g., "₹55,000")
    """
    return f"₹{amount:,}"

def filter_transactions_by_customer(data: List[Dict], customer_name: str) -> List[Dict]:
    """
    Filter transactions by customer name (case-insensitive).
    
    Args:
        data: List of transaction dictionaries
        customer_name: Name of the customer to filter by
        
    Returns:
        Filtered list of transactions
    """
    try:
        return [t for t in data if t['customer'].lower() == customer_name.lower()]
    except Exception as e:
        logger.error(f"Error filtering transactions by customer: {e}")
        return []

def filter_transactions_by_month(data: List[Dict], year: int, month: int) -> List[Dict]:
    """
    Filter transactions by year and month.
    
    Args:
        data: List of transaction dictionaries
        year: Year to filter (e.g., 2024)
        month: Month to filter (1-12)
        
    Returns:
        Filtered list of transactions
    """
    try:
        filtered = []
        for transaction in data:
            try:
                date_obj = datetime.strptime(transaction['date'], "%Y-%m-%d")
                if date_obj.year == year and date_obj.month == month:
                    filtered.append(transaction)
            except ValueError as ve:
                logger.warning(f"Invalid date format in transaction {transaction.get('id', 'unknown')}: {ve}")
                continue
        return filtered
    except Exception as e:
        logger.error(f"Error filtering transactions by month: {e}")
        return []

def calculate_total_spending(data: List[Dict], customer_name: str = None) -> int:
    """
    Calculate total spending, optionally for a specific customer.
    
    Args:
        data: List of transaction dictionaries
        customer_name: Optional customer name to filter by
        
    Returns:
        Total spending amount
    """
    try:
        if customer_name:
            filtered = filter_transactions_by_customer(data, customer_name)
            return sum(t['amount'] for t in filtered)
        return sum(t['amount'] for t in data)
    except Exception as e:
        logger.error(f"Error calculating total spending: {e}")
        return 0

def get_customer_list(data: List[Dict]) -> List[str]:
    """
    Get unique list of customer names from transaction data.
    
    Args:
        data: List of transaction dictionaries
        
    Returns:
        List of unique customer names
    """
    try:
        return sorted(list(set(t['customer'] for t in data)))
    except Exception as e:
        logger.error(f"Error getting customer list: {e}")
        return []