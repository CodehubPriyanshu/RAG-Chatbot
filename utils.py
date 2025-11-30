"""
Utility functions for the RAG Chatbot project.
Helper functions for data processing and formatting.
"""

import json
from typing import List, Dict
from datetime import datetime


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
    return [t for t in data if t['customer'].lower() == customer_name.lower()]


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
    filtered = []
    for transaction in data:
        date_obj = datetime.strptime(transaction['date'], "%Y-%m-%d")
        if date_obj.year == year and date_obj.month == month:
            filtered.append(transaction)
    return filtered


def calculate_total_spending(data: List[Dict], customer_name: str = None) -> int:
    """
    Calculate total spending, optionally for a specific customer.
    
    Args:
        data: List of transaction dictionaries
        customer_name: Optional customer name to filter by
        
    Returns:
        Total spending amount
    """
    if customer_name:
        filtered = filter_transactions_by_customer(data, customer_name)
        return sum(t['amount'] for t in filtered)
    return sum(t['amount'] for t in data)


def get_customer_list(data: List[Dict]) -> List[str]:
    """
    Get unique list of customer names from transaction data.
    
    Args:
        data: List of transaction dictionaries
        
    Returns:
        List of unique customer names
    """
    return sorted(list(set(t['customer'] for t in data)))

