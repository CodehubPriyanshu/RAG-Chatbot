"""
Streamlit UI for RAG Chatbot
Provides interactive interface with memory feature and monthly spending chart.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from chatbot import initialize_rag_system, retrieve_transactions, generate_answer
import json


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - Transaction Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'last_question' not in st.session_state:
    st.session_state.last_question = None
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'texts' not in st.session_state:
    st.session_state.texts = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None


def load_transaction_data():
    """Load transaction data from JSON file."""
    with open("transactions.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def create_monthly_spending_chart(data):
    """
    Create a bar chart showing monthly spending.
    
    Args:
        data: List of transaction dictionaries
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    
    # Group by month and sum amounts
    monthly_spending = df.groupby('month')['amount'].sum().reset_index()
    monthly_spending.columns = ['Month', 'Total Amount']
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(monthly_spending['Month'], monthly_spending['Total Amount'], 
           color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Spending (₹)', fontsize=12, fontweight='bold')
    ax.set_title('Monthly Spending Overview', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, v in enumerate(monthly_spending['Total Amount']):
        ax.text(i, v, f'₹{v:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# Initialize RAG system (only once)
@st.cache_resource
def initialize_rag():
    """Initialize RAG system with caching to avoid reloading."""
    return initialize_rag_system()


# Main UI
def main():
    st.title("💬 RAG Chatbot - Transaction Assistant")
    st.markdown("---")
    st.markdown("""
    **Welcome!** Ask me questions about customer transactions, spending patterns, and purchase history.
    
    **Example queries:**
    - "Show me Riya's purchase history"
    - "What is Amit's total spending?"
    - "List all February transactions"
    """)
    
    # Initialize RAG system
    if not st.session_state.rag_initialized:
        with st.spinner("Loading transaction data and initializing AI model..."):
            st.session_state.model, st.session_state.embeddings, \
            st.session_state.texts, st.session_state.raw_data = initialize_rag()
            st.session_state.rag_initialized = True
        st.success("✅ System ready!")
    
    st.markdown("---")
    
    # Memory feature: Show last question
    if st.session_state.last_question:
        st.info(f"💭 **Last Question:** {st.session_state.last_question}")
        if st.button("🔄 Use Last Question"):
            query = st.session_state.last_question
        else:
            query = None
    else:
        query = None
    
    # Input section
    st.subheader("Ask a Question")
    user_query = st.text_input(
        "Enter your question:",
        value=query if query else "",
        placeholder="e.g., What is Amit's total spending?",
        key="user_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and user_query:
        st.session_state.last_question = user_query
        
        with st.spinner("Searching transaction data..."):
            # Retrieve relevant transactions
            context = retrieve_transactions(
                user_query,
                st.session_state.model,
                st.session_state.embeddings,
                st.session_state.texts,
                top_k=3
            )
            
            # Generate answer
            answer = generate_answer(
                user_query,
                context,
                st.session_state.raw_data
            )
        
        # Display answer
        st.markdown("---")
        st.subheader("🤖 Chatbot Response")
        st.markdown(f"**Question:** {user_query}")
        st.markdown(f"**Answer:**")
        st.success(answer)
        
        # Show retrieved context (optional, for debugging/transparency)
        with st.expander("📋 View Retrieved Context"):
            st.write("Top relevant transactions:")
            for i, (text, score) in enumerate(context, 1):
                st.write(f"{i}. {text} (similarity: {score:.3f})")
    
    st.markdown("---")
    
    # Monthly spending chart
    st.subheader("📊 Monthly Spending Chart")
    transaction_data = load_transaction_data()
    chart_fig = create_monthly_spending_chart(transaction_data)
    st.pyplot(chart_fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>RAG Chatbot - Powered by SentenceTransformers & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

