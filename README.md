# RAG Chatbot - Transaction Assistant

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with Python that answers questions about customer transactional data. The chatbot uses semantic search to retrieve relevant transactions and generates accurate answers based solely on the provided data.

## 🎯 Project Overview

This RAG chatbot system allows users to query customer transaction data using natural language. It leverages:
- **SentenceTransformers** for creating semantic embeddings
- **Cosine similarity** for retrieving relevant transactions
- **Streamlit** for an interactive web interface
- **Context-aware answer generation** that prevents hallucination

## ✨ Features

1. **Natural Language Querying**: Ask questions in plain English about transactions
2. **Semantic Search**: Uses advanced embeddings to find relevant transactions
3. **Accurate Answers**: Generates answers based only on retrieved transaction data
4. **Memory Feature**: "Show my last question" functionality
5. **Monthly Spending Chart**: Visual representation of spending patterns
6. **Clean UI**: Modern, user-friendly Streamlit interface

## 📁 Project Structure

```
rag_chatbot/
│── transactions.json      # Sample transaction data
│── chatbot.py            # Core RAG logic (embeddings, retrieval, generation)
│── app.py                # Streamlit UI application
│── utils.py              # Helper utility functions
│── requirements.txt      # Python dependencies
│── README.md            # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: The first run will download the SentenceTransformer model (`all-MiniLM-L6-v2`), which may take a few minutes.

## 🏃 How to Run

1. **Activate your virtual environment** (if using one)

2. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

4. **Start asking questions!**
   - Try: "Show me Riya's purchase history"
   - Try: "What is Amit's total spending?"
   - Try: "List all February transactions"

## 🔄 RAG Workflow

The chatbot follows this RAG (Retrieval-Augmented Generation) workflow:

1. **Data Loading**: Transactions are loaded from `transactions.json`
2. **Text Preprocessing**: Each transaction is converted to a descriptive string
   - Example: "On 2024-01-12, Amit purchased a Laptop for 55000."
3. **Embedding Creation**: All transaction texts are converted to embeddings using SentenceTransformer
4. **Query Processing**: User's question is converted to an embedding
5. **Retrieval**: Cosine similarity finds the top-k most relevant transactions
6. **Answer Generation**: The chatbot generates an answer using only the retrieved context

### Why RAG?

- **Prevents Hallucination**: Answers are based only on actual transaction data
- **Accurate**: No made-up values or incorrect information
- **Transparent**: Users can see which transactions were used to generate the answer
- **Efficient**: Semantic search finds relevant information quickly

## 📊 Example Queries

The chatbot can handle various types of questions:

- **Customer History**: "Show me Riya's purchase history"
- **Total Spending**: "What is Amit's total spending?"
- **Date Filtering**: "List all February transactions"
- **Product Queries**: "What did Karan buy?"
- **General Queries**: "Tell me about recent transactions"

## 🛠️ Technical Details

### Core Components

- **`chatbot.py`**: Contains all RAG logic
  - `load_and_prepare()`: Loads and formats transaction data
  - `create_embeddings()`: Generates embeddings using SentenceTransformer
  - `retrieve_transactions()`: Performs cosine similarity search
  - `generate_answer()`: Generates context-aware answers

- **`app.py`**: Streamlit UI
  - Interactive query interface
  - Memory feature (last question)
  - Monthly spending visualization
  - Response display with context

- **`utils.py`**: Helper functions
  - Currency formatting
  - Data filtering utilities
  - Calculation helpers

### Model Information

- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
  - Lightweight and fast
  - Good performance for semantic similarity
  - 384-dimensional embeddings

## 📈 Monthly Spending Chart

The app includes a bar chart showing total spending per month, automatically generated from the transaction data. This provides a quick visual overview of spending patterns.

## 💾 Memory Feature

The chatbot remembers your last question. Click "Use Last Question" to quickly re-ask it, or view it in the info box at the top of the interface.

## 🔧 Customization

### Adding More Transactions

Edit `transactions.json` to add more transaction data. The format is:
```json
{
  "id": 6,
  "customer": "CustomerName",
  "product": "ProductName",
  "amount": 10000,
  "date": "2024-04-15"
}
```

### Adjusting Retrieval

Modify the `top_k` parameter in `retrieve_transactions()` to retrieve more or fewer relevant transactions.

### Changing the Model

Update the `model_name` parameter in `create_embeddings()` to use a different SentenceTransformer model.

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to extend this project with additional features:
- Support for more query types
- Additional visualizations
- Database integration
- User authentication
- Export functionality

## 📧 Support

For questions or issues, please refer to the code comments or create an issue in the repository.

---

**Built with ❤️ using Python, Streamlit, and SentenceTransformers**

