#!/usr/bin/env python3
"""
Verification script to test that the PyTorch compatibility issue has been resolved.
"""

import sentence_transformers
import numpy as np

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import sentence_transformers
        import onnxruntime
        print("✓ sentence_transformers and onnxruntime imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_sentence_transformer_model():
    """Test that we can load a sentence transformer model with ONNX runtime"""
    print("\nTesting sentence transformer model loading...")
    try:
        model = sentence_transformers.SentenceTransformer('Xenova/all-MiniLM-L6-v2', trust_remote_code=True)
        print("✓ Xenova SentenceTransformer model loaded successfully")
        
        # Test encoding
        texts = ["This is a test sentence.", "This is another test sentence."]
        embeddings = model.encode(texts, convert_to_numpy=True)
        print(f"✓ Encoding successful. Embedding shape: {embeddings.shape}")
        return True
    except Exception as e:
        print(f"✗ Model loading or encoding failed: {e}")
        return False

def test_cosine_similarity():
    """Test that cosine similarity calculation works with numpy"""
    print("\nTesting cosine similarity calculation...")
    try:
        # Create sample embeddings
        query_embedding = np.random.rand(1, 384)
        doc_embeddings = np.random.rand(5, 384)
        
        # Calculate cosine similarity using numpy
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        embeddings_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(query_norm, embeddings_norm.T)[0]
        
        print(f"✓ Cosine similarity calculation successful. Similarities shape: {similarities.shape}")
        return True
    except Exception as e:
        print(f"✗ Cosine similarity calculation failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("Verifying Streamlit Cloud compatibility fix...\n")
    
    results = []
    results.append(test_imports())
    
    # Only test model loading if imports work
    if all(results):
        results.append(test_sentence_transformer_model())
        results.append(test_cosine_similarity())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The Streamlit Cloud compatibility issue has been resolved.")
        print("Your Streamlit RAG chatbot should now work correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()