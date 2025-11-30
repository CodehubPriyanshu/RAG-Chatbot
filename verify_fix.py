#!/usr/bin/env python3
"""
Verification script to test that the PyTorch compatibility issue has been resolved.
"""

import torch
import sentence_transformers
import transformers

def test_torch_compiler():
    """Test that torch.compiler is available"""
    print("Testing torch.compiler availability...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Has torch.compiler: {hasattr(torch, 'compiler')}")
    
    # This should not raise an AttributeError
    if hasattr(torch, 'compiler'):
        print("✓ torch.compiler is available")
        return True
    else:
        print("✗ torch.compiler is not available")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("\nTesting imports...")
    try:
        import sentence_transformers
        import transformers
        print("✓ sentence_transformers and transformers imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_sentence_transformer_model():
    """Test that we can load a sentence transformer model"""
    print("\nTesting sentence transformer model loading...")
    try:
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ SentenceTransformer model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("Verifying PyTorch compatibility fix...\n")
    
    results = []
    results.append(test_torch_compiler())
    results.append(test_imports())
    
    # Only test model loading if imports work
    if all(results):
        results.append(test_sentence_transformer_model())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The PyTorch compatibility issue has been resolved.")
        print("Your Streamlit RAG chatbot should now work correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()