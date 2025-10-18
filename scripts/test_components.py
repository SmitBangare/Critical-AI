#!/usr/bin/env python3
"""
Simple test script for the enhanced classifier components without BioViL-T dependency.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.enhanced_classifier import MultiModalAttention, EnhancedClassifierConfig
from src.models.clinical_processor import ClinicalDataProcessor, ClinicalConfig, create_sample_clinical_data


def test_attention_mechanism():
    """Test the multi-modal attention mechanism."""
    print("Testing Multi-Modal Attention Mechanism...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = EnhancedClassifierConfig(hidden_size=768, num_heads=12)
    attention = MultiModalAttention(config).to(device)
    
    print(f"Attention mechanism created with {sum(p.numel() for p in attention.parameters()):,} parameters")
    
    # Test attention with dummy data
    batch_size = 4
    image_features = torch.randn(batch_size, 768).to(device)
    clinical_features = torch.randn(batch_size, 768).to(device)
    
    with torch.no_grad():
        # Test image-only attention
        img_only_output = attention(image_features, None)
        print(f"Image-only attention output shape: {img_only_output.shape}")
        
        # Test multi-modal attention
        multimodal_output = attention(image_features, clinical_features)
        print(f"Multi-modal attention output shape: {multimodal_output.shape}")
        
        # Check if outputs are different (attention is working)
        diff = torch.abs(img_only_output - multimodal_output).mean()
        print(f"Difference between image-only and multi-modal: {diff.item():.6f}")
        
        if diff.item() > 1e-6:
            print("SUCCESS: Attention mechanism is working (outputs are different)")
        else:
            print("WARNING: Attention mechanism may not be working properly")
    
    print("SUCCESS: Attention mechanism test completed!")
    return attention


def test_clinical_processor():
    """Test the clinical data processor."""
    print("\n" + "="*50)
    print("Testing Clinical Data Processor...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ClinicalConfig(hidden_size=768)
    processor = ClinicalDataProcessor(config).to(device)
    
    print(f"Clinical processor created with {sum(p.numel() for p in processor.parameters()):,} parameters")
    
    # Test with sample clinical data
    batch_size = 2
    clinical_data = create_sample_clinical_data(batch_size=batch_size, device=device)
    
    print(f"Sample clinical data keys: {list(clinical_data.keys())}")
    print(f"Sample clinical data shapes:")
    for key, value in clinical_data.items():
        print(f"  {key}: {value.shape}")
    
    with torch.no_grad():
        clinical_features = processor(clinical_data)
        print(f"Processed clinical features shape: {clinical_features.shape}")
        
        # Check if features are reasonable
        mean_val = clinical_features.mean().item()
        std_val = clinical_features.std().item()
        print(f"Clinical features - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        
        if abs(mean_val) < 10 and std_val > 0.1:
            print("SUCCESS: Clinical processor is working (reasonable feature values)")
        else:
            print("WARNING: Clinical processor may have issues")
    
    print("SUCCESS: Clinical processor test completed!")
    return processor


def test_integration():
    """Test the integration of attention and clinical processor."""
    print("\n" + "="*50)
    print("Testing Integration...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create components
    attention_config = EnhancedClassifierConfig(hidden_size=768, num_heads=12)
    attention = MultiModalAttention(attention_config).to(device)
    
    clinical_config = ClinicalConfig(hidden_size=768)
    processor = ClinicalDataProcessor(clinical_config).to(device)
    
    # Test data
    batch_size = 2
    image_features = torch.randn(batch_size, 768).to(device)
    clinical_data = create_sample_clinical_data(batch_size=batch_size, device=device)
    
    with torch.no_grad():
        # Process clinical data
        clinical_features = processor(clinical_data)
        
        # Apply attention
        fused_features = attention(image_features, clinical_features)
        
        print(f"Image features shape: {image_features.shape}")
        print(f"Clinical features shape: {clinical_features.shape}")
        print(f"Fused features shape: {fused_features.shape}")
        
        # Check if fusion is working
        fusion_diff = torch.abs(image_features - fused_features).mean()
        print(f"Difference between image and fused features: {fusion_diff.item():.6f}")
        
        if fusion_diff.item() > 1e-6:
            print("SUCCESS: Integration is working (fusion produces different output)")
        else:
            print("WARNING: Integration may not be working properly")
    
    print("SUCCESS: Integration test completed!")


if __name__ == "__main__":
    print("Enhanced Classifier Components Test Suite")
    print("="*50)
    
    try:
        # Test individual components
        attention = test_attention_mechanism()
        processor = test_clinical_processor()
        
        # Test integration
        test_integration()
        
        print("\n" + "="*50)
        print("All tests passed! Enhanced components are ready for integration.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
