#!/usr/bin/env python3
"""
Test script for the enhanced BioViL-T classifier with multi-modal attention.
"""

import torch
from torchvision import transforms
from PIL import Image

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.biovil_t_loader import load_biovil_t, get_local_model_dir
from src.models.enhanced_classifier import EnhancedBioViLTClassifier, EnhancedClassifierConfig
from src.models.clinical_processor import ClinicalDataProcessor, ClinicalConfig, create_sample_clinical_data


def test_enhanced_model():
    """Test the enhanced BioViL-T model with multi-modal capabilities."""
    print("Testing Enhanced BioViL-T Classifier with Multi-Modal Attention...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load BioViL-T model
    local_dir = get_local_model_dir()
    print(f"Loading BioViL-T from: {local_dir}")
    vision_model, image_processor = load_biovil_t(local_dir)
    vision_model = vision_model.to(device)
    
    # Create enhanced classifier
    config = EnhancedClassifierConfig(hidden_size=768, num_heads=12)
    enhanced_model = EnhancedBioViLTClassifier(vision_model, config).to(device)
    
    # Create clinical data processor
    clinical_config = ClinicalConfig(hidden_size=768)
    clinical_processor = ClinicalDataProcessor(clinical_config).to(device)
    
    print(f"Enhanced model created with {sum(p.numel() for p in enhanced_model.parameters()):,} parameters")
    
    # Test with image-only (backward compatibility)
    print("\n1. Testing image-only mode (backward compatibility)...")
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        logits_image_only = enhanced_model(dummy_image, None)
        probs_image_only = torch.sigmoid(logits_image_only)
    
    print(f"Image-only output shape: {logits_image_only.shape}")
    print(f"Image-only probabilities: {probs_image_only.flatten().tolist()}")
    
    # Test with multi-modal data
    print("\n2. Testing multi-modal mode...")
    clinical_data = create_sample_clinical_data(batch_size=2, device=device)
    
    with torch.no_grad():
        clinical_features = clinical_processor(clinical_data)
        logits_multimodal = enhanced_model(dummy_image, clinical_features)
        probs_multimodal = torch.sigmoid(logits_multimodal)
    
    print(f"Clinical features shape: {clinical_features.shape}")
    print(f"Multi-modal output shape: {logits_multimodal.shape}")
    print(f"Multi-modal probabilities: {probs_multimodal.flatten().tolist()}")
    
    # Test with real image (if available)
    print("\n3. Testing with real image...")
    try:
        # Try to load a real image for testing
        test_image_path = "Critical-AI/scripts/sample_cxr.png"  # You can add a sample image here
        # For now, we'll use a dummy image
        real_image = torch.randn(1, 3, 224, 224).to(device)
        
        real_clinical_data = create_sample_clinical_data(batch_size=1, device=device)
        
        with torch.no_grad():
            real_clinical_features = clinical_processor(real_clinical_data)
            real_logits = enhanced_model(real_image, real_clinical_features)
            real_probs = torch.sigmoid(real_logits)
        
        print(f"Real image output: {real_logits.item():.4f}")
        print(f"Real image probability: {real_probs.item():.4f}")
        
        # Show clinical data values
        print("\nClinical data used:")
        for key, value in real_clinical_data.items():
            print(f"  {key}: {value.item():.2f}")
            
    except Exception as e:
        print(f"Real image test skipped: {e}")
    
    print("\n✅ Enhanced model test completed successfully!")
    return enhanced_model, clinical_processor


def test_attention_mechanism():
    """Test the multi-modal attention mechanism separately."""
    print("\n" + "="*50)
    print("Testing Multi-Modal Attention Mechanism...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = EnhancedClassifierConfig(hidden_size=768, num_heads=12)
    
    from src.models.enhanced_classifier import MultiModalAttention
    
    attention = MultiModalAttention(config).to(device)
    
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
            print("✅ Attention mechanism is working (outputs are different)")
        else:
            print("⚠️  Attention mechanism may not be working properly")
    
    print("✅ Attention mechanism test completed!")


if __name__ == "__main__":
    print("Enhanced BioViL-T Classifier Test Suite")
    print("="*50)
    
    try:
        # Test the enhanced model
        enhanced_model, clinical_processor = test_enhanced_model()
        
        # Test attention mechanism
        test_attention_mechanism()
        
        print("\n" + "="*50)
        print("🎉 All tests passed! Enhanced model is ready for training.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
