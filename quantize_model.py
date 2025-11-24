"""
Quantize Phi-4 model and save to disk for Intel GPU acceleration.
This creates optimized INT8 or INT4 models using OpenVINO.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

def quantize_and_save_model(
    model_id: str = "C:\\Users\\mayangupta\\Documents\\models\\phi-4",
    output_dir: str = "C:\\Users\\mayangupta\\Documents\\models\\phi-4-int8-openvino",
    weight_format: str = "int8"  # Options: "int8", "int4", "fp16"
):
    """
    Quantize a model and save it in OpenVINO format.
    
    Args:
        model_id: Source model path or HuggingFace ID
        output_dir: Where to save the quantized model
        weight_format: Quantization format (int8, int4, fp16)
    """
    print(f"Quantizing model: {model_id}")
    print(f"Output directory: {output_dir}")
    print(f"Weight format: {weight_format}")
    print("-" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure quantization based on format
    quantization_config = None
    if weight_format == "int8":
        print("Configuring INT8 weight compression...")
        quantization_config = OVWeightQuantizationConfig(bits=8)
    elif weight_format == "int4":
        print("Configuring INT4 weight compression...")
        quantization_config = OVWeightQuantizationConfig(bits=4, sym=True, group_size=128)
    
    # Load and quantize model
    print(f"Loading and quantizing model with {weight_format.upper()} (this may take several minutes)...")
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,  # Convert to OpenVINO IR format
        trust_remote_code=True,
        quantization_config=quantization_config,  # Apply weight quantization
        compile=False  # Don't compile yet, just export
    )
    
    # Save quantized model
    print(f"Saving quantized model to {output_dir}...")
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("✓ Quantization complete!")
    print(f"✓ Quantized model saved to: {output_dir}")
    print("\nTo use this model:")
    print(f'  model = OVModelForCausalLM.from_pretrained("{output_dir}", device="GPU")')
    print("=" * 60)
    
    # Show size comparison
    original_size = get_dir_size(model_id) if os.path.isdir(model_id) else 0
    quantized_size = get_dir_size(output_dir)
    
    if original_size > 0:
        reduction = ((original_size - quantized_size) / original_size) * 100
        print(f"\nOriginal size: {original_size / (1024**3):.2f} GB")
        print(f"Quantized size: {quantized_size / (1024**3):.2f} GB")
        print(f"Size reduction: {reduction:.1f}%")


def get_dir_size(path: str) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def quantize_both_formats(
    model_id: str = "C:\\Users\\mayangupta\\Documents\\models\\phi-4",
    base_output_dir: str = "C:\\Users\\mayangupta\\Documents\\models"
):
    """
    Quantize model in both INT8 and INT4 formats.
    
    Args:
        model_id: Source model path or HuggingFace ID
        base_output_dir: Base directory for outputs (will create subdirs for each format)
    """
    print("=" * 70)
    print("QUANTIZING MODEL IN BOTH INT8 AND INT4 FORMATS")
    print("=" * 70)
    
    # Quantize INT8
    print("\n" + "=" * 70)
    print("STEP 1/2: Quantizing to INT8...")
    print("=" * 70)
    int8_dir = os.path.join(base_output_dir, "phi-4-int8-openvino")
    quantize_and_save_model(model_id, int8_dir, "int8")
    
    # Quantize INT4
    print("\n" + "=" * 70)
    print("STEP 2/2: Quantizing to INT4...")
    print("=" * 70)
    int4_dir = os.path.join(base_output_dir, "phi-4-int4-openvino")
    quantize_and_save_model(model_id, int4_dir, "int4")
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ ALL QUANTIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nINT8 model: {int8_dir}")
    print(f"INT4 model: {int4_dir}")
    
    int8_size = get_dir_size(int8_dir) / (1024**3)
    int4_size = get_dir_size(int4_dir) / (1024**3)
    
    print(f"\nINT8 size: {int8_size:.2f} GB (~50% smaller)")
    print(f"INT4 size: {int4_size:.2f} GB (~75% smaller)")
    print(f"Savings from INT8 to INT4: {int8_size - int4_size:.2f} GB ({((int8_size - int4_size) / int8_size * 100):.1f}%)")
    print("\nRecommendation: Use INT4 for maximum memory savings, INT8 for best quality")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize HuggingFace model for Intel GPU")
    parser.add_argument(
        "--model_id",
        type=str,
        default="C:\\Users\\mayangupta\\Documents\\models\\phi-4",
        help="Model ID or local path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for quantized model (default: auto-named based on format)"
    )
    parser.add_argument(
        "--weight_format",
        type=str,
        choices=["int8", "int4", "fp16", "both"],
        default="both",
        help="Quantization format (use 'both' to create both int8 and int4 versions)"
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="C:\\Users\\mayangupta\\Documents\\models",
        help="Base output directory when using --weight_format=both"
    )
    
    args = parser.parse_args()
    
    if args.weight_format == "both":
        # Quantize in both formats
        quantize_both_formats(
            model_id=args.model_id,
            base_output_dir=args.base_output_dir
        )
    else:
        # Quantize in single format
        if args.output_dir is None:
            args.output_dir = os.path.join(
                args.base_output_dir,
                f"phi-4-{args.weight_format}-openvino"
            )
        
        quantize_and_save_model(
            model_id=args.model_id,
            output_dir=args.output_dir,
            weight_format=args.weight_format
        )
