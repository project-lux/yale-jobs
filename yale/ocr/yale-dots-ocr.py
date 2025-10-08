"""
Yale-adapted DoTS.ocr script for use with Yale HPC Jobs.
Based on: https://huggingface.co/datasets/uv-scripts/ocr/blob/main/dots-ocr.py

This version is designed to work with Yale's job submission system and
supports all Yale data sources (PDF, IIIF, directories, web URLs, HF datasets).

Usage with Yale Jobs:
    yale jobs ocr path/to/pdfs output-dataset --source-type pdf --gpus v100:2
    yale jobs ocr https://example.com/manifest.json output --source-type iiif
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Union
from datetime import datetime

import torch
from datasets import load_dataset, load_from_disk, Dataset
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# DoTS OCR Prompt Templates (from official dots.ocr repo)
# ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "ocr": "Extract the text content from this image.",

    "layout-all": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.
4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.""",

    "layout-only": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = PROMPT_TEMPLATES["ocr"],
) -> List[Dict]:
    """Create chat message for OCR processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "rednote-hilab/dots.ocr",
    max_model_len: int = 8192,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    split: str = "train",
    max_samples: int = None,
    shuffle: bool = False,
    seed: int = 42,
    prompt_mode: str = "ocr",
    custom_prompt: str = None,
    output_column: str = "markdown",
    load_from_disk_path: bool = False,
):
    """Process images from dataset through DoTS.ocr model.
    
    Args:
        input_dataset: Input dataset ID or path
        output_dataset: Output dataset path
        image_column: Column containing images
        batch_size: Batch size for processing
        model: Model to use
        max_model_len: Maximum model context length
        max_tokens: Maximum tokens to generate
        gpu_memory_utilization: GPU memory utilization
        split: Dataset split
        max_samples: Maximum samples to process
        shuffle: Whether to shuffle dataset
        seed: Random seed
        prompt_mode: Prompt template to use
        custom_prompt: Custom prompt text
        output_column: Output column name
        load_from_disk_path: Whether input is a disk path (for Yale jobs)
    """

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Determine prompt to use
    if custom_prompt:
        prompt = custom_prompt
        logger.info(f"Using custom prompt: {prompt[:50]}...")
    else:
        prompt = PROMPT_TEMPLATES.get(prompt_mode, PROMPT_TEMPLATES["ocr"])
        logger.info(f"Using prompt mode: {prompt_mode}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    
    if load_from_disk_path:
        # Load from disk (for Yale jobs with prepared data)
        dataset = load_from_disk(input_dataset)
    else:
        # Load from HuggingFace Hub
        dataset = load_dataset(input_dataset, split=split)

    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM model
    logger.info(f"Initializing vLLM with model: {model}")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for OCR
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="DoTS.ocr processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img, prompt) for img in batch_images]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_outputs.extend(["[OCR ERROR]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt_mode": prompt_mode if not custom_prompt else "custom",
    }

    if "inference_info" in dataset.column_names:
        # Append to existing inference info
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = json.loads(example["inference_info"]) if example["inference_info"] else []
            except (json.JSONDecodeError, TypeError):
                existing_info = []

            existing_info.append(inference_entry)
            return {"inference_info": json.dumps(existing_info)}

        dataset = dataset.map(update_inference_info)
    else:
        # Create new inference_info column
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    # Save to disk
    logger.info(f"Saving to {output_dataset}")
    dataset.save_to_disk(output_dataset)

    logger.info("✅ DoTS.ocr processing complete!")
    logger.info(f"Dataset saved to: {output_dataset}")
    logger.info(f"Processing time: {processing_time_str}")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("Yale DoTS.ocr Document Processing")
        print("=" * 80)
        print("\nCompact 1.7B multilingual OCR model for Yale HPC")
        print("\nFeatures:")
        print("- 🌍 Multilingual support (100+ languages)")
        print("- ⚡ Fast processing with vLLM")
        print("- 📊 Table extraction and formatting")
        print("- 📐 Formula recognition")
        print("- 📝 Layout-aware text extraction")
        print("\nUsage with Yale Jobs CLI:")
        print("\n1. OCR on PDFs:")
        print("   yale jobs ocr path/to/pdfs output-dataset --source-type pdf")
        print("\n2. OCR on IIIF manifest:")
        print("   yale jobs ocr https://example.com/manifest.json output --source-type iiif")
        print("\n3. OCR on image directory:")
        print("   yale jobs ocr path/to/images output --source-type directory")
        print("\n4. Direct script usage (on cluster):")
        print("   python yale-dots-ocr.py dataset_path output_path --load-from-disk-path")
        print("\n" + "=" * 80)
        print("\nFor full help, run: python yale-dots-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using DoTS.ocr for Yale HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input_dataset", help="Input dataset path or ID")
    parser.add_argument("output_dataset", help="Output dataset path")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--model",
        default="rednote-hilab/dots.ocr",
        help="Model to use (default: rednote-hilab/dots.ocr)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=list(PROMPT_TEMPLATES.keys()),
        default="ocr",
        help=f"Prompt template to use (default: ocr)",
    )
    parser.add_argument(
        "--custom-prompt",
        help="Custom prompt text (overrides --prompt-mode)",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for output text (default: markdown)",
    )
    parser.add_argument(
        "--load-from-disk-path",
        action="store_true",
        help="Input is a disk path (for Yale jobs prepared data)",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        custom_prompt=args.custom_prompt,
        output_column=args.output_column,
        load_from_disk_path=args.load_from_disk_path,
    )

