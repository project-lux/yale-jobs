"""
Examples of using different data sources with Yale Jobs.

This shows how to work with PDFs, IIIF, web URLs, directories, and HF datasets.
"""
from yale.data import (
    PDFDataSource,
    IIIFDataSource,
    WebDataSource,
    DirectoryDataSource,
    HFDataSource,
)

# ═══════════════════════════════════════════════════════════════
# 1. PDF Data Source
# ═══════════════════════════════════════════════════════════════

# Single PDF file
pdf_ds = PDFDataSource("document.pdf")
images = pdf_ds.to_images(dpi=300)
print(f"Extracted {len(images)} pages from PDF")

# Directory of PDFs
pdf_dir_ds = PDFDataSource("path/to/pdfs/")
dataset = pdf_dir_ds.to_dataset()
print(f"Created dataset with {len(dataset)} pages")

# ═══════════════════════════════════════════════════════════════
# 2. IIIF Manifest Data Source
# ═══════════════════════════════════════════════════════════════

# Load from IIIF manifest URL
iiif_ds = IIIFDataSource("https://example.com/iiif/manifest.json")
print(f"IIIF version: {iiif_ds.version}")

# Get image URLs
image_urls = iiif_ds.get_image_urls()
print(f"Found {len(image_urls)} images in manifest")

# Download images and create dataset
dataset = iiif_ds.to_dataset(max_size=2000)
print(f"Downloaded {len(dataset)} images")

# ═══════════════════════════════════════════════════════════════
# 3. Web URLs Data Source
# ═══════════════════════════════════════════════════════════════

# Single URL
web_ds = WebDataSource("https://example.com/image.jpg")
images = web_ds.to_images()

# Multiple URLs
urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    "https://example.com/image3.jpg",
]
web_ds = WebDataSource(urls)
dataset = web_ds.to_dataset()

# URLs from file (one per line)
web_ds = WebDataSource("urls.txt")
dataset = web_ds.to_dataset()

# ═══════════════════════════════════════════════════════════════
# 4. Directory Data Source
# ═══════════════════════════════════════════════════════════════

# Load images from directory (recursive)
dir_ds = DirectoryDataSource("path/to/images/", recursive=True)
print(f"Found {len(dir_ds.get_image_files())} image files")

# Create dataset
dataset = dir_ds.to_dataset()
print(f"Dataset columns: {dataset.column_names}")

# ═══════════════════════════════════════════════════════════════
# 5. HuggingFace Dataset Data Source
# ═══════════════════════════════════════════════════════════════

# Load from HuggingFace Hub
hf_ds = HFDataSource(
    dataset_name="davanstrien/ufo-ColPali",
    split="train",
    image_column="image"
)
dataset = hf_ds.to_dataset()
print(f"Loaded {len(dataset)} samples from HuggingFace")

