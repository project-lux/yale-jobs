"""
IIIF Manifest OCR example using Yale Jobs.

This example shows how to run OCR on a IIIF manifest.
"""
from yale import run_ocr_job

# Run OCR on a IIIF manifest
job = run_ocr_job(
    data_source="https://example.com/iiif/manifest.json",
    output_dataset="iiif-ocr-results",
    source_type="iiif",
    job_name="iiif-ocr",
    gpus="p100:2",
    batch_size=32,
    wait=True  # Wait for completion
)

# Download results
job.download_results("./results")

print("OCR complete! Results downloaded to ./results")

