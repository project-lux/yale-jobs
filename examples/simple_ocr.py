"""
Simple OCR example using Yale Jobs.

This example shows how to run OCR on a directory of PDFs.
"""
from yale import run_ocr_job

# Run OCR on a directory of PDFs
job = run_ocr_job(
    data_source="path/to/pdfs",
    output_dataset="my-ocr-results",
    source_type="pdf",
    job_name="pdf-ocr",
    gpus="v100:2",
    wait=False  # Don't wait for completion
)

print(f"Job submitted: {job.job_id}")
print(f"Check status with: yale jobs status {job.job_id}")

