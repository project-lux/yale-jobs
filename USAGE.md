# Yale Jobs Usage Guide

This guide provides detailed examples of using Yale Jobs for various tasks.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [OCR Tasks](#ocr-tasks)
3. [Custom Jobs](#custom-jobs)
4. [Working with Data Sources](#working-with-data-sources)
5. [Job Management](#job-management)
6. [Advanced Topics](#advanced-topics)

## Basic Setup

### Install Yale Jobs

```bash
# Install from source
git clone https://github.com/your-username/yale-jobs.git
cd yale-jobs
pip install -e .
```

### Configure Connection

Create `config.yaml`:

```yaml
alias: grace.hpc.yale.edu  # Your cluster hostname
login: true
env: my-env                # Optional: Conda environment
job_dir: project/jobs      # Job directory
result_dir: project/results
2fa: true
```

### Test Connection

```python
from yale import ClusterConnection

conn = ClusterConnection("config.yaml")
conn.connect()

# Run test command
result = conn.execute_command("echo 'Connected!'")
print(result['stdout'])

conn.close()
```

## OCR Tasks

### OCR on PDFs

#### CLI

```bash
# Basic usage
yale jobs ocr pdfs/ output-dataset --source-type pdf

# With custom settings
yale jobs ocr pdfs/ output \
    --source-type pdf \
    --gpus v100:2 \
    --batch-size 32 \
    --max-samples 100

# Wait for completion
yale jobs ocr pdfs/ output --source-type pdf --wait
```

#### Python

```python
from yale import run_ocr_job

job = run_ocr_job(
    data_source="manuscripts/",
    output_dataset="manuscripts-ocr",
    source_type="pdf",
    batch_size=32,
    gpus="v100:2"
)

print(f"Job ID: {job.job_id}")
```

### OCR on IIIF Manifests

#### CLI

```bash
yale jobs ocr https://example.com/manifest.json output \
    --source-type iiif \
    --gpus p100:2
```

#### Python

```python
from yale import run_ocr_job

job = run_ocr_job(
    data_source="https://digital.library.yale.edu/manifest",
    output_dataset="yale-collection-ocr",
    source_type="iiif",
    gpus="p100:2",
    batch_size=16
)
```

### OCR on Image Directories

```python
from yale import run_ocr_job

# Process all images in a directory
job = run_ocr_job(
    data_source="/path/to/images",
    output_dataset="images-ocr",
    source_type="directory",
    gpus="v100:2"
)
```

### OCR on Web URLs

```python
from yale import run_ocr_job

# Create urls.txt with one URL per line
job = run_ocr_job(
    data_source="urls.txt",
    output_dataset="web-images-ocr",
    source_type="web"
)
```

### OCR on HuggingFace Datasets

```python
from yale import run_ocr_job

job = run_ocr_job(
    data_source="davanstrien/ufo-ColPali",
    output_dataset="ufo-ocr",
    source_type="hf",
    gpus="a100:2",
    batch_size=64
)
```

## Custom Jobs

### Run Custom Python Script

```python
from yale import run_job

# Define your script
script = """
import torch
from datasets import load_from_disk
import pandas as pd

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load prepared dataset
dataset = load_from_disk("dataset")
print(f"Loaded {len(dataset)} samples")

# Your processing
results = []
for item in dataset:
    # Process item
    result = {
        'filename': item['filename'],
        'processed': True
    }
    results.append(result)

# Save results
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("Done!")
"""

# Submit job
job = run_job(
    script=script,
    data_source="path/to/data",
    source_type="directory",
    job_name="custom-processing",
    gpus="v100:2",
    cpus_per_task=4,
    time_limit="02:00:00",
    memory="32G"
)

print(f"Job ID: {job.job_id}")
```

### Run from Script File

```python
from yale import run_job

# Load script from file
with open("my_script.py", "r") as f:
    script = f.read()

job = run_job(
    script=script,
    data_source="data/",
    gpus="v100:2"
)
```

## Working with Data Sources

### PDFs

```python
from yale.data import PDFDataSource

# Single PDF
pdf_ds = PDFDataSource("document.pdf")
images = pdf_ds.to_images(dpi=300)
print(f"Pages: {len(images)}")

# Directory of PDFs
pdf_ds = PDFDataSource("pdfs/")
dataset = pdf_ds.to_dataset()
print(f"Total pages: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Save dataset
dataset.save_to_disk("pdf_dataset")
```

### IIIF Manifests

```python
from yale.data import IIIFDataSource

# Load manifest
iiif_ds = IIIFDataSource("https://example.com/manifest.json")
print(f"IIIF version: {iiif_ds.version}")

# Get image URLs
urls = iiif_ds.get_image_urls()
for url_info in urls[:3]:
    print(f"Canvas {url_info['metadata']['canvas']}: {url_info['url']}")

# Download and create dataset
dataset = iiif_ds.to_dataset(max_size=2000)
print(f"Downloaded {len(dataset)} images")
```

### Directories

```python
from yale.data import DirectoryDataSource

# Recursive search
dir_ds = DirectoryDataSource("images/", recursive=True)
files = dir_ds.get_image_files()
print(f"Found {len(files)} images")

# Create dataset
dataset = dir_ds.to_dataset()
```

### Web URLs

```python
from yale.data import WebDataSource

# Multiple URLs
urls = [
    "https://example.com/img1.jpg",
    "https://example.com/img2.jpg",
]
web_ds = WebDataSource(urls)
dataset = web_ds.to_dataset()

# From file
web_ds = WebDataSource("urls.txt")
dataset = web_ds.to_dataset()
```

## Job Management

### Check Job Status

```python
from yale import YaleJobs

yale = YaleJobs()
yale.connect()

status = yale.get_job_status("12345")
print(f"State: {status['state']}")
print(f"Elapsed: {status['elapsed']}")

yale.close()
```

### Monitor Job Until Complete

```python
job = run_ocr_job(
    data_source="pdfs/",
    output_dataset="output",
    wait=True  # Wait for completion
)

print("Job completed!")
```

### Download Results

```python
from yale import ClusterConnection
from yale.jobs import YaleJob

conn = ClusterConnection()
conn.connect()

job = YaleJob(conn, job_name="my-job")
job.download_results("./results")

conn.close()
```

### View Logs

```bash
# CLI
yale jobs logs --job-name my-job

# Or in Python
job = YaleJob(conn, job_name="my-job")
output = job.get_output()
print(output)
```

### Cancel Job

```bash
# CLI
yale jobs cancel 12345

# Python
job.cancel("12345")
```

## Advanced Topics

### Object-Oriented API

```python
from yale import YaleJobs

# Use context manager
with YaleJobs(config_path="config.yaml") as yale:
    yale.connect()
    
    # Submit multiple jobs
    jobs = []
    for data_dir in ["data1", "data2", "data3"]:
        job = yale.submit_job(
            script=my_script,
            data_source=data_dir,
            job_name=f"job-{data_dir}",
            gpus="v100:1"
        )
        jobs.append(job)
    
    # Monitor all jobs
    for job in jobs:
        status = yale.get_job_status(job.job_id)
        print(f"{job.job_name}: {status['state']}")
```

### Custom SLURM Settings

```python
from yale.cluster import ClusterConnection
from yale.jobs import YaleJob

conn = ClusterConnection()
conn.connect()

job = YaleJob(conn, job_name="custom-job")

# Create custom SLURM script
script = job.create_sbatch_script(
    script_content="python my_script.py",
    cpus_per_task=8,
    gpus="a100:4",
    partition="gpu",
    time_limit="24:00:00",
    memory="128G"
)

print(script)
```

### Direct Cluster Access

```python
from yale.cluster import ClusterConnection

conn = ClusterConnection()
conn.connect()

# Execute commands
result = conn.execute_command("ls -la")
print(result['stdout'])

# Upload file
conn.upload_file("local_file.txt", "/cluster/path/file.txt")

# Download file
conn.download_file("/cluster/results.csv", "local_results.csv")

# Upload directory
conn.upload_directory("local_dir/", "/cluster/remote_dir/")

conn.close()
```

### Batch Processing Multiple Sources

```python
from yale import run_ocr_job

sources = [
    ("pdfs/collection1/", "pdf"),
    ("pdfs/collection2/", "pdf"),
    ("https://example.com/manifest1.json", "iiif"),
    ("https://example.com/manifest2.json", "iiif"),
]

jobs = []
for source, source_type in sources:
    job = run_ocr_job(
        data_source=source,
        output_dataset=f"output-{len(jobs)}",
        source_type=source_type,
        gpus="v100:2"
    )
    jobs.append(job)
    print(f"Submitted job {job.job_id} for {source}")

# Monitor all jobs
for job in jobs:
    status = job.get_status()
    print(f"Job {job.job_id}: {status['state']}")
```

## Tips and Best Practices

### GPU Selection

- **P100** - Good for most OCR tasks
- **V100** - Better performance, use for larger batches
- **A100** - Best performance, use for very large jobs

### Batch Sizes

- Start with 16-32 for most OCR models
- Increase to 64-128 for A100 GPUs
- Decrease if you get OOM errors

### Time Limits

- Add buffer to your estimate
- Check cluster policies for max time
- Break very long jobs into smaller pieces

### Data Preparation

- Test locally with small samples first
- Use `max_samples` for testing
- Compress large datasets before upload

### Debugging

```python
# Test data source locally
from yale.data import PDFDataSource

pdf_ds = PDFDataSource("test.pdf")
dataset = pdf_ds.to_dataset()
print(dataset[0])  # Check first item

# Test script locally (if you have GPU)
# before submitting to cluster
```

## Troubleshooting

### SSH Connection Fails

```bash
# Test SSH manually
ssh netid@cluster-hostname

# Check config
cat config.yaml
```

### Job Fails Immediately

```bash
# Check logs
yale jobs logs --job-name my-job

# Check SLURM output file
ssh netid@cluster
cat job_dir/my-job.out
```

### Out of Memory Errors

- Reduce batch size
- Request more memory with `memory="64G"`
- Use fewer/smaller images

### Slow Processing

- Increase batch size
- Use better GPUs (V100/A100)
- Check GPU utilization on cluster

For more help, see the main [README.md](README.md) or open an issue on GitHub.

