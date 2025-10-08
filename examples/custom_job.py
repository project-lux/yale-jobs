"""
Custom job example using Yale Jobs.

This example shows how to run a custom Python script on the cluster.
"""
from yale import YaleJobs

# Create SDK instance
yale = YaleJobs(config_path="config.yaml")

# Connect to cluster
yale.connect()

# Define custom script
custom_script = """
import torch
from datasets import load_from_disk
import pandas as pd

# Load prepared dataset
dataset = load_from_disk("dataset")

print(f"Processing {len(dataset)} samples...")
print(f"CUDA available: {torch.cuda.is_available()}")

# Your custom processing here
results = []
for item in dataset:
    # Process each item
    result = process_item(item)  # Your function
    results.append(result)

# Save results
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("Processing complete!")
"""

# Submit job
job = yale.submit_job(
    script=custom_script,
    data_source="path/to/images",
    source_type="directory",
    job_name="custom-processing",
    gpus="v100:2",
    cpus_per_task=4,
    time_limit="02:00:00",
    memory="32G"
)

print(f"Job submitted: {job.job_id}")

# Close connection
yale.close()

