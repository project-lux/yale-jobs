"""Python SDK for Yale HPC jobs - simplified API similar to HuggingFace Jobs."""
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

from yale.cluster import ClusterConnection
from yale.jobs import YaleJob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_job(
    script: str,
    data_source: Optional[Union[str, Any]] = None,
    source_type: str = "auto",
    job_name: str = "yale-job",
    gpus: str = "p100:2",
    partition: str = "gpu",
    cpus_per_task: int = 2,
    time_limit: str = "10:00",
    memory: Optional[str] = None,
    wait: bool = False,
    config_path: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> YaleJob:
    """Run a job on Yale HPC cluster.
    
    Simple API similar to HuggingFace Jobs.
    
    Args:
        script: Python script content to run
        data_source: Data source (path, URL, dataset name, etc.)
        source_type: Type of data source (auto, pdf, iiif, web, directory, hf)
        job_name: Name for the job
        gpus: GPU specification (e.g., "p100:2", "v100:1")
        partition: SLURM partition
        cpus_per_task: Number of CPUs per task
        time_limit: Time limit (HH:MM:SS or HH:MM)
        memory: Memory limit (e.g., "32G")
        wait: Whether to wait for job completion
        config_path: Path to config.yaml file
        username: Username for SSH connection
        password: Password for SSH connection
        
    Returns:
        YaleJob instance for job management
        
    Example:
        >>> from yale import run_job
        >>> 
        >>> script = '''
        ... import pandas as pd
        ... from datasets import load_from_disk
        ... 
        ... dataset = load_from_disk("dataset")
        ... print(f"Loaded {len(dataset)} samples")
        ... '''
        >>> 
        >>> job = run_job(
        ...     script=script,
        ...     data_source="path/to/pdfs",
        ...     job_name="my-ocr-job",
        ...     gpus="v100:2"
        ... )
    """
    # Create cluster connection
    connection = ClusterConnection(config_path)
    connection.connect(username, password)
    
    try:
        # Create job manager
        job = YaleJob(connection, job_name=job_name)
        
        # Prepare data if provided
        dataset_path = None
        if data_source:
            dataset_path = job.prepare_data(data_source, source_type)
        
        # Modify script to use the dataset if provided
        if dataset_path and "load_from_disk" in script:
            # Replace placeholder with actual path
            script = script.replace('load_from_disk("dataset")', f'load_from_disk("{dataset_path}")')
        
        # Create SLURM batch script
        sbatch_content = job.create_sbatch_script(
            script_content=script,
            cpus_per_task=cpus_per_task,
            gpus=gpus,
            partition=partition,
            time_limit=time_limit,
            memory=memory,
        )
        
        # Save batch script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(sbatch_content)
            local_script_path = f.name
        
        # Upload script to cluster
        remote_script_path = f"{job.job_dir}/{job_name}.sh"
        connection.upload_file(local_script_path, remote_script_path)
        
        # Submit job
        job.submit(remote_script_path, wait=wait)
        
        return job
        
    except Exception as e:
        connection.close()
        raise e


def run_ocr_job(
    data_source: Union[str, Any],
    output_dataset: str,
    source_type: str = "auto",
    model: str = "rednote-hilab/dots.ocr",
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    job_name: str = "yale-ocr",
    gpus: str = "p100:2",
    partition: str = "gpu",
    wait: bool = False,
    config_path: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> YaleJob:
    """Run an OCR job on Yale HPC cluster.
    
    Convenience function for running OCR on documents/images.
    
    Args:
        data_source: Data source (PDF, IIIF, directory, etc.)
        output_dataset: Name for output dataset
        source_type: Type of data source (auto, pdf, iiif, web, directory, hf)
        model: OCR model to use
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process
        job_name: Name for the job
        gpus: GPU specification
        partition: SLURM partition (default: gpu)
        wait: Whether to wait for job completion
        config_path: Path to config.yaml file
        username: Username for SSH connection
        password: Password for SSH connection
        
    Returns:
        YaleJob instance for job management
        
    Example:
        >>> from yale import run_ocr_job
        >>> 
        >>> job = run_ocr_job(
        ...     data_source="manuscripts/",
        ...     output_dataset="manuscripts-ocr",
        ...     source_type="pdf",
        ...     gpus="v100:2"
        ... )
        >>> 
        >>> # Check status
        >>> status = job.get_status()
        >>> print(status)
    """
    # Create OCR script
    ocr_script = f"""
import torch
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
dataset = load_from_disk("dataset")
logger.info(f"Loaded {{len(dataset)}} samples")

# Limit samples if requested
{f'dataset = dataset.select(range(min({max_samples}, len(dataset))))' if max_samples else ''}

# Initialize vLLM model
logger.info("Initializing model: {model}")
llm = LLM(
    model="{model}",
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.8,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
)

# Process images
logger.info("Processing images...")
results = []

for i in tqdm(range(0, len(dataset), {batch_size})):
    batch = dataset[i:i+{batch_size}]
    images = batch['image']
    
    # Create messages for batch
    messages = []
    for img in images:
        messages.append([{{
            "role": "user",
            "content": [
                {{"type": "image_url", "image_url": {{"url": img}}}},
                {{"type": "text", "text": "Extract the text content from this image."}}
            ]
        }}])
    
    # Process with vLLM
    outputs = llm.chat(messages, sampling_params)
    
    for output in outputs:
        results.append(output.outputs[0].text.strip())

# Add results to dataset
dataset = dataset.add_column("markdown", results)

# Save results
output_path = "{output_dataset}"
dataset.save_to_disk(output_path)
logger.info(f"Results saved to {{output_path}}")
"""
    
    return run_job(
        script=ocr_script,
        data_source=data_source,
        source_type=source_type,
        job_name=job_name,
        gpus=gpus,
        partition=partition,
        time_limit="02:00:00",
        wait=wait,
        config_path=config_path,
        username=username,
        password=password,
    )


class YaleJobs:
    """Yale Jobs SDK - object-oriented API."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Yale Jobs SDK.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.connection = None
    
    def connect(self, username: Optional[str] = None, password: Optional[str] = None):
        """Connect to the cluster.
        
        Args:
            username: Username for SSH connection
            password: Password for SSH connection
        """
        self.connection = ClusterConnection(self.config_path)
        self.connection.connect(username, password)
        logger.info("Connected to Yale HPC")
    
    def submit_job(
        self,
        script: str,
        data_source: Optional[Union[str, Any]] = None,
        source_type: str = "auto",
        job_name: str = "yale-job",
        **kwargs
    ) -> YaleJob:
        """Submit a job to the cluster.
        
        Args:
            script: Python script content to run
            data_source: Data source (path, URL, dataset name, etc.)
            source_type: Type of data source
            job_name: Name for the job
            **kwargs: Additional job parameters (gpus, partition, etc.)
            
        Returns:
            YaleJob instance
        """
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        
        job = YaleJob(self.connection, job_name=job_name)
        
        # Prepare data if provided
        if data_source:
            dataset_path = job.prepare_data(data_source, source_type)
            if "load_from_disk" in script:
                script = script.replace('load_from_disk("dataset")', f'load_from_disk("{dataset_path}")')
        
        # Create and upload batch script
        sbatch_content = job.create_sbatch_script(script, **kwargs)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(sbatch_content)
            local_script_path = f.name
        
        remote_script_path = f"{job.job_dir}/{job_name}.sh"
        self.connection.upload_file(local_script_path, remote_script_path)
        
        # Submit job
        job.submit(remote_script_path)
        
        return job
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dict
        """
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        
        job = YaleJob(self.connection)
        job.job_id = job_id
        return job.get_status()
    
    def close(self):
        """Close the connection."""
        if self.connection:
            self.connection.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

