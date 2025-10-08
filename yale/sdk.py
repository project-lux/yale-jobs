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
    env: Optional[str] = None,
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
        env: Conda environment name (overrides config.yaml)
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
    import uuid
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"[RUN_JOB CALLED] run_id={run_id}, job_name={job_name}")
    
    # Create cluster connection
    connection = ClusterConnection(config_path)
    
    # Override env in config if provided
    if env:
        connection.config['env'] = env
    
    connection.connect(username, password)
    
    try:
        # Create job manager
        job = YaleJob(connection, job_name=job_name)
        logger.info(f"[RUN_JOB {run_id}] Created YaleJob instance")
        
        # Prepare data if provided
        dataset_path = None
        if data_source:
            dataset_path = job.prepare_data(data_source, source_type)
        
        # Modify script to use the dataset if provided
        if dataset_path and "load_from_disk" in script:
            # Replace placeholder with actual path
            script = script.replace('load_from_disk("dataset")', f'load_from_disk("{dataset_path}")')
        
        # Save Python script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            local_python_script = f.name
        
        # Upload Python script to cluster
        remote_python_script = f"{job.job_dir}/{job_name}.py"
        connection.upload_file(local_python_script, remote_python_script)
        
        # Create SLURM batch script that calls the Python script
        bash_command = f"python {remote_python_script}"
        sbatch_content = job.create_sbatch_script(
            script_content=bash_command,
            cpus_per_task=cpus_per_task,
            gpus=gpus,
            partition=partition,
            time_limit=time_limit,
            memory=memory,
        )
        
        # Save batch script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(sbatch_content)
            local_sbatch_script = f.name
        
        # Upload batch script to cluster
        remote_sbatch_script = f"{job.job_dir}/{job_name}.sh"
        connection.upload_file(local_sbatch_script, remote_sbatch_script)
        
        # Submit job
        logger.info(f"[RUN_JOB {run_id}] About to call job.submit()")
        job.submit(remote_sbatch_script, wait=wait)
        logger.info(f"[RUN_JOB {run_id}] job.submit() completed")
        
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
    time_limit: str = "02:00:00",
    env: Optional[str] = None,
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
        time_limit: Time limit in HH:MM:SS format (default: 02:00:00)
        env: Conda environment name (overrides config.yaml)
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
    logger.info(f"[RUN_OCR_JOB CALLED] job_name={job_name}, source={data_source}")
    
    # Create OCR script using file:// URLs (more efficient than base64)
    ocr_script = f"""
import os
import tempfile
import torch
from datasets import load_from_disk
from PIL import Image
from vllm import LLM, SamplingParams
from tqdm import tqdm
from toolz import partition_all
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_image_to_temp(image, temp_dir):
    \"\"\"Save PIL Image to temporary file and return file:// URL.\"\"\"
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        import io
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {{type(image)}}")
    
    # Convert to RGB
    pil_img = pil_img.convert("RGB")
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir)
    pil_img.save(temp_file.name, format="PNG")
    temp_file.close()
    
    # Return file:// URL
    return f"file://{{temp_file.name}}", temp_file.name


def make_ocr_message(image_url, prompt="Extract the text content from this image."):
    \"\"\"Create vLLM chat message with file:// URL.\"\"\"
    return [
        {{
            "role": "user",
            "content": [
                {{"type": "text", "text": prompt}},
                {{"type": "image_url", "image_url": {{"url": image_url}}}},
            ],
        }}
    ]


# Load dataset
dataset = load_from_disk("dataset")
logger.info(f"Loaded {{len(dataset)}} samples")

# Limit samples if requested
{f'dataset = dataset.select(range(min({max_samples}, len(dataset))))' if max_samples else ''}

# Create temporary directory for image files
temp_dir = tempfile.mkdtemp(prefix="yale_ocr_")
logger.info(f"Created temp directory: {{temp_dir}}")

# Get the parent directory of the dataset for allowed_local_media_path
dataset_parent = os.path.dirname(os.path.abspath("dataset"))

# Initialize vLLM model with allowed_local_media_path
logger.info("Initializing model: {model}")
llm = LLM(
    model="{model}",
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.8,
    allowed_local_media_path=dataset_parent,  # Allow loading from dataset directory
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
)

# Process images
logger.info(f"Processing {{len(dataset)}} images in batches of {batch_size}")
results = []
temp_files = []

try:
    for batch_indices in tqdm(
        partition_all({batch_size}, range(len(dataset))),
        total=(len(dataset) + {batch_size} - 1) // {batch_size},
        desc="DoTS.ocr processing"
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i]['image'] for i in batch_indices]
        
        try:
            # Save images to temp files and get file:// URLs
            batch_urls = []
            for img in batch_images:
                file_url, temp_path = save_image_to_temp(img, temp_dir)
                batch_urls.append(file_url)
                temp_files.append(temp_path)
            
            # Create messages for batch with file:// URLs
            batch_messages = [make_ocr_message(url) for url in batch_urls]
            
            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)
            
            # Extract outputs
            for output in outputs:
                results.append(output.outputs[0].text.strip())
        
        except Exception as e:
            logger.error(f"Error processing batch: {{e}}")
            # Add error placeholders for failed batch
            results.extend(["[OCR ERROR]"] * len(batch_images))

finally:
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    try:
        os.rmdir(temp_dir)
    except:
        pass

# Add results to dataset
logger.info("Adding markdown column to dataset")
dataset = dataset.add_column("markdown", results)

# Save results
output_path = "{output_dataset}"
logger.info(f"Saving to {{output_path}}")
dataset.save_to_disk(output_path)
logger.info("✅ OCR processing complete!")
"""
    
    return run_job(
        script=ocr_script,
        data_source=data_source,
        source_type=source_type,
        job_name=job_name,
        gpus=gpus,
        partition=partition,
        time_limit=time_limit,
        env=env,
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
        
        # Save Python script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            local_python_script = f.name
        
        remote_python_script = f"{job.job_dir}/{job_name}.py"
        self.connection.upload_file(local_python_script, remote_python_script)
        
        # Create and upload batch script that calls Python script
        bash_command = f"python {remote_python_script}"
        sbatch_content = job.create_sbatch_script(bash_command, **kwargs)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(sbatch_content)
            local_sbatch_script = f.name
        
        remote_sbatch_script = f"{job.job_dir}/{job_name}.sh"
        self.connection.upload_file(local_sbatch_script, remote_sbatch_script)
        
        # Submit job
        job.submit(remote_sbatch_script)
        
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

