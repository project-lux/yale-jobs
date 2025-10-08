"""Yale HPC Jobs - Job management for Yale's HPC cluster."""
from yale.sdk import run_job, run_ocr_job, YaleJobs
from yale.cluster import ClusterConnection
from yale.jobs import YaleJob
from yale.data import (
    PDFDataSource,
    IIIFDataSource,
    WebDataSource,
    DirectoryDataSource,
    HFDataSource,
)

__version__ = "0.0.1"
__all__ = [
    "run_job",
    "run_ocr_job",
    "YaleJobs",
    "ClusterConnection",
    "YaleJob",
    "PDFDataSource",
    "IIIFDataSource",
    "WebDataSource",
    "DirectoryDataSource",
    "HFDataSource",
]

