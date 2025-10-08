"""Command-line interface for Yale HPC jobs."""
import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

from yale.sdk import run_job, run_ocr_job, YaleJobs
from yale.cluster import ClusterConnection
from yale.jobs import YaleJob

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def cmd_run(args):
    """Run a custom job."""
    # Load script from file
    with open(args.script, 'r') as f:
        script_content = f.read()
    
    job = run_job(
        script=script_content,
        data_source=args.data_source,
        source_type=args.source_type,
        job_name=args.job_name,
        gpus=args.gpus,
        partition=args.partition,
        cpus_per_task=args.cpus,
        time_limit=args.time,
        memory=args.memory,
        env=args.env,
        wait=args.wait,
        config_path=args.config,
    )
    
    logger.info(f"Job submitted: {job.job_id}")
    logger.info(f"To check status: yale jobs status {job.job_id}")


def cmd_ocr(args):
    """Run an OCR job."""
    job = run_ocr_job(
        data_source=args.data_source,
        output_dataset=args.output,
        source_type=args.source_type,
        model=args.model,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        job_name=args.job_name,
        gpus=args.gpus,
        partition=args.partition,
        time_limit=args.time,
        env=args.env,
        wait=args.wait,
        config_path=args.config,
    )
    
    logger.info(f"OCR job submitted: {job.job_id}")
    logger.info(f"To check status: yale jobs status {job.job_id}")


def cmd_status(args):
    """Check job status."""
    connection = ClusterConnection(args.config)
    connection.connect()
    
    try:
        job = YaleJob(connection)
        job.job_id = args.job_id
        status = job.get_status()
        
        logger.info("\nJob Status:")
        logger.info(f"  Job ID: {status.get('job_id', 'N/A')}")
        logger.info(f"  Name: {status.get('name', 'N/A')}")
        logger.info(f"  State: {status.get('state', 'UNKNOWN')}")
        logger.info(f"  Elapsed: {status.get('elapsed', 'N/A')}")
        logger.info(f"  Nodes: {status.get('nodes', 'N/A')}")
        if 'exit_code' in status:
            logger.info(f"  Exit Code: {status['exit_code']}")
    finally:
        connection.close()


def cmd_cancel(args):
    """Cancel a job."""
    connection = ClusterConnection(args.config)
    connection.connect()
    
    try:
        job = YaleJob(connection)
        job.cancel(args.job_id)
    finally:
        connection.close()


def cmd_download(args):
    """Download job results."""
    connection = ClusterConnection(args.config)
    connection.connect()
    
    try:
        job = YaleJob(connection, job_name=args.job_name)
        job.download_results(args.output_dir, args.pattern)
    finally:
        connection.close()


def cmd_logs(args):
    """View job logs."""
    connection = ClusterConnection(args.config)
    connection.connect()
    
    try:
        job = YaleJob(connection, job_name=args.job_name)
        output = job.get_output()
        print(output)
    finally:
        connection.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='yale',
        description='Yale HPC job management - similar to HuggingFace Jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run OCR on PDFs
  yale jobs ocr path/to/pdfs output-dataset --source-type pdf --gpus v100:2
  
  # Run OCR on IIIF manifest
  yale jobs ocr https://example.com/manifest.json output --source-type iiif
  
  # Run custom script
  yale jobs run my_script.py --data-source my_data/ --gpus p100:2
  
  # Check job status
  yale jobs status 12345
  
  # Download results
  yale jobs download --job-name my-job --output-dir ./results
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config.yaml file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Jobs subcommand
    jobs_parser = subparsers.add_parser('jobs', help='Job management commands')
    jobs_subparsers = jobs_parser.add_subparsers(dest='jobs_command', help='Job commands')
    
    # Run command
    run_parser = jobs_subparsers.add_parser('run', help='Run a custom script')
    run_parser.add_argument('script', help='Path to Python script to run')
    run_parser.add_argument('--data-source', help='Data source (path, URL, etc.)')
    run_parser.add_argument('--source-type', default='auto',
                           choices=['auto', 'pdf', 'iiif', 'web', 'directory', 'hf'],
                           help='Type of data source')
    run_parser.add_argument('--job-name', default='yale-job', help='Job name')
    run_parser.add_argument('--gpus', default='p100:2', help='GPU specification')
    run_parser.add_argument('--partition', default='gpu', help='SLURM partition')
    run_parser.add_argument('--cpus', type=int, default=2, help='CPUs per task')
    run_parser.add_argument('--time', default='10:00', help='Time limit')
    run_parser.add_argument('--memory', help='Memory limit (e.g., 32G)')
    run_parser.add_argument('--env', help='Conda environment name (overrides config.yaml)')
    run_parser.add_argument('--wait', action='store_true', help='Wait for completion')
    run_parser.set_defaults(func=cmd_run)
    
    # OCR command
    ocr_parser = jobs_subparsers.add_parser('ocr', help='Run OCR job')
    ocr_parser.add_argument('data_source', help='Data source (PDF, IIIF, directory, etc.)')
    ocr_parser.add_argument('output', help='Output dataset name')
    ocr_parser.add_argument('--source-type', default='auto',
                           choices=['auto', 'pdf', 'iiif', 'web', 'directory', 'hf'],
                           help='Type of data source')
    ocr_parser.add_argument('--model', default='rednote-hilab/dots.ocr',
                           help='OCR model to use')
    ocr_parser.add_argument('--batch-size', type=int, default=16,
                           help='Batch size for processing')
    ocr_parser.add_argument('--max-samples', type=int,
                           help='Maximum number of samples to process')
    ocr_parser.add_argument('--job-name', default='yale-ocr', help='Job name')
    ocr_parser.add_argument('--gpus', default='p100:2', help='GPU specification')
    ocr_parser.add_argument('--partition', default='gpu', help='SLURM partition (default: gpu)')
    ocr_parser.add_argument('--time', default='02:00:00', help='Time limit (HH:MM:SS, default: 02:00:00)')
    ocr_parser.add_argument('--env', help='Conda environment name (overrides config.yaml)')
    ocr_parser.add_argument('--wait', action='store_true', help='Wait for completion')
    ocr_parser.set_defaults(func=cmd_ocr)
    
    # Status command
    status_parser = jobs_subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('job_id', help='Job ID to check')
    status_parser.set_defaults(func=cmd_status)
    
    # Cancel command
    cancel_parser = jobs_subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('job_id', help='Job ID to cancel')
    cancel_parser.set_defaults(func=cmd_cancel)
    
    # Download command
    download_parser = jobs_subparsers.add_parser('download', help='Download job results')
    download_parser.add_argument('--job-name', required=True, help='Job name')
    download_parser.add_argument('--output-dir', default='./results',
                                help='Local directory for results')
    download_parser.add_argument('--pattern', default='*',
                                help='File pattern to download')
    download_parser.set_defaults(func=cmd_download)
    
    # Logs command
    logs_parser = jobs_subparsers.add_parser('logs', help='View job logs')
    logs_parser.add_argument('--job-name', required=True, help='Job name')
    logs_parser.set_defaults(func=cmd_logs)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command given
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'jobs' and not args.jobs_command:
        jobs_parser.print_help()
        sys.exit(0)
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

