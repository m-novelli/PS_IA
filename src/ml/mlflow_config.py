# src/ml/mlflow_config.py
"""
MLflow configuration utilities for cross-platform compatibility.
"""
import os
import mlflow
from pathlib import Path
from typing import Optional


def setup_mlflow_tracking(
    base_dir: Optional[Path] = None,
    experiment_name: str = "default-experiment",
    force_clean: bool = False
) -> str:
    """
    Set up MLflow tracking with cross-platform path handling.
    
    Args:
        base_dir: Base directory for MLflow runs. If None, uses current working directory.
        experiment_name: Name of the MLflow experiment.
        force_clean: If True, clears any existing tracking URI first.
    
    Returns:
        The tracking URI that was set.
    """
    if force_clean:
        # Clear any existing tracking URI to avoid conflicts
        mlflow.set_tracking_uri(None)
    
    # Determine the base directory
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir).resolve()
    
    # Create mlruns directory
    mlruns_dir = base_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tracking URI using absolute path
    tracking_uri = f"file://{mlruns_dir.absolute()}"
    
    # Set the tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI set to: {tracking_uri}")
    print(f"MLflow experiment set to: {experiment_name}")
    
    return tracking_uri


def get_mlflow_tracking_info() -> dict:
    """
    Get current MLflow tracking information.
    
    Returns:
        Dictionary with tracking URI and experiment info.
    """
    try:
        tracking_uri = mlflow.get_tracking_uri()
        experiment = mlflow.get_experiment_by_name(mlflow.active_run().info.experiment_id) if mlflow.active_run() else None
        
        return {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment.name if experiment else None,
            "experiment_id": experiment.experiment_id if experiment else None,
        }
    except Exception as e:
        return {
            "tracking_uri": mlflow.get_tracking_uri(),
            "error": str(e)
        }


def safe_log_dict(data: dict, artifact_path: str) -> bool:
    """
    Safely log a dictionary to MLflow, handling potential path issues.
    
    Args:
        data: Dictionary to log.
        artifact_path: Path for the artifact.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        mlflow.log_dict(data, artifact_path)
        print(f"Successfully logged dictionary to {artifact_path}")
        return True
    except OSError as e:
        if "Read-only file system" in str(e) or "/C:" in str(e):
            print(f"MLflow path issue detected: {e}")
            print("Attempting to reset MLflow tracking URI...")
            
            # Get current experiment name
            try:
                experiment = mlflow.get_experiment_by_name(mlflow.active_run().info.experiment_id)
                experiment_name = experiment.name if experiment else "default-experiment"
            except:
                experiment_name = "default-experiment"
            
            # Reset with clean configuration
            setup_mlflow_tracking(force_clean=True, experiment_name=experiment_name)
            
            # Retry the log operation
            try:
                mlflow.log_dict(data, artifact_path)
                print(f"Successfully logged dictionary to {artifact_path} after reset")
                return True
            except Exception as retry_error:
                print(f"Failed to log dictionary even after reset: {retry_error}")
                return False
        else:
            print(f"Failed to log dictionary: {e}")
            return False
    except Exception as e:
        print(f"Failed to log dictionary: {e}")
        return False
