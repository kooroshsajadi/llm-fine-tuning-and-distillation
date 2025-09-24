import logging
from pathlib import Path
import hashlib

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def sha256sum(filename: Path) -> str:
    """
    Compute SHA-256 checksum of a file.

    Args:
        filename (Path): Path to the file.

    Returns:
        str: SHA-256 checksum as a hexadecimal string.
    """
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def check_integrity(directory: str) -> bool:
    """
    Check integrity of model or adapter files by comparing computed SHA-256 checksums
    with stored .sha256 files.

    Args:
        directory (str): Path to the directory containing model/adapter files and their .sha256 checksums.

    Returns:
        bool: True if all files pass integrity check, False otherwise.

    Raises:
        FileNotFoundError: If directory or expected .sha256 files are missing.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.error(f"Directory does not exist: {directory}")
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # Supported file extensions for models and adapters
    valid_extensions = {'.json', '.safetensors', '.bin'}
    files_to_check = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix in valid_extensions
    ]

    if not files_to_check:
        logger.error(f"No valid files (.json, .safetensors, .bin) found in {directory}")
        raise FileNotFoundError(f"No valid files found in {directory}")

    all_passed = True
    for file_path in files_to_check:
        checksum_file = file_path.with_suffix(file_path.suffix + '.sha256')
        if not checksum_file.exists():
            logger.error(f"Checksum file missing for {file_path.name}")
            all_passed = False
            continue

        # Compute checksum
        computed_checksum = sha256sum(file_path)
        # Read stored checksum
        with open(checksum_file, 'r') as f:
            stored_checksum = f.read().strip()

        if computed_checksum == stored_checksum:
            logger.info(f"Integrity check passed for {file_path.name}")
        else:
            logger.error(f"Integrity check failed for {file_path.name}: "
                        f"Computed={computed_checksum}, Stored={stored_checksum}")
            all_passed = False

    return all_passed

def main():
    """
    Main function to check integrity of saved model or adapter files.

    Example:
        python check_integrity.py
    """
    # Specify the path to the model/adapter directory
    directory = "adapters/gpt2_small_kd"  # Change this to your target directory

    try:
        result = check_integrity(directory)
        if result:
            logger.info(f"All files in {directory} passed integrity check")
        else:
            logger.error(f"Some files in {directory} failed integrity check")
            exit(1)
    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during integrity check: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()