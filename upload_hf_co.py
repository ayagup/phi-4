"""
Upload Phi-4 model to HuggingFace Hub
Handles SSL issues and provides retry logic
"""
import os
import time
from pathlib import Path
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError
import ssl
import urllib3
import warnings
import certifi

# Disable all SSL warnings
urllib3.disable_warnings()
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Completely disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_TOKEN'] = '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'

# Monkey patch SSL
import httpx
import requests

# Disable SSL for requests
try:
    requests.packages.urllib3.disable_warnings()
    from urllib3.util import ssl_
    if hasattr(ssl_, 'DEFAULT_CIPHERS'):
        ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
except:
    pass

# Create custom SSL context
def create_unverified_context(*args, **kwargs):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

ssl._create_default_https_context = create_unverified_context

# Patch httpx to disable SSL
original_httpx_client = httpx.Client

class NoSSLClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)

httpx.Client = NoSSLClient


def upload_with_retry(
    folder_path: str,
    repo_id: str,
    max_retries: int = 3,
    chunk_size: int = 100 * 1024 * 1024  # 100MB chunks
):
    """Upload model with retry logic and chunking"""
    
    print(f"Preparing to upload from: {folder_path}")
    print(f"Target repository: {repo_id}")
    print(f"Max retries: {max_retries}")
    print("=" * 60)
    
    # Login first (you'll need to set HF_TOKEN environment variable)
    token = os.getenv("HF_TOKEN")
    if not token:
        print("\n⚠️  HF_TOKEN not found in environment variables")
        print("Please create a token at: https://huggingface.co/settings/tokens")
        token = input("Enter your HuggingFace token: ").strip()
    
    try:
        login(token=token)
        print("✅ Successfully logged in to HuggingFace")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
    
    # Get list of files to upload
    folder = Path(folder_path)
    files = list(folder.rglob("*"))
    files = [f for f in files if f.is_file()]
    
    print(f"\n📁 Found {len(files)} files to upload")
    total_size = sum(f.stat().st_size for f in files)
    print(f"📊 Total size: {total_size / (1024**3):.2f} GB")
    
    # Upload files one by one with retry logic
    successful = 0
    failed = []
    
    for idx, file_path in enumerate(files, 1):
        relative_path = file_path.relative_to(folder)
        file_size = file_path.stat().st_size
        
        print(f"\n[{idx}/{len(files)}] Uploading: {relative_path}")
        print(f"   Size: {file_size / (1024**2):.2f} MB")
        
        for attempt in range(max_retries):
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=str(relative_path),
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"   ✅ Uploaded successfully")
                successful += 1
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"   ⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed after {max_retries} attempts: {e}")
                    failed.append((relative_path, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Successfully uploaded: {successful}/{len(files)} files")
    
    if failed:
        print(f"❌ Failed uploads: {len(failed)}")
        print("\nFailed files:")
        for file, error in failed:
            print(f"  - {file}: {error}")
    else:
        print("🎉 All files uploaded successfully!")
    
    print(f"\n🔗 Model URL: https://huggingface.co/{repo_id}")


def upload_folder_method(folder_path: str, repo_id: str):
    """Alternative: Upload entire folder at once"""
    
    print("Using folder upload method...")
    print("This may fail with large models or network issues.")
    print("Consider using the file-by-file method instead.\n")
    
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your HuggingFace token: ").strip()
    
    login(token=token)
    
    api = HfApi()
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository {repo_id} created/exists")
        
        print("\nUploading folder...")
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            multi_commits=True,  # Use multiple commits for large uploads
            multi_commits_verbose=True,
        )
        print(f"\n🎉 Upload complete!")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\n💡 Try the file-by-file method instead:")
        print("   Change upload_method='file_by_file' in the script")


if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "C:\\Users\\mayangupta\\Documents\\models\\phi-4"
    REPO_ID = "ayagup/phi-4"
    
    # Choose upload method
    # 'file_by_file' - More reliable, slower, better for large files
    # 'folder' - Faster but may fail with network issues
    UPLOAD_METHOD = 'file_by_file'  # Change to 'folder' for folder method
    
    print("=" * 60)
    print("HuggingFace Model Upload Script")
    print("=" * 60)
    
    if UPLOAD_METHOD == 'file_by_file':
        upload_with_retry(
            folder_path=FOLDER_PATH,
            repo_id=REPO_ID,
            max_retries=3
        )
    else:
        upload_folder_method(
            folder_path=FOLDER_PATH,
            repo_id=REPO_ID
        )