import os
import requests
import shutil
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Function to download file content
def download_file(file_url, local_path, github_token=None):
    headers = {}
    if github_token:
        headers['Authorization'] = f"token {github_token}"
        
    response = requests.get(file_url, headers=headers)
    if response.status_code == 200:
        with open(local_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {local_path}")
    else:
        print(f"Failed to download {file_url} - HTTP {response.status_code}: {response.text}")

# Recursive function to fetch files and folders
def fetch_contents(url, folder_path, github_token=None):
    headers = {}
    if github_token:
        headers['Authorization'] = f"token {github_token}"
        
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        items = response.json()
        for item in items:
            if item["type"] == "file":
                # Check for coding file extensions
                if item["name"].endswith((
                    ".h5", ".pkl",  # Model and serialized objects
                    ".rst",  # Documentation and text files
                    ".pdf", ".docx", ".odt",  # Document files
                    ".jpg", ".jpeg", ".png", ".gif",  # Image files
                    ".mp4", ".avi", ".mov",  # Video files
                    ".mp3", ".wav",  # Audio files
                    ".zip", ".tar", ".tar.gz", ".tar.bz2", ".7z",  # Archive files
                    ".bak", ".log", ".tmp",  # Backup, log, and temporary files
                    ".gitattributes", ".gitignore", ".LICENSE",  # Git and license files
                    ".exe", ".bat", ".sh", ".jar",  # Executable and script files
                    ".psd", ".ai", ".svg",  # Design and vector files
                )):
                    pass
                else:
                    local_path = os.path.join(folder_path, item["name"])
                    os.makedirs(folder_path, exist_ok=True)
                    download_file(item["download_url"], local_path, github_token)
            elif item["type"] == "dir":
                # Recursively process directories
                new_folder_path = os.path.join(folder_path, item["name"])
                fetch_contents(item["url"], new_folder_path, github_token)
    else:
        print(f"Error accessing {url} - HTTP {response.status_code}: {response.text}")

# Main function to scrape GitHub repository
def scrape_github_repo(repo_url, download_folder, github_token=None):
    """
    Scrape all coding files from a GitHub repository using the GitHub API.
    
    Args:
        repo_url (str): URL of the GitHub repository.
        download_folder (str): Local folder to save the downloaded files.
        github_token (str): Personal access token for the GitHub API.

    Returns:
        str: Success message or error.
    """
    if os.path.exists(download_folder):
        shutil.rmtree(download_folder)  # Clear old files
    os.makedirs(download_folder, exist_ok=True)
    github = os.getenv('GITHUB')
    # Extract repo name from URL
    repo_name = repo_url.split("github.com/")[-1].strip("/")
    repo_api_url = f"https://api.github.com/repos/{repo_name}/contents"

    print(f"Starting to scrape repository: {repo_url}")
    fetch_contents(repo_api_url, download_folder, github)
    return f"Files from {repo_url} downloaded successfully to {download_folder}!"
