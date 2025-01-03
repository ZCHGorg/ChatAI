import os
import threading
import requests

# Define the base URL for your service
SERVICE_BASE_URL = "http://localhost:8080"

def upload_training_data(file_name, directory):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "rb") as file:
        data = file.read()

    upload_url = f"{SERVICE_BASE_URL}/upload/{directory}"
    response = requests.post(upload_url, data=data)

    return response.status_code == 200

def download_training_data(file_name, directory):
    download_url = f"{SERVICE_BASE_URL}/download/{directory}/{file_name}"
    response = requests.get(download_url)

    if response.status_code == 200:
        file_path = os.path.join(directory, file_name)
        with open(file_path, "wb") as file:
            file.write(response.content)
        return True
    else:
        return False

def scrape_web_page(url):
    with threading.RLock() as scrape_web_page_lock:
        response = requests.get(url)

        if response.status_code == 200:
            return response.text
        else:
            return None

if __name__ == "__main__":
    # Example usage
    file_to_upload = "data.txt"
    upload_directory = "/uploads"
    if upload_training_data(file_to_upload, upload_directory):
        print("File uploaded successfully")
    else:
        print("File upload failed")

    file_to_download = "downloaded_data.txt"
    download_directory = "/downloads"
    if download_training_data(file_to_download, download_directory):
        print("File downloaded successfully")
    else:
        print("File download failed")

    url_to_scrape = "https://www.example.com"
    scraped_content = scrape_web_page(url_to_scrape)
    if scraped_content:
        print("Web page content scraped successfully")
    else:
        print("Web page scraping failed")
