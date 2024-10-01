import os
import json
import requests
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()
subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')
endpoint = os.getenv('BING_SEARCH_V7_ENDPOINT')

if not subscription_key or not endpoint:
    raise ValueError("Please make sure your BING_SEARCH_V7_SUBSCRIPTION_KEY and BING_SEARCH_V7_ENDPOINT are set in your environment variables or .env file")

search_url = f"{endpoint}/v7.0/search"
save_dir = "imgs"
os.makedirs(save_dir, exist_ok=True)
query = "Cartier Watch W2PA0008"
mkt = 'en-US'
params = {'q': query, 'mkt': mkt}
headers = {'Ocp-Apim-Subscription-Key': subscription_key}

try:
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()

    print("\nHeaders:\n")
    pprint(response.headers)

    search_results = response.json()
    images_saved = 0
    for i, result in enumerate(search_results['images']['value']):
        image_url = result['contentUrl']
        if image_url:
            try:
              image_data = requests.get(image_url).content
              image_extension = '.jpg'
              image_filename = os.path.join(save_dir, f"{query.replace(' ', '_')}_{i}{image_extension}")
              with open(image_filename, 'wb') as image_file:
                  image_file.write(image_data)
              print(f"Saved image {i+1}: {image_filename}")
              images_saved += 1
            except Exception as e:
                print(f"Failed to download image {i+1}: {e}")

    if images_saved == 0:
        print("No images found in the search results.")
    else:
        print(f"Total images saved: {images_saved}")

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as ex:
    print(f"An error occurred: {ex}")
