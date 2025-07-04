import os
import requests
import json
import base64
from io import BytesIO

def download_image_from_drive(file_id):
    """Downloads an image file from Google Drive using its file ID."""
    api_key = os.environ['GOOGLE_DRIVE_API_KEY']
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code} - {response.text}")

def extract_text_from_image(image_data):
    """
    Extracts text from image data using Google Cloud Vision REST API with an API Key.
    This version uses the more robust DOCUMENT_TEXT_DETECTION feature.
    """
    try:
        api_key = os.environ['GOOGLE_VISION_API_KEY']
        vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

        # Base64 encode the image data
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Construct the request payload
        payload = {
            "requests": [{
                "image": {"content": encoded_image},
                # Use the more powerful detection feature
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}] 
            }]
        }

        # Make the POST request
        response = requests.post(vision_url, json=payload)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()

        # Check for errors within the Vision API's own response structure
        if 'error' in result['responses'][0]:
            error_details = result['responses'][0]['error']['message']
            raise Exception(f"Google Vision API Error: {error_details}")

        # Extract the full text description if it exists
        if 'textAnnotations' in result['responses'][0] and result['responses'][0]['textAnnotations']:
            return result['responses'][0]['textAnnotations'][0]['description'].strip()
        else:
            return "" # Return empty string if no text is found

    except Exception as e:
        print(f"Error during Google Vision OCR processing: {e}")
        return ""

def is_valid_plant_name(text):
    """A simple check to see if the extracted text could be a valid plant name."""
    if not text or len(text) < 3 or len(text) > 100:
        return False
    
    # Filter out common junk words
    text_lower = text.lower()
    gibberish_patterns = ['screenshot', 'camera', 'photo']
    if any(pattern in text_lower for pattern in gibberish_patterns):
        return False
        
    return True

def identify_plant_with_plantnet(image_data):
    """Sends image data to the PlantNet API for identification."""
    api_key = os.environ['PLANTNET_API_KEY']
    api_url = "https://my-api.plantnet.org/v2/identify/all"
    params = {
        'include-related-images': 'false', 
        'no-reject': 'false', 
        'nb-results': '10', 
        'api-key': api_key
    }
    files = {'images': ('plant_image.jpg', image_data, 'image/jpeg')}
    headers = {'accept': 'application/json'}
    
    response = requests.post(api_url, params=params, files=files, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        # Provide a more specific error message based on the status code
        raise Exception(f"PlantNet API error: {response.status_code}")

def main():
    """Main function to run the image recognition process."""
    image_id = os.environ.get('IMAGE_ID')
    if not image_id:
        print("Error: IMAGE_ID environment variable not set.")
        return

    final_result = {}

    try:
        print(f"Downloading image with ID: {image_id}")
        image_data = download_image_from_drive(image_id)
        
        print("Extracting text from image using Google Vision REST API...")
        extracted_text = extract_text_from_image(image_data)
        # Replace newlines with spaces for cleaner output and validation
        clean_text = " ".join(extracted_text.splitlines())
        print(f"Extracted text: '{clean_text}'")
        
        if is_valid_plant_name(clean_text):
            print("Valid plant name found via Google Vision.")
            final_result = {
                'source': 'ocr_google_vision', 
                'plant_name': clean_text, 
                'image_id': image_id
            }
        else:
            print("No valid plant name found, switching to PlantNet API...")
            plantnet_result = identify_plant_with_plantnet(image_data)
            
            if plantnet_result.get('results'):
                best_match = plantnet_result['results'][0]
                final_result = {
                    'source': 'plantnet',
                    'plant_name': best_match['species']['scientificNameWithoutAuthor'],
                    'confidence': best_match['score'],
                    'common_names': best_match['species'].get('commonNames', []),
                    'image_id': image_id
                }
            else:
                final_result = {'source': 'plantnet', 'error': 'No results found.'}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        final_result = {'source': 'error', 'message': str(e), 'image_id': image_id}

    # Print and save the final result as a JSON file and a log file for the artifact
    print("\n--- Recognition Result ---")
    result_json = json.dumps(final_result, indent=2)
    print(result_json)

    with open("result.json", "w") as f:
        f.write(result_json)
    with open("result.log", "w") as f:
        f.write(result_json)

if __name__ == "__main__":
    main()
