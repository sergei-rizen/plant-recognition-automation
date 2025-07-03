import os
import requests
import pytesseract
from PIL import Image
from io import BytesIO
import json

def download_image_from_drive(file_id):
    """Download image from Google Drive using file ID"""
    api_key = os.environ['GOOGLE_DRIVE_API_KEY']
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

def extract_text_from_image(image_data):
    """Extract text from image using OCR"""
    image = Image.open(BytesIO(image_data))
    text = pytesseract.image_to_string(image)
    return text.strip()

def is_valid_plant_name(text):
    """Check if extracted text contains valid plant name"""
    # Add your logic here to validate if text contains meaningful plant name
    # This is a simplified example
    if len(text) < 3 or len(text) > 100:
        return False
    
    # Check for common gibberish patterns
    gibberish_patterns = ['screenshot', 'image', 'photo', '###', '???']
    text_lower = text.lower()
    
    for pattern in gibberish_patterns:
        if pattern in text_lower:
            return False
    
    return True

def identify_plant_with_plantnet(image_data):
    """Send image to PlantNet API for identification"""
    api_key = os.environ['PLANTNET_API_KEY']
    
    url = "https://my-api.plantnet.org/v2/identify/all"
    params = {
        'include-related-images': 'false',
        'no-reject': 'false',
        'nb-results': '10',
        'type': 'kt',
        'api-key': api_key
    }
    
    files = {'images': ('plant.jpg', image_data, 'image/jpeg')}
    headers = {'accept': 'application/json'}
    
    response = requests.post(url, params=params, files=files, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"PlantNet API error: {response.status_code}")

def main():
    image_id = os.environ['IMAGE_ID']
    
    try:
        # Step 1: Download image from Google Drive
        print(f"Downloading image with ID: {image_id}")
        image_data = download_image_from_drive(image_id)
        
        # Step 2: Extract text from image
        print("Extracting text from image...")
        extracted_text = extract_text_from_image(image_data)
        print(f"Extracted text: {extracted_text}")
        
        # Step 3: Check if text is valid plant name
        if is_valid_plant_name(extracted_text):
            print("Valid plant name found in image text")
            result = {
                'source': 'ocr',
                'plant_name': extracted_text,
                'confidence': 'medium'
            }
        else:
            print("No valid plant name found, using PlantNet API...")
            # Step 4: Use PlantNet API for identification
            plantnet_result = identify_plant_with_plantnet(image_data)
            
            if plantnet_result.get('results'):
                best_match = plantnet_result['results'][0]
                result = {
                    'source': 'plantnet',
                    'plant_name': best_match['species']['scientificNameWithoutAuthor'],
                    'confidence': best_match['score'],
                    'common_names': best_match['species'].get('commonNames', [])
                }
            else:
                result = {
                    'source': 'none',
                    'plant_name': 'Unknown',
                    'confidence': 0
                }
        
        # Output result
        print("Recognition result:")
        print(json.dumps(result, indent=2))
        
        # You can add code here to update your Coda table or send results back
        
    except Exception as e:
        print(f"Error: {str(e)}")
        result = {
            'source': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    main()
