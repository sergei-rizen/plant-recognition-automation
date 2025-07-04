import os
import requests
import json
from io import BytesIO
from PIL import Image

# Import the OCR libraries
import pytesseract
from PIL import Image, ImageOps

def download_image_from_drive(file_id):
    """Downloads an image file from Google Drive using its file ID."""
    api_key = os.environ['GOOGLE_DRIVE_API_KEY']
    # This URL is for downloading the file content directly
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        # Raise an exception with detailed error info
        error_details = response.text
        raise Exception(f"Failed to download image: {response.status_code} - {error_details}")

def extract_text_from_image(image_data):
    """
    Extracts text from image data using OCR.
    Includes pre-processing steps to improve accuracy.
    """
    try:
        image = Image.open(BytesIO(image_data))
        
        # --- Image Pre-processing for better OCR ---
        # 1. Convert to grayscale
        grayscale_image = image.convert('L')
        
        # 2. Apply thresholding to create a pure black-and-white image.
        # The '180' value is a good starting point but may need tuning.
        threshold_image = grayscale_image.point(lambda x: 0 if x < 180 else 255, '1')
        
        # --- End of Pre-processing ---
        
        # Use a specific page segmentation mode (PSM) for finding a block of text
        # PSM 6: Assume a single uniform block of text.
        config = r'--psm 6'
        text = pytesseract.image_to_string(threshold_image, config=config)
        
        return text.strip()
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""

def is_valid_plant_name(text):
    """
    A simple check to see if the extracted text could be a valid plant name.
    """
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
        raise Exception(f"PlantNet API error: {response.status_code}")

def main():
    """Main function to run the image recognition process."""
    image_id = os.environ.get('IMAGE_ID')
    if not image_id:
        print("Error: IMAGE_ID environment variable not set.")
        return

    final_result = {}

    try:
        # Step 1: Download image from Google Drive
        print(f"Downloading image with ID: {image_id}")
        image_data = download_image_from_drive(image_id)
        
        # Step 2: Extract text from image using OCR
        print("Extracting text from image...")
        extracted_text = extract_text_from_image(image_data)
        print(f"Extracted text: '{extracted_text}'")
        
        # Step 3: Validate extracted text
        if is_valid_plant_name(extracted_text):
            print("Valid plant name found via OCR.")
            final_result = {
                'source': 'ocr',
                'plant_name': extracted_text,
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

    # Print and save the final result as JSON and a log file for the artifact
    print("\n--- Recognition Result ---")
    result_json = json.dumps(final_result, indent=2)
    print(result_json)

    with open("result.json", "w") as f:
        f.write(result_json)
    with open("result.log", "w") as f:
        f.write(result_json)


if __name__ == "__main__":
    main()
