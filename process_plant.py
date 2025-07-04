import os
import requests
import json
import base64

def download_image_from_drive(file_id):
    api_key = os.environ['GOOGLE_DRIVE_API_KEY']
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code} - {response.text}")

def extract_text_from_image(image_data):
    try:
        api_key = os.environ['GOOGLE_VISION_API_KEY']
        vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        payload = {"requests": [{"image": {"content": encoded_image}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}
        response = requests.post(vision_url, json=payload)
        response.raise_for_status()
        result = response.json()
        if 'error' in result['responses'][0]:
            raise Exception(f"Google Vision API Error: {result['responses'][0]['error']['message']}")
        if 'textAnnotations' in result['responses'][0] and result['responses'][0]['textAnnotations']:
            return result['responses'][0]['textAnnotations'][0]['description'].strip()
        else:
            return ""
    except Exception as e:
        print(f"Error during Google Vision OCR processing: {e}")
        return ""

def is_valid_plant_name(text):
    """
    A much simpler validation. If the text exists and isn't ridiculously long, it's valid.
    """
    if text and len(text) < 100:
        return True
    return False

def identify_plant_with_plantnet(image_data):
    api_key = os.environ['PLANTNET_API_KEY']
    api_url = "https://my-api.plantnet.org/v2/identify/all"
    params = {'include-related-images': 'false', 'no-reject': 'false', 'nb-results': '1', 'api-key': api_key}
    files = {'images': ('plant_image.jpg', image_data, 'image/jpeg')}
    headers = {'accept': 'application/json'}
    response = requests.post(api_url, params=params, files=files, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"PlantNet API error: {response.status_code}")

def main():
    image_id = os.environ.get('IMAGE_ID')
    final_result_string = ""

    try:
        print(f"Downloading image with ID: {image_id}")
        image_data = download_image_from_drive(image_id)
        
        print("Extracting text from image using Google Vision REST API...")
        extracted_text = extract_text_from_image(image_data)
        clean_text = " ".join(extracted_text.splitlines())
        print(f"Extracted text: '{clean_text}'")
        
        if is_valid_plant_name(clean_text):
            print("Valid text found via Google Vision.")
            final_result_string = clean_text
        else:
            print("No valid text found, switching to PlantNet API...")
            plantnet_result = identify_plant_with_plantnet(image_data)
            
            if plantnet_result.get('results'):
                best_match = plantnet_result['results'][0]
                final_result_string = best_match['species']['scientificNameWithoutAuthor']
            else:
                final_result_string = "PlantNet could not identify."
        
    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        final_result_string = f"Error: {error_message}"
    
    # --- THIS IS THE NEW PART THAT SENDS THE RESULT BACK TO CODA ---
    print(f"\n--- Final Result for Coda: {final_result_string} ---")
    
    # Set the output for the GitHub Action workflow
    output_file = os.environ.get('GITHUB_OUTPUT')
    if output_file:
        with open(output_file, 'a') as f:
            # We output the result as a variable named 'plant_result'
            f.write(f"plant_result={final_result_string}\n")

if __name__ == "__main__":
    main()
