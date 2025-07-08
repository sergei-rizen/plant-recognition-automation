import os
import requests
import json
import base64

# Import the official Google AI Library
import google.generativeai as genai

def update_coda_row(row_id, result_string):
    """Updates a specific row in Coda with the final recognition result."""
    token = os.environ.get('CODA_API_TOKEN')
    doc_id = os.environ.get('CODA_DOC_ID')
    table_id = os.environ.get('CODA_TABLE_ID')
    
    if not all([token, doc_id, table_id, row_id]):
        print("Coda API credentials or Row ID are missing. Cannot update.")
        return

    url = f"https://coda.io/apis/v1/docs/{doc_id}/tables/{table_id}/rows/{row_id}"
    headers = {'Authorization': f'Bearer {token}'}
    
    # Ensure your results column in Coda is named exactly "Results"
    payload = {
        'row': {
            'cells': [
                {'column': 'Results', 'value': result_string}
            ]
        }
    }
    
    response = requests.put(url, headers=headers, json=payload)
    if 200 <= response.status_code < 300:
        print(f"Successfully sent update to Coda for row {row_id}.")
    else:
        print(f"Failed to update Coda row. Status: {response.status_code}, Response: {response.text}")

def download_image_from_drive(file_id):
    """Downloads an image file from Google Drive using its file ID."""
    api_key = os.environ['GOOGLE_DRIVE_API_KEY']
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content

def extract_text_from_image(image_data):
    """Extracts text from image data using Google Cloud Vision REST API."""
    try:
        api_key = os.environ['GOOGLE_VISION_API_KEY']
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {"requests": [{"image": {"content": base64.b64encode(image_data).decode('utf-8')}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if 'textAnnotations' in result['responses'][0] and result['responses'][0]['textAnnotations']:
            return result['responses'][0]['textAnnotations'][0]['description'].strip()
        return ""
    except Exception as e:
        print(f"Error during Google Vision OCR processing: {e}")
        return ""

def identify_plant_with_plantnet(image_data):
    """Sends image data to the PlantNet API for identification."""
    api_key = os.environ['PLANTNET_API_KEY']
    url = "https://my-api.plantnet.org/v2/identify/all"
    params = {'include-related-images': 'false', 'nb-results': '1', 'api-key': api_key}
    files = {'images': ('plant_image.jpg', image_data, 'image/jpeg')}
    response = requests.post(url, params=params, files=files, headers={'accept': 'application/json'}, timeout=30)
    response.raise_for_status()
    return response.json()

def is_generic_name(file_name):
    """Checks if a file name is generic or contains a potential plant name."""
    if not file_name: return True
    name_part = os.path.splitext(file_name)[0].lower()
    if len(name_part) <= 3 or any(char.isdigit() for char in name_part): return True
    if name_part.startswith(("img_", "dsc_", "image", "screenshot")): return True
    return False

def get_name_from_text_hint(text_hint):
    """Uses Gemini to correct a plant name hint."""
    print(f"Asking Gemini (Text) to correct hint: '{text_hint}'")
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""Analyze the following text, which is supposed to be a plant name: \"{text_hint}\". It might be a common name, a misspelled Latin name, or a partial name. Respond with only the full, corrected scientific name for this plant. If you cannot determine a name with high confidence, respond with the single word: 'Unknown'."""
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        return None if "Unknown" in result else result
    except Exception as e:
        print(f"Gemini (Text) API Error: {e}")
        return None
        
def get_name_from_image(image_data):
    """Uses Gemini Vision as a last resort to identify the plant from the image."""
    print("All other methods failed. Asking Gemini (Vision) to identify plant from image.")
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    # Updated to the new, recommended multimodal model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    image_part = {"mime_type": "image/jpeg", "data": base64.b64encode(image_data)}
    prompt = "Identify the plant in this image. Respond with only its scientific name."
    try:
        response = model.generate_content([prompt, image_part])
        return response.text.strip()
    except Exception as e:
        print(f"Gemini (Vision) API Error: {e}")
        return None

def main():
    """Main function to run the intelligent, tiered image recognition process."""
    image_id = os.environ.get('IMAGE_ID')
    row_id = os.environ.get('ROW_ID')
    image_name = os.environ.get('IMAGE_NAME')
    final_result_string = None
    
    try:
        print(f"Starting process for image '{image_name}' (Row: {row_id})")
        image_data = download_image_from_drive(image_id)
        
        text_source = None
        if image_name and not is_generic_name(image_name):
            text_source = os.path.splitext(image_name)[0]
            print(f"Using filename as hint: '{text_source}'")
        else:
            print("Filename is generic. Trying OCR...")
            ocr_text = extract_text_from_image(image_data)
            if ocr_text:
                text_source = " ".join(ocr_text.splitlines())
                print(f"Using OCR result as hint: '{text_source}'")

        if text_source:
            final_result_string = get_name_from_text_hint(text_source)

        if not final_result_string:
            print("Text-based AI failed or no hint found. Falling back to PlantNet.")
            try:
                plantnet_result = identify_plant_with_plantnet(image_data)
                if plantnet_result.get('results'):
                    final_result_string = plantnet_result['results'][0]['species']['scientificNameWithoutAuthor']
            except Exception as e:
                print(f"PlantNet identification failed: {e}")

        if not final_result_string:
            final_result_string = get_name_from_image(image_data)

    except Exception as e:
        final_result_string = f"Critical Workflow Error: {str(e)}"
    
    if not final_result_string:
        final_result_string = "Complete failure: All identification methods failed."

    print(f"\n--- Final Result: {final_result_string} ---")
    update_coda_row(row_id, final_result_string)

if __name__ == "__main__":
    main()
