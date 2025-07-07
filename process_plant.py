import os
import requests
import json
import base64

def update_coda_row(row_id, result_string):
    """Updates a specific row in Coda with the final result."""
    token = os.environ.get('CODA_API_TOKEN')
    doc_id = os.environ.get('CODA_DOC_ID')
    table_id = os.environ.get('CODA_TABLE_ID')
    
    if not all([token, doc_id, table_id, row_id]):
        print("Coda API credentials or Row ID are missing. Cannot update.")
        return

    url = f"https://coda.io/apis/v1/docs/{doc_id}/tables/{table_id}/rows/{row_id}"
    
    headers = {'Authorization': f'Bearer {token}'}
    
    # IMPORTANT: Change "Results" to the exact name of your results column in Coda.
    payload = {
        'row': {
            'cells': [
                {'column': 'Results', 'value': result_string}
            ]
        }
    }
    
    response = requests.put(url, headers=headers, json=payload)
    if response.status_code == 200:
        print(f"Successfully updated Coda row {row_id}.")
    else:
        print(f"Failed to update Coda row. Status: {response.status_code}, Response: {response.text}")


def download_image_from_drive(file_id):
    api_key = os.environ['GOOGLE_DRIVE_API_KEY']
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

def extract_text_from_image(image_data):
    try:
        api_key = os.environ['GOOGLE_VISION_API_KEY']
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {"requests": [{"image": {"content": base64.b64encode(image_data).decode('utf-8')}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        if 'textAnnotations' in result['responses'][0] and result['responses'][0]['textAnnotations']:
            return result['responses'][0]['textAnnotations'][0]['description'].strip()
        else:
            return ""
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def is_valid_plant_name(text):
    return True if text and len(text) < 100 else False

def identify_plant_with_plantnet(image_data):
    api_key = os.environ['PLANTNET_API_KEY']
    url = "https://my-api.plantnet.org/v2/identify/all"
    params = {'include-related-images': 'false', 'nb-results': '1', 'api-key': api_key}
    files = {'images': ('plant_image.jpg', image_data, 'image/jpeg')}
    response = requests.post(url, params=params, files=files, headers={'accept': 'application/json'})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"PlantNet API error: {response.status_code}")

def main():
    image_id = os.environ.get('IMAGE_ID')
    row_id = os.environ.get('ROW_ID')
    final_result_string = ""

    try:
        print(f"Processing Image ID: {image_id} for Coda Row ID: {row_id}")
        image_data = download_image_from_drive(image_id)
        extracted_text = extract_text_from_image(image_data)
        clean_text = " ".join(extracted_text.splitlines())
        
        if is_valid_plant_name(clean_text):
            final_result_string = clean_text
        else:
            plantnet_result = identify_plant_with_plantnet(image_data)
            if plantnet_result.get('results'):
                final_result_string = plantnet_result['results'][0]['species']['scientificNameWithoutAuthor']
            else:
                final_result_string = "Identification failed."
    except Exception as e:
        final_result_string = f"Error: {str(e)}"
    
    print(f"\n--- Final Result: {final_result_string} ---")
    
    # Update the Coda table directly
    update_coda_row(row_id, final_result_string)

if __name__ == "__main__":
    main()
