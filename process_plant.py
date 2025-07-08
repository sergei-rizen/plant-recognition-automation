import os
import requests
import json
import base64
from PIL import Image
from io import BytesIO

# Import the official Google AI Library
import google.generativeai as genai

# --- Helper Functions ---
def update_coda_row(row_id, result_string):
    # This function is unchanged
    token = os.environ.get('CODA_API_TOKEN')
    doc_id = os.environ.get('CODA_DOC_ID')
    table_id = os.environ.get('CODA_TABLE_ID')
    if not all([token, doc_id, table_id, row_id]): return
    url = f"https://coda.io/apis/v1/docs/{doc_id}/tables/{table_id}/rows/{row_id}"
    headers = {'Authorization': f'Bearer {token}'}
    payload = {'row': {'cells': [{'column': 'Results', 'value': result_string}]}}
    response = requests.put(url, headers=headers, json=payload)
    if 200 <= response.status_code < 300: print(f"Successfully updated Coda row {row_id}.")
    else: print(f"Failed to update Coda row. Status: {response.status_code}, Response: {response.text}")

def download_image_from_drive(file_id):
    # This function is unchanged
    api_key = os.environ['GOOGLE_DRIVE_API_KEY']
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content

def extract_text_from_image(image_data):
    # This function is unchanged
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
        print(f"Error during OCR: {e}")
        return ""

def is_generic_name(file_name):
    # This function is unchanged
    if not file_name: return True
    name_part = os.path.splitext(file_name)[0].lower()
    if len(name_part) <= 3 or any(char.isdigit() for char in name_part): return True
    if name_part.startswith(("img_", "dsc_", "image", "screenshot")): return True
    return False

def get_name_from_text_hint(text_hint):
    # This function is unchanged
    print(f"Asking Gemini (Text) to correct hint: '{text_hint}'")
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""Analyze the text: \"{text_hint}\". It is supposed to be a plant name but may be misspelled or a common name. Respond with only the correct scientific name. If you cannot determine a name, respond with 'Unknown'."""
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        return None if "Unknown" in result else result
    except Exception as e:
        print(f"Gemini (Text) API Error: {e}")
        return None
        
def get_name_from_image(image_data):
    # This function is unchanged
    print("All other methods failed. Asking Gemini (Vision) to identify plant from image.")
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        img = Image.open(BytesIO(image_data))
        prompt = "Identify the plant in this image. Respond with only its scientific name."
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"Gemini (Vision) API Error: {e}")
        return None

# --- NEW MAIN LOGIC ---

def main():
    """Main function to run the intelligent, tiered image recognition process."""
    image_id = os.environ.get('IMAGE_ID')
    row_id = os.environ.get('ROW_ID')
    image_name = os.environ.get('IMAGE_NAME')
    plantnet_result = os.environ.get('PLANTNET_RESULT') # Get the new data from Make.com
    final_result_string = None
    
    try:
        # Step 1: Use the filename as a hint if it's not generic
        if image_name and not is_generic_name(image_name):
            hint = os.path.splitext(image_name)[0]
            print(f"Using filename as hint: '{hint}'")
            final_result_string = get_name_from_text_hint(hint)

        # Step 2: If filename fails or is generic, fallback to OCR
        if not final_result_string:
            print("Filename failed or was generic. Trying Google Vision OCR...")
            image_data = download_image_from_drive(image_id) # Download image only if needed
            ocr_text = extract_text_from_image(image_data)
            if ocr_text:
                hint = " ".join(ocr_text.splitlines())
                print(f"Using OCR result as hint: '{hint}'")
                final_result_string = get_name_from_text_hint(hint)

        # Step 3: If both filename and OCR fail, use the PlantNet result from Make.com
        if not final_result_string:
            print("AI text correction failed. Using pre-fetched PlantNet result.")
            if plantnet_result and "error" not in plantnet_result.lower():
                final_result_string = plantnet_result
            else:
                 print("PlantNet result was empty or an error.")

        # Step 4: If all previous methods fail, use Gemini Vision on the image
        if not final_result_string:
            print("All other data sources failed. Using Gemini Vision.")
            if 'image_data' not in locals(): # Download image if we haven't already
                 image_data = download_image_from_drive(image_id)
            final_result_string = get_name_from_image(image_data)

    except Exception as e:
        final_result_string = f"Critical Workflow Error: {str(e)}"
    
    # Final check to ensure we always have a result
    if not final_result_string:
        final_result_string = "Complete failure: All identification methods failed."

    print(f"\n--- Final Result: {final_result_string} ---")
    update_coda_row(row_id, final_result_string)

if __name__ == "__main__":
    main()
