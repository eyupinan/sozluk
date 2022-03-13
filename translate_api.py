from google.cloud import translate
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="steel-minutia-328909-be0ddc217908.json"

def translate_text(text=[], project_id="steel-minutia-328909"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": text,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": "tr",
        }
    )

    # Display the translation for each input text provided
    upshot=[]
    for translation in response.translations:
        upshot.append(translation.translated_text)
    return upshot
if __name__=="__main__":
    print(translate_text(["spy"]))
