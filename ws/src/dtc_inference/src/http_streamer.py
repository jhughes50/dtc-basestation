import requests
from io import BytesIO
from transformers import TextStreamer
from PIL import Image


class HttpStreamer(TextStreamer):
    def __init__(self, server_url: str, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.server_url = server_url

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Sends the new text to a server via an HTTP POST request."""
        data = {"text": text, "stream_end": stream_end}
        try:
            response = requests.post(self.server_url, json=data)
            response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as e:
            print(f"Error sending HTTP request: {e}")

    def send_image(self, image: Image.Image):
        """Sends an image to the /image endpoint of the server via an HTTP POST request."""
        buffered = BytesIO()
        image_format = "PNG"  # Change to 'JPEG' if needed
        image.save(buffered, format=image_format)
        buffered.seek(0)  # Move to the beginning of the BytesIO buffer

        files = {"image": (f"image.{image_format.lower()}", buffered, f"image/{image_format.lower()}")}

        try:
            # Send the image to the /image endpoint
            response = requests.post(f"{self.server_url}/image", files=files)
            response.raise_for_status()  # Check for HTTP errors
            print("Image sent successfully.")
        except requests.RequestException as e:
            print(f"Error sending image: {e}")
