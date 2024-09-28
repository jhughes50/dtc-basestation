from flask import Flask, request, jsonify, render_template, Response
from PIL import Image
import io
import threading
import time
import base64

app = Flask(__name__)

# Shared lists to store streamed text and images
streamed_results = []
result_lock = threading.Lock()


# Endpoint to receive text data
@app.route("/", methods=["POST"])
def receive_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Invalid data"}), 400

    text = data["text"]
    stream_end = data.get("stream_end", False)

    # Add text to the shared list
    with result_lock:
        streamed_results.append({"type": "text", "content": text})
        # if stream_end:
        #     streamed_results.append({"type": "text", "content": "<br>Stream ended.<br>"})

    return jsonify({"message": "Text received successfully"}), 200


# Endpoint to receive image data
@app.route("/image", methods=["POST"])
def receive_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files["image"]

    try:
        # Open the image using PIL
        image = Image.open(image_file.stream)

        # Convert the image to a base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Add the image data to the shared list
        with result_lock:
            streamed_results.append({"type": "image", "content": img_str})

        print(f"Received image with size: {image.size}")
        return jsonify({"message": "Image received successfully"}), 200

    except IOError:
        return jsonify({"error": "Invalid image file"}), 400


# Endpoint to display the webpage
@app.route("/results", methods=["GET"])
def results_page():
    return render_template("results.html")


# Server-Sent Events (SSE) endpoint to stream updates to the webpage
@app.route("/stream")
def stream():
    def generate():
        previous_index = 0
        while True:
            time.sleep(1)  # Check for new data every second
            with result_lock:
                if previous_index < len(streamed_results):
                    # Send the new data
                    for item in streamed_results[previous_index:]:
                        yield f"data: {item['type']}:{item['content']}\n\n"
                    previous_index = len(streamed_results)

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
