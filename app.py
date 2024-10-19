from flask import Flask, request, send_file
from rembg import new_session, remove
from io import BytesIO
from flask_cors import CORS
from PIL import Image
# For input points
import numpy as np

app = Flask(__name__)
#CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/remove-background', methods=['POST'])
def remove_background():
    binary_data = request.data  # Receive the binary data

    # Call rembg to remove the background
    output = remove(binary_data)
    # Convert output to a BytesIO object to send back as a response
    output_io = BytesIO(output).convert("RGBA")
    output_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(output_io, mimetype='image/png')

### With a specific model
@app.route('/remove_background_isnet', methods=['POST'])
def remove_background_isnet():
    binary_data = request.data  # Receive the binary data

    model_name = "isnet-general-use"
    session = new_session(model_name)
    output = remove(binary_data, session=session)
    # Convert output to a BytesIO object to send back as a response
    output_io = BytesIO(output).convert("RGBA")
    output_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(output_io, mimetype='image/png')

### With a Alpa matting
@app.route('/remove_background_alpha', methods=['POST'])
def remove_background_alpha():
    binary_data = request.data  # Receive the binary data

    output = remove(binary_data,  alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11)
    # Convert output to a BytesIO object to send back as a response
    output_io = BytesIO(output)
    output_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(output_io, mimetype='image/png')

### With a post processing
@app.route('/remove_background_post_processing', methods=['POST'])
def remove_background_post_processing():
    binary_data = request.data  # Receive the binary data

    output = remove(binary_data, post_process_mask=True)
    # Convert output to a BytesIO object to send back as a response
    output_io = BytesIO(output).convert("RGBA")
    output_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(output_io, mimetype='image/png')

### Replacing the background color
@app.route('/remove_background_replace_background', methods=['POST'])
def remove_background_replace_background():
    binary_data = request.data  # Receive the binary data

    output = remove(binary_data, bgcolor=(255, 255, 255, 255))
    # Convert output to a BytesIO object to send back as a response
    output_io = BytesIO(output).convert("RGBA")
    output_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(output_io, mimetype='image/png')

### Using input points
@app.route('/remove_background_input_points', methods=['POST'])
def remove_background_input_points():
    binary_data = request.data  # Receive the binary data

    # Define the points and labels
    # The points are defined as [y, x]
    input_points = np.array([[400, 350], [700, 400], [200, 400]])
    input_labels = np.array([1, 1, 2])
    model_name = "isnet-general-use"
    session = new_session(model_name)
    output = remove(binary_data, session=session, input_points=input_points, input_labels=input_labels)
    # Convert output to a BytesIO object to send back as a response
    output_io = BytesIO(output).convert("RGBA")
    output_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(output_io, mimetype='image/png')

if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0' , port=8000)
    app.run(debug=True, port=8000)
