# Flask Face Recognition

## Description
This is a Flask-based web application that uses the `face-recognition` library to perform face recognition and face matching.

## Installation
1. Clone the repository.
2. Install the required dependencies.
3. Run the Flask app.

## Usage
The application exposes two API endpoints: `/api/facematch` and `/api/facecrop`.

The `/api/facematch` endpoint accepts two base64-encoded images and returns whether the faces in the two images match, along with a similarity score.

The `/api/facecrop` endpoint accepts a single base64-encoded image and returns an array of base64-encoded images, representing the cropped faces found in the input image.

## Contributing
Contributions are welcome!

## License
This project is released under the MIT License.
