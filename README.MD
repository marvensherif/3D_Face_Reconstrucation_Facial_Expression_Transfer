## 3D Face Reconstruction and Face Expression Transfer Project
This project performs facial expression transfer between two images (source and target). The project uses MediaPipe for facial landmark detection and applies Delaunay triangulation to warp facial expressions from the source image onto the target image.

## Structure
1. `src/`: Contains the source code for the project.
2. `test/`: Contains the test images.
3. `output/`: Folder where the processed images will be saved (generated after running the project).
4. `requirements.txt`: List of dependencies.
5. `3D_Face_Reconstruction_Expression_Transfer_Report`: A detailed Report about 3D Face Reconstruction and Face Expression Transfer.
6. `3D_Face_Reconstruction_Expression_Transfer_Presentation`: Presentation slides summarizing key aspects of the 3D Face Reconstruction and Face Expression Transfer.
7. `Dockerfile`: to build docker image.

## Setup Instructions
1. Clone the repository:
     - git clone https://github.com/marvensherif/3D_Face_Reconstrucation_Facial_Expression_Transfer
     - cd 3D_Face_Reconstrucation_Facial_Expression_Transfer

2. Install the required dependencies:
     - pip install -r requirements.txt

3. Place your test images in the `test/` folder.

4. Run the main script:
      - python src/main.py test/your_image_name.extention test/your_image_name.extention --output_dir output/
      - **ex**: python src/main.py test/source.png test/target.png--output_dir output/

## Running using docker
1. Place your test images in the `test/` folder.
2. docker build -t your_image_name .
3. docker run -it your_image_name
4. python src/main.py test/your_image_name.extention test/your_image_name.extention --output_dir output/
5. docker cp <container_id>:/app/output ./output
- **get container_id from docker ps command**

- output
The [2d_source_landmarks,2d_target_landmarks,3d_source_landmarks[HTML],3d_target_landmarks[HTML],
output_image_after_transfer] will be saved in the output/ folder.
