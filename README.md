# GameSense Tennis (Adobe GenSolve 2024 Finale)
![image](https://github.com/user-attachments/assets/0fc954a7-d2f7-4dd0-8aee-e49003399b04)

## Overview

This project focuses on detecting and analyzing tennis court scenarios using advanced computer vision techniques. It aims to identify players, balls, and keypoints on the tennis court from video frames, providing valuable insights into the game.

GameSense Tennis was developed as part of the final round of Adobe GenSolve 24, where our team, Status 200, was one of the six national finalists. The project was built on Linux Ubuntu, but it is fully compatible and runnable on Windows or macOS systems.

## Models

### YOLOv8
- **Description**: YOLOv8 is a real-time object detection model known for its efficiency and accuracy in detecting various objects, including players and balls in tennis matches.
- **Usage**: This model processes video frames to detect and localize players and balls.

### Additional Models
1. **Trained YOLOv5 Model**
   - **Link**: [Download YOLOv5 Model](https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing)
   - **Description**: A previously trained YOLOv5 model used for object detection. This model serves as a baseline comparison to YOLOv8.

2. **Trained Tennis Court Keypoint Model**
   - **Link**: [Download Tennis Court Keypoint Model](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing)
   - **Description**: A model trained specifically to detect keypoints on the tennis court, such as court boundaries and lines. It helps in accurate visualization and analysis of player and ball positions.

## Installation

Follow these steps to set up and run the project:

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/rishn/Adobe-GenSolve-Finale.git
   cd Adobe-GenSolve24-Finale
   ```

2. **Install Required Packages**

   Install the necessary Python packages listed in the requirements.txt file:

   ```bash
   pip install -r requirements.txt
   ```
3. **Download Model Files**

    Download the pre-trained model files and place them in the models directory of your project:
        Trained YOLOv5 Model
        Trained Tennis Court Keypoint Model

    Ensure that these files are correctly placed in the models directory for the application to function properly.

4. **Usage**

    Make sure you have the video files you want to process. Place them in the input_videos directory.

    Run the Main Script

    Execute the main.py script to start the processing of your videos:

    ```bash
    python main.py
    ```
    This script will process the video files, perform object detection, and visualize the results.

## Research and Results
1. Approach

    Object Detection: Utilized YOLOv8 for real-time object detection of players and balls. YOLOv5 served as a comparative baseline for performance evaluation.

    Court Keypoint Detection: Employed a dedicated model to identify and map keypoints on the tennis court, improving the accuracy of player and ball tracking.

2. Results

    Detection Accuracy: YOLOv8 demonstrated high accuracy and efficiency in detecting players and balls, with YOLOv5 providing a robust comparative analysis.

    Court Visualization: The keypoint detection model accurately mapped the tennis court, facilitating precise tracking and analysis of game elements.

## Demos

https://github.com/user-attachments/assets/789f5ab2-abac-40c5-9b52-91d50785a02e

https://github.com/user-attachments/assets/87ae329c-4c16-4783-abe0-71336734a7b5

https://github.com/user-attachments/assets/307ca66f-b572-482d-9235-e497fb8f7a47

