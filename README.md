
# Offline Exam AI Monitoring

This project is about detecting cheating behavior in offline examinations using Computer Vision. I trained a YOLOv8 model to identify students and classify whether they are cheating or not. The system works on images and marks each student with a unique ID.

## What the System Does
- Detects students in the exam hall image
- Assigns IDs like STD001, STD002, etc.
- Marks students as **Cheating** or **Not Cheating**
- Saves cropped images of students who are detected as cheating
- Creates a CSV file of results (ID, status, confidence, bounding box)

## Tools and Libraries Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Dataset taken from Roboflow (not included here due to dataset policy)

## Why I Made This
I wanted to try a real-world computer vision use-case. This project helped me understand:
- dataset preparation and annotation
- model training and tuning
- how to use detection results in a practical workflow

## How to Run
Install required libraries:
pip install ultralytics opencv-python

Then run the detection script:

Outputs will be stored as:
- processed image → output_{filename}.jpg
- cropped cheating cases → in `cheaters/` folder
- results log → CSV file

## Notes
- You can train your own dataset by updating `data.yaml`.
- Model file (`best.pt`) can be replaced with your trained weights.
