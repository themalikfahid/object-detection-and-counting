# AI Object Counting Using Computer Vision

Name: Fahid Asif Malik

Date: October 23, 2026

## 1. Introduction

This project is a small program that finds objects in a picture or a video and counts them. It can count things like people, cars, buses and bottles. The program draws a box around each object and writes its name and a confidence score. It also shows the total count and the count of each class.

Counting objects by hand is slow, and people make mistakes when there are many objects. A program can do it faster. This kind of counting is useful for traffic counting, checking stock on shelves, and checking how crowded a place is. In this project I also checked how correct the program is by comparing its count with my own manual count.

## 2. Method

I did not train a new model. I used YOLOv8n, which is a ready-made model already trained on the COCO dataset. It can find 80 types of objects. I ran it with the Ultralytics library in Python.

The steps of the program are:

1. Read the image (or a frame from the video) with OpenCV.
2. Run the model on it.
3. Keep only the classes that the user selected, and remove detections whose confidence is lower than the chosen value.
4. Count the detections for each class and also the total.
5. Draw the boxes, labels and confidence scores on the image, and show the counts.
6. Save the result image, a CSV file and a JSON file. The file names include the input file name and the time.

The user interface is a Streamlit web page. The user can upload a picture, choose the classes to count, and move the sliders for confidence, overlap (NMS IoU) and image size. The page also shows a table with the class, confidence and box position of every detected object.

Settings used in my test:

- Model: yolov8n.pt
- Classes selected: person, car
- Confidence threshold: 0.25
- NMS IoU threshold: 0.45
- Inference size: 640

## 3. Results

I tested the program on a picture named images.jpg. It has four different cars in one picture (a 2 by 2 collage). I counted the cars by hand first, and there are 4 cars. The program drew a box on each car and showed a total of 4 cars. No person was detected, which is correct because there is no person in the picture.

The detections given by the program:

| No. | Class | Confidence |
|---|---|---|
| 0 | car | 0.956 |
| 1 | car | 0.934 |
| 2 | car | 0.933 |
| 3 | car | 0.359 |

### Evaluation table

| Image | Class | Manual | Predicted | Error | Reason |
|---|---|---:|---:|---:|---|
| images.jpg | car | 4 | 4 | 0 | Counted correctly |
| images.jpg | person | 0 | 0 | 0 | No person in the picture |

Mean absolute error: 0

Exact matches: 100 %

Bias (plus means too many, minus means too few): 0

This result is from only one picture, so it does not prove that the program is always correct. The picture is also an easy case because the four cars are separate, clear and not hidden by anything. More pictures, especially crowded and dark ones, are needed for a proper test.

## 4. Where the program made mistakes

In this test the count was correct, but the program was not equally sure about all four cars. Three cars had a confidence above 0.93. The fourth car (the grey car in the top left of the picture, seen from the front) had a confidence of only 0.359. It was counted only because the threshold was set to 0.25. If I had set the confidence to 0.40 or higher, this car would have been removed and the count would have been 3, which is wrong.

This shows how the confidence setting changes the count. A low value keeps weak detections (this helped here, but it can also give wrong boxes in other pictures). A high value removes weak detections (this gives a cleaner result, but it can miss real objects). The reason for the low score may be that this car looks different from normal road cars, or that only a part of the car is visible in the picture, but I did not test this further.

## 5. Limits

- Objects that are hidden behind other objects can be missed.
- When objects are very close or touching, two objects can become one box, so the count is too low.
- Dark, blurry or very bright pictures give worse results.
- Very small or far objects are often missed.
- The model only knows its fixed list of 80 classes. It cannot count things like cardboard boxes or a special product unless it is trained again for them.
- It can make false detections, for example it may call a shadow or a pole a person.
- If the confidence value is low, there are more wrong boxes. If it is high, more objects are missed.
- In video, the count is for one frame at a time. The tracker can also give a new ID to the same object, so the line crossing count may not be perfect.
- The results are estimates. For important work they should be checked by a person.

## 6. Privacy

People are counted only as boxes with the label "person". The program does not do face recognition and does not find out who anyone is.

## 7. Conclusion

The program worked well on my test picture. It found all four cars, gave the correct total and showed the box and confidence for each one. The web page was easy to use, and the results can be saved as an image, a CSV file and a JSON file.

The test was small, so I cannot say how the program behaves in harder scenes. It also showed that the confidence setting can change the count, as the fourth car was found with a confidence of only 0.359. To improve the work, I can test more pictures with crowds, dark light and hidden objects, compare yolov8n with a bigger model like yolov8s, and train the model on my own pictures if I need to count objects that are not in the COCO list.