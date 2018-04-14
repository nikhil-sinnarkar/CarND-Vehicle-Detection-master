## Vehicle Detection using Support Vector Machine
### Project Writeup

---

**Outline**

The goals / steps of this project are the following:
* Extract features from images to train the SVM classifier. The features include Histogram of Oriented Gradients (HOG), binned color features and histogram of color.
* Preprocess the extracted features. This includes normalization, shuffling and splitting the data into train and test set.
* Train the SVM classifier to detect vehicles.
* Extract parts of image using sliding window and classify it as car/non-car.
* Eliminate false detections and draw a bounding box around the detected cars.
* Build a pipeline and run it on the video.


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the IPython notebook. For extraction of HOG features I defined a function `get_hog_features()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that the HOG image for the below parameters worked well. Mainly the image showed a good difference in structure for car and non-car image inputs. Obviously there were other combination of parameters which worked better but they resulted in more number of features per image. 

| HOG parameters         		|     Value           |
|:-------------------------:|:-------------------:|
| Orientations              | 5                   |
| Pixels per cell           | 8                   |
| Cells per block           | 2                   |
| Hog channel               | 0                   |
| Color space               | YCrCb               |

Along with HOG features I also used binned spatial and color histogram features. The parameters for them were:

Number of bins for spatial binnig : 8

Number of bins for color histogram : 12

The final feature vector length for the selected parameters was 1226.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is contained in cell 7 of my jupyter notebook.

Training the classifier was a iterative process for me. I would set the parameters and train the classifier and run the classifier on test data to see the accuracy. With the linear SVC I got around only 96% accuracy. This was causing many false detections in the image. So I tried SVC with `rbf` kernel. This increased the accuracy to 98%. I again tried to fine tune my parameters and once I got the desired accuracy I stopped the iteration.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First I analysed a few random frames from the project video to get an idea of where exactaly the road is and where could the carrs be. This gave me a rough estimate of the area over which I needed to slide the window to detect vehicles. Lets call this area as `scan_area`. I took windows of 4 different sizes - 56x56, 64x64, 96x96 and 128x128. Now, I don't have to slide all the windows over the full `scan_area` this is because the size of the vehicle changes depending on where it is in the `scan_area`. So I set the start and stop positions of my 4 different size windows accordingly. Here is an example of it:

![alt text][image3]

As for setting up the overlap values, the more the overlap is the better it is. But there is a disadvantage of increasing the overlap between the windows. It increases the number of windows and inturn the number of computations performed for every frame of the video. On the flip side if I reduce the overlap I was loosing on some of the detections. As a starting point I kept the overlap to 50% and started increasing it until a point where I didn't loose any detection for any of the frame. The overlap value of 75% worked for my case. But for the windows of size 96x96 I had to overlap them to 80% to accomodate for the vehicles coming in from the side and moving forward.

The setting up of window sizes is done inside `pipeline()` method which is defined in `VideoProcessor()` class. Once I have the list of all the windows I pass them to 'run_classifier()`. This method runs the classifier on all the windows and the returns only the windows where it detects a vehicle. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

