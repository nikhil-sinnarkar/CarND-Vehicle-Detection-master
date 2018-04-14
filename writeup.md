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

I started by reading in all the `vehicle` and `non-vehicle` images. I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![6](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/writeup/6.png)
![7](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/writeup/7.png)

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

The final feature vector length for the selected parameters was 1208.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is contained in cell 7 of my jupyter notebook.

Training the classifier was a iterative process for me. I would set the parameters and train the classifier and run the classifier on test data to see the accuracy. With the linear SVC I got around only 95% accuracy. This was causing many false detections in the image. So I tried SVC with `rbf` kernel. This increased the accuracy to 97%. I again tried to fine tune my parameters and once I got the desired accuracy I stopped the iteration.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First I analysed a few random frames from the project video to get an idea of where exactaly the road is and where could the carrs be. This gave me a rough estimate of the area over which I needed to slide the window to detect vehicles. Lets call this area as `scan_area`. I took windows of 4 different sizes - 56x56, 64x64, 96x96 and 128x128. Now, I don't have to slide all the windows over the full `scan_area` this is because the size of the vehicle changes depending on where it is in the `scan_area`. So I set the start and stop positions of my 4 different size windows accordingly. Here is an example of it:

![1](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/writeup/1.png)

As for setting up the overlap values, the more the overlap is the better it is. But there is a disadvantage of increasing the overlap between the windows. It increases the number of windows and inturn the number of computations performed for every frame of the video. On the flip side if I reduce the overlap I was loosing on some of the detections. As a starting point I kept the overlap to 50% and started increasing it until a point where I didn't loose any detection for any of the frame. The overlap value of 75% worked for my case. But for the windows of size 96x96 I had to overlap them to 80% to accomodate for the vehicles coming in from the side and moving forward.

The setting up of window sizes is done inside `pipeline()` method which is defined in `VideoProcessor()` class. Once I have the list of all the windows I pass them to `run_classifier()`. This method runs the classifier on all the windows and the returns only the windows where it detects a vehicle. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tweaked the parameters of to get the best result from the classifier. I experimented with different color spaces and found that YCrCb was giving the best result with channel 0. To this I also added spatial and color histogram featues. So after setting up of all the parameters I took a test run on few images to see how well the classifier is working. Here are some of the examples:


![2](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/writeup/2.png)

Note: All the green boxes show window where I get a positive detection for vehicle.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/video_output/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As mentioned earlier the `run_classifier()` method returns the coordinates of all the windows where I get a positive detections. I then pass all these coordinates to `generate_heatmap()`. This method outputs a single channel image with pixel values incremented by 1 at all the window locations given as input to it.

Next, I pass this heatmap to `avg_heatmap()` which takes average of previous 7 heatmaps and returns the averahe heatmap. I did this to reduce the wobble in drawing the boxes around detected vehicles. 
After I get the average heatmap I apply a threshold over it to eliminate the false detections if any. After experimenting several times I settled to a threshold value of 7.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing various steps:

![3](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/writeup/3.png)
![4](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/writeup/4.png)
![5](https://github.com/nikhil-sinnarkar/CarND-Vehicle-Detection-master/blob/master/writeup/5.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced several issues during making my pipeline. One of the issue was improving the accuracy of the classifier as I was getting many false positives. For this I changed my kernel from linear to rbf. This helped me to reduce the number of false positive. I also added some of my own images to the training dataset, this further reduced the number of false positives. The next major issue I faced was setting up the dimensions of the sliding windows. I couldn't find a better way to do this other than trial and error. I tried various combination of different window sizes and once I got good enough results I stopped experimenting with it. I think it could be improved further. 

Another issue that I faced was that my bounding box was too wobbly. I resolved this issue to most extent by averaging my heatmap for 7 frames and then using it to draw the bounding boxes.

Talking about improvements, I think there are several. As discussed earlier one of them is to set the size and position of windows such that we cover maximum area with minimum number of windows without loosing any positive detection. The lesser the number of windows the faster it is to iterate over each frame. Extracting features is another area of improvement. We can try and minimize the feature vector length. At the end we can also improve the accuracy of the classifier. Currently for me the the accuracy was 97.39% which can further be improved.

