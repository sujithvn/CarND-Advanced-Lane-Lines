# Project 2 - Advanced Lane Finding

#### Submitted as part of Self Driving Car Nanodegree Program by Udacity

---

### **Advanced Lane Finding Project**

**The Goal:**
As the next-level of lane finding (compared to project 1), the goal is to detect the lanes in more complex conditions which includes curves & different lighting condition. We would be using more techniques to achieve this goal.

**Steps:**
1. Initial one-time step of Camera Calibration
    - Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Image Processing
    - Apply a distortion correction to raw images.
    - Use color transforms, gradients, etc., to create a thresholded binary image.
3. Apply a perspective transform to rectify binary image ("birds-eye or top-down view").
4. Detect lane pixels and fit to find the lane boundary.
    - Here we use different algorithms based on the existing FIT availability to increase the efficiency.
    - Also Sanity check is done on the calculated FIT and also the current values of various parameters.
5. Determine the curvature of the lane and vehicle position with respect to center.
6. Warp the detected lane boundaries back onto the original image.
7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



[//]: # "Image References"

[image1]: ./output_images/Undistort_Chess.png "Undistorted Chessboard"
[image2]: ./output_images/Undistort_Road.png "Road Transformed"
[image3]: ./output_images/Thresholded.png "Binary Example"
[image4]: ./output_images/TopDown_RGB.png "Warp RGB Example"
[image5]: ./output_images/TopDown_BIN.png "Warp Binary Example"
[image6]: ./output_images/Histo.png "Histogram"
[image7]: ./output_images/SlideW.png "Sliding Window"
[image8]: ./output_images/PrevFit.png "Previous Fit"
[image9]: ./output_images/InverseTrans.png "Inverse Transform"
[video1]: project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

This is the writeup document for this project. It summarizes the steps involved a high level. Details for each step is available in the accompanying Jupyter Notebook. [Here](https://github.com/sujithvn/CarND-Advanced-Lane-Lines/blob/master/Advanced_Lane_Finding.md) is link to the same.  

### Camera Calibration

#### 1. Computation of the camera matrix and distortion coefficients.

The code for this step is contained in the second code cell of the IPython notebook mentioned above. Please refer the the functions camera_calib() and undisto().  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

In the function _undisto()_, I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

The same process of undistortion was applied on test image also. In the below example image, we can notice the difference in the _**sign-board**_ to the left of the road which appears more paralel to the screen. Also the changes are visible to the bonnet of the car.

![alt text][image2]

#### 2. Gradient Threshold and Color Space thresholds

Here we define three differents ways of applying thresholds on the image for better detection of the shapes (particularly the lanes). The below mentioned functions are in the notebook.

* __abs sobel thresh__ : The SOBEL threshold is done either on x-axis or y-axis and looks for edges along the specified axis.
* __mag thresh__ : The MAGnitude threshold apply a threshold to the overall magnitude of the gradient, in both x and y.
* __dir threshold__ : The GRADient DIRection threshold is used to detect only edges of a particular orientation. The direction of the gradient is simply the inverse tangent (arctangent) of the y gradient divided by the x gradient

In all the cases we can specify the threshold limits and also the size of the kernel (with larger kernel-size effecting a more smoothing effect). 

**Combine above three with Color Spaces & Mask** 
Function used: _apply grad color threshold_

Here we apply (on a trial-error basis) different combinations of the gradient thresholds that we defined above. 
**Color Spaces**
Apart from the above, lane detection can be more effective in different color spaces with different thresholds for each channel within that color space. Here we experiment with both RGB and HLS color spaces. The ROI ( _Region-of-Interest_ ) mask is also defined in an approximate area where we expect to find the lane lines. The final output is threshold with a good combination of above parameters.


![alt text][image3]

#### 3. Perspective transformation.

We apply a perpective transform to get a top-down (bird's eye) view of the the selected region. This is useful for calculating the lane curvature in the following steps.

A set of points in the source image (call it SRC) and corresponding destination points (say DST) where we want to position the SRC points in the destination image is defined. These points are used to calculate the perspective transform 'M' and the inverse perspective transform 'Minv' using the function _cv2.getPerspectiveTransform_.

The perspective transform 'M' is applied on the image to get the transformed image. We also take out the inverse perspective transform 'Minv' to apply back to get original perspective towards the end.

The implementation is detailed in the function _top down view_ in the notebook

![alt text][image4]
![alt text][image5]

#### 4. Identifying the lane-line pixels and fit their positions with a polynomial.

In the next steps, we use the processed image to detect lane pixels and fit to find the lane boundary.

Initially to identify the lane we use the histogram to get a starting point and perform sliding-window algorithm. But as we do not expect the lane positions to change dramatically in each frame, we could focus our lane searching area in subsequent frames to a smaller area (last lane position +/- margin). For this we use an alternate algorithm.

However, we fall back on the sliding window if our sanity check fails in cases like no FIT found, substantial difference from last frame etc.

Below we can see how we identify the initial lane position using histogram.

![alt text][image6]

#### 5. Finding Lane pixels from Image using Sliding Window.
Refer the function _slidingW fit_ in the notebook. The high-level steps are given below:
**Steps:**
* Use the histogram method mentioned above to identify initial position
* Define the hyperparameters
  - number of sliding windows
  - width of the windows defined by margin
  - minimum number of pixels required to recenter window
* Identify the pixels in the defined area
* Check if reset is required based on minpix defined
* Calculate the FIT for the identified lane pixels
  ![alt text][image7]

#### 6. Searching from Prior FIT.
Refer the function _search around poly_ in the notebook. The high-level steps are given below:
**Steps**
* Define the margin
* Identify the active pixels
* Calculate the position using previous FIT details
* Define the new area with a +/- margin to previous position
* Identify the lane in the newly defined area
* Calculate the FIT for the identified lane pixels

![alt text][image8]

#### 7. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

Details notes with formulas are available in the Notebook and the implementation is done via the function _measure_curvature_pixels()_.

#### 8. Inverse Transform.

Finally we draw back the identified lane and the bounding area to the original image. We also add details of the curvature and the off-set of the camera/car from the centre of the lane.  
The function _lane_marker()_ does the inverse transform and updates the lane marking.
The function _data_marker()_ updates Radius and Offset details.
Here is an example of my result on a test image:

![alt text][image9]

#### 9. Putting it all Together.
A new class **'Line()'** sets the initial and reused values across iterations. We also perform sanity check to skip any FITs if it seems to be considerably different from the average of all previous FITs available. 

The function **_complete_pipeline()_** pulls together all the individual funtions that we have detailed above. From within the Video processing section, we call this complete_pipeline function to process each frame.

Details available in the Notebook.

---

### Pipeline (video)

#### 1. Link to the final video output.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### Problems / issues you faced during implementation of this project including the areas where the pipeline is likely fail. 

1. The program fails to detect the lane lines in certain frames. This could be due to extremely bad lighting conditions or faded lane-markings. Reasonable lighting conditions are handled by the program.
2. Bad lane line FITS could also result in high/low Curvature_Radius calculation which is visible in the video. 
3. We have assumed the road to be a flat plane. This is not always possible in the real-world scenario.

####  What to do to make it more robust:

1. Extensive fine-tuning of the thresholds and Color Space filtering could make the program robust to extreme lighting conditions as well.
2. More sanity checking needs to be implemented. This could be comparing the lanes for consistent curve and spacing.
3. Additional code for a better smoothing for various parameters would result in a smoother video experience.