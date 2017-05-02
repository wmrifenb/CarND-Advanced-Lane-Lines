
# Advanced Lane Finding Project

#### William Rifenburgh

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
### Camera Calibration

The following cell calculates camera calibration parameters, undistorts the calibration images and saves them to the output_images folder. An example of an undistorted calibration image next to its original form is also plotted inline.


```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

% matplotlib inline

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

img = cv2.imread(images[0])
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

def undist_images(images):
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('./output_images/undist_' + fname.split('/')[-1], dst)

    img = cv2.imread(images[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(cv2.undistort(img, mtx, dist, None, mtx))
    ax2.set_title('Undistorted Image', fontsize=15)


undist_images(images)
```


![png](output_1_0.png)


### Undistort test images

The following cell will undistort all test images in the test_images directory, save them to the output_images folder and compare one image's original to its undistorted form.


```python
images = glob.glob('./test_images/*.jpg')

undist_images(images)
```


![png](output_3_0.png)


### Perspective Transformation

The following cells define the functions to perform a perspective transform to road images

---

The following cell allows us to preview where our source and destination points for the prototype of our perspective transformation will go. The red line polygon and cyan line polygon indicate the source and destination regions respectively. The coordinates for the source polygon were chosen to match the lane lines. The destination polygon was chosen to represent what would be a bird's eye view of the lanes.



```python
img = cv2.imread(images[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_size = (img.shape[1], img.shape[0])

src = np.float32([[200, img_size[1]], [1110, img_size[1]], [685, 450], [595, 450]])

img_src_lines = cv2.polylines(img, np.int32([src]), True, (255, 0, 0), thickness=10)
plt.imshow(img_src_lines)

dst = np.float32([[200, img_size[1]], [1110, img_size[1]], [1110, 0], [200, 0]])

img_dst_lines = cv2.polylines(img, np.int32([dst]), True, (0, 255, 255), thickness=10)
plt.imshow(img_dst_lines)
```




    <matplotlib.image.AxesImage at 0x10614c198>




![png](output_5_1.png)


The following cell defines the function that will take the source and destinations we defined along with the camera calibration to undistort and perspective transform an image. The resulting bird's eye view image of this function performed on some of the test images is then plotted.


```python
def undistort_and_transform(img, src, dst, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # and the inverse as well
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv

img = cv2.imread(images[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
warped, M, Minv = undistort_and_transform(img, src, dst, mtx, dist)
warped_dst_lines = cv2.polylines(warped, np.int32([dst]), True, (0, 255, 255), thickness=10)

img = cv2.imread(images[4])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
warped, M, Minv = undistort_and_transform(img, src, dst, mtx, dist)
warped_dst_lines_curved = cv2.polylines(warped, np.int32([dst]), True, (0, 255, 255), thickness=10)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(warped_dst_lines)
ax1.set_title('Straight and Centered Lane Image', fontsize=15)
ax2.imshow(warped_dst_lines_curved)
ax2.set_title('Curved Lane Image', fontsize=15)
```




    <matplotlib.text.Text at 0x1168192e8>




![png](output_7_1.png)


### Color and Gradient Thresholding

The following cell contains code for the function used to identify lane markings using thresholding of sobel x operator convolutions and HLS colorspace transforms. A result of thresholding is shown where the green indicates where x-gradient thresholding criteria were met and blue indicates where color lightness thresholding criteria was met.




```python
image = mpimg.imread('./output_images/undist_test1.jpg')

# Edit this function to create your own pipeline.
def threshold_image(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSL color space and separate the L channel
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsl[:,:,1]
    s_channel = hsl[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return color_binary, combined_binary
    
result, result_binary = threshold_image(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)


ax2.imshow(result)
ax2.set_title('Thresholding Result', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

result = 255 * result.astype("uint8")
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_images/colored_binary.jpg", result)

binary = 255 * result_binary.astype("uint8")
cv2.imwrite("./output_images/combined_binary.jpg", binary)
```




    True




![png](output_9_1.png)


### Lane Line Polynomial Fitting

Below is an example of thresholding and perspective transforming a road image using the functions we have previously developed



```python
image = mpimg.imread('./test_images/test2.jpg')
result, result_binary = threshold_image(image)
warped, M, Minv = undistort_and_transform(result_binary, src, dst, mtx, dist)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)

ax2.imshow(warped, cmap='gray')
ax2.set_title('Threshold-Transform Result', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_11_0.png)


The code cell below will plot a histogram of the bottom half of the binary, perspective transformed image. The  peaks indicate the x-position of both lane lines and where the find_polynomails function will begin its search for the lane lines.


```python
#Plot histogram of threshold-transform binary image
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)
```




    [<matplotlib.lines.Line2D at 0x112d86400>]




![png](output_13_1.png)


The following cell defines the lane finding function which takes in a warped, binary image and fits polynomials to each lane marking side. The functions plotgraph option enables us to preview these lane line polynomials.

Each polynomial was found using numpy's polyfit function. Peaks in histograms of 9 horizontal sections of the binarized image were used to determine the location of a sampling window. The nonzero points found in these windows were used together in the polyfit function for each lane line.

The function find_curvature_and_offset is also defined beforehand so that it may be used in find_polynomials as much of its required arguments are caculated in the find_polynomials function.



```python
def find_curvature_and_offset(leftx, rightx, lefty, righty, img_height_pix, img_width_pix):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature at the base of the image (img_height_pix)
    left_curverad = ((1 + (2 * left_fit_cr[0] * img_height_pix * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = (
                         (1 + (
                         2 * right_fit_cr[0] * img_height_pix * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    img_height_m = img_height_pix * ym_per_pix
    img_width_m = img_width_pix * xm_per_pix

    left_intercept = np.polyval(left_fit_cr, img_height_m)
    right_intercept = np.polyval(right_fit_cr, img_width_m)

    center = (left_intercept + right_intercept) / 2.0

    offset_m = center - img_width_m / 2.0

    return left_curverad, right_curverad, offset_m

def find_polynomials(binary_warped, plotgraph=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if plotgraph:
        plt.figure(figsize=(16, 8))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    l_curverad, r_curverad, offset_m = find_curvature_and_offset(leftx, rightx, lefty, righty, binary_warped.shape[0],
                                                                 binary_warped.shape[1])

    return left_fit, right_fit, l_curverad, r_curverad, offset_m


left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m = find_polynomials(warped, True)

print('left_curverad: ',left_curverad, ' [m], right_curverad: ' , right_curverad, ' [m], offset_m: ' , vehicle_offset_m, ' [m]')
```

    left_curverad:  318.573161466  [m], right_curverad:  259.484867417  [m], offset_m:  0.0966507636163  [m]



![png](output_15_1.png)


To help keep track of lane lines in sharp curves and challenging conditions as well as skip the windowing operations, we can use the polynomials found from the initial find_polynomials function to narrow our search to a region of interest around the previous polynomials for the next frame in a video. The next cell defines the function for this method. A plot showing the region of interest may be shown by setting plotgraph=True.


```python
def find_polynomials_skip_windowing(binary_warped, left_fit, right_fit, plotgraph=False):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if plotgraph:
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.figure(figsize=(16, 8))
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    l_curverad, r_curverad, offset_m = find_curvature_and_offset(leftx, rightx, lefty, righty, binary_warped.shape[0],
                                                                 binary_warped.shape[1])

    return left_fit, right_fit, l_curverad, r_curverad, offset_m

left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m = find_polynomials_skip_windowing(warped, left_fit,
                                                                                                       right_fit, True)

print('left_curverad: ',left_curverad, ' [m], right_curverad: ' , right_curverad, ' [m], offset_m: ' , vehicle_offset_m, ' [m]')
```

    left_curverad:  318.284465766  [m], right_curverad:  216.412606797  [m], offset_m:  0.0811410231584  [m]



![png](output_17_1.png)


The following function draws the green overlay of what our algorithms perceive to be the lane as well as radius of curvature for the lanes and deviation off the center of the lane.


```python
def draw_lane(undistorted_img, warped, left_fit, right_fit, minv, l_radius, r_radius, deviation_offset):
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, minv, (undistorted_img.shape[1], undistorted_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

    # Add Curvature and Deviation Text
    l_radius_txt = str(np.round(l_radius, 2))
    r_radius_txt = str(np.round(r_radius, 2))
    deviation_offset_txt = "{:.3f}".format(deviation_offset)
    font = cv2.FONT_HERSHEY_PLAIN
    curvature_txt = "left_curvature: " + l_radius_txt + " [m], right_curvature: " + r_radius_txt + " [m]"
    deviation_txt = "deviation: " + deviation_offset_txt + " [m]"
    cv2.putText(result, curvature_txt, (30, 60), font, 2, (255, 255, 0), 3)
    cv2.putText(result, deviation_txt, (30, 110), font, 2, (255, 255, 0), 3)
    
    return result

image = mpimg.imread('./test_images/test2.jpg')
undist = cv2.undistort(image, mtx, dist, None, mtx)
result = draw_lane(undist, warped, left_fit, right_fit, Minv, left_curverad, right_curverad, vehicle_offset_m)
plt.figure(figsize=(16, 8))
plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x1174c9ef0>




![png](output_19_1.png)


The following cell defines the process_image function which is used to perfom all previous functions with the exception of find_polynomials_skip_windowing. This function will be applied to each frame of the mp4 project video to generate one with the overlays as in the above picture.


```python
def process_image(image):    
    result, result_binary = threshold_image(image)
    warped, M, Minv = undistort_and_transform(result_binary, src, dst, mtx, dist)
    left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m = find_polynomials(warped, False)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    result = draw_lane(undist, warped, left_fit, right_fit, Minv, left_curverad, right_curverad, vehicle_offset_m)
    return result

image = mpimg.imread('./test_images/test4.jpg')
result = process_image(image)
plt.figure(figsize=(16, 8))
plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x1172eea58>




![png](output_21_1.png)


### Videos

The following three cells use the process image function to create videos overlayed with our lane finding algorithm results. The videos are available here:

[project_video](./output_images/project_video_output.mp4)

[project_video_challenge](./output_images/project_video_challenge_output.mp4)

[project_video_harder_challenge](./output_images/project_video_harder_challenge_output.mp4)




```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

video_output = "./output_images/project_video_output.mp4"
clip1 = VideoFileClip("./project_video.mp4")
clip1_output = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time clip1_output.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video ./output_images/project_video_output.mp4
    [MoviePy] Writing video ./output_images/project_video_output.mp4


    100%|█████████▉| 1260/1261 [03:36<00:00,  5.91it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: ./output_images/project_video_output.mp4 
    
    CPU times: user 3min 59s, sys: 44.1 s, total: 4min 43s
    Wall time: 3min 36s



```python
video_challenge_output = "output_images/project_video_challenge_output.mp4"
clip1 = VideoFileClip("challenge_video.mp4")
clip1_output = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time clip1_output.write_videofile(video_challenge_output, audio=False)
```

    [MoviePy] >>>> Building video output_images/project_video_challenge_output.mp4
    [MoviePy] Writing video output_images/project_video_challenge_output.mp4


    100%|██████████| 485/485 [01:16<00:00,  6.54it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output_images/project_video_challenge_output.mp4 
    
    CPU times: user 1min 28s, sys: 14.7 s, total: 1min 43s
    Wall time: 1min 17s



```python
video_harder_challenge_output = "output_images/project_video_harder_challenge_output.mp4"
clip1 = VideoFileClip("harder_challenge_video.mp4")
clip1_output = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time clip1_output.write_videofile(video_harder_challenge_output, audio=False)
```

    [MoviePy] >>>> Building video output_images/project_video_harder_challenge_output.mp4
    [MoviePy] Writing video output_images/project_video_harder_challenge_output.mp4


    100%|█████████▉| 1199/1200 [03:38<00:00,  5.75it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output_images/project_video_harder_challenge_output.mp4 
    
    CPU times: user 4min 2s, sys: 42.7 s, total: 4min 45s
    Wall time: 3min 39s


## Discussion

The biggest challenges faced in this project were misleading lines, sharp curves in the road, and changes in lighting.
The pipeline was able to handle the first video fairly well. 

The challenge video presented the pipeline with lines that were composed of gradual changes from concrete to ashpalt, paralell to the actual lane lines. This likely caused sobel x gradient lines to diverege and confuse the histogram analysis, forcing the polyfit functions to fit the wrong lines. A sanity check for distance between lane lines as well as a temporal moving average filter on lane line location would have mitigated the asphalt-concrete trickery.

The harder challenge video presented the pipeline with extremely sharp turns and changes in lighting. The changes in lighting most likely caused the color lightness thresholding to become useless as the lane lines became impossible to differentiate in the bright sunlight. A temporal moving average filter on lane line location would have mitigated this problem as the episodes of intense sunlight were brief. The curvature of the road however is something that probably proved too much for the perspective transformation being performed. In the test preview of the perspective transformation function earlier, we saw that the resulting image becomes mostly only what is near the front of the car and not much else. The road curves presented in this video were nearly off the screen and most likely didnt make it past the perspective transformation in any helpful way. Readjusting the perspective transformation or using additional cameras would be required to mitigate effects of such curvyness.



```python

```
