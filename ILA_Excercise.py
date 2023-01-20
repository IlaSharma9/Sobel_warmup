#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement:
# We would like to apply the Sobel Edge filter manually on an image.
# 
# The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter, is used in image processing and computer vision, particularly within edge detection algorithms where it creates an image emphasising edges.
# 
# 
# #### Steps:
# 1. Research the 2-D matrix of Sobel Edge Filter
# 2. Load an image to a 3 dimensional array (Red, Green, Blue channels) (hint: You can use `PIL` library for this
# 3. Pick-up one of the 3 channels and turn that 3D object into a 2D array
# 3. Build a function to apply Sobel Filter as we traverse across the 2D dimension
# 4. Display the resulted image using `matplotlib`

# ![blob.jpg](attachment:blob.jpg)

# In[ ]:


#Student warmup: Ila Sharma , student id:c0852428


# ## Step 1: Research the 2-D matrix of Sobel Edge Filter
# 
# 

# In[9]:



from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

image_file = "blob.jpg"
input_image = imread(image_file)  # this is the array representation of the input image
plt.imshow(input_image, cmap='gray')

[nx, ny, nz] = np.shape(input_image)  # nx: height, ny: width, nz: colors (RGB)


# 
# 
# ## Step 2: Load an image to a 3 dimensional array (Red, Green, Blue channels) (hint: You can use PIL library for this

# In[10]:


r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]

"""## Step 3: Pick-up one of the 3 channels and turn that 3D object into a 2D array"""

gamma = 1.400  # a parameter
r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma


# ## Step 3: Pick-up one of the 3 channels and turn that 3D object into a 2D array

# In[11]:


fig1 = plt.figure(1)
ax1, ax2 = fig1.add_subplot(121), fig1.add_subplot(122)
ax1.imshow(input_image)
ax2.imshow(grayscale_image, cmap=plt.get_cmap('gray'))
fig1.show()


# ## Step 4: Build a function to apply Sobel Filter as we traverse across the 2D dimension

# In[12]:


def sobel_filter(grayscale_image_sample,input_image_file):
  # Here we define the matrices associated with the Sobel filter
  Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
  
  Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
  
  [rows, columns] = np.shape(grayscale_image_sample)  # we need to know the shape of the input grayscale image
  
  sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

  # Now we "sweep" the image in both x and y directions and compute the output
  for i in range(rows - 2):
    for j in range(columns - 2):
      gx = np.sum(np.multiply(Gx, grayscale_image_sample[i:i + 3, j:j + 3]))  # x direction
      gy = np.sum(np.multiply(Gy, grayscale_image_sample[i:i + 3, j:j + 3]))  # y direction
      sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
     # Display the original image and the Sobel filtered image
  fig2 = plt.figure(2)
  ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
  
  ax1.imshow(input_image_file)
  
  ax2.imshow(sobel_filtered_image, cmap=plt.get_cmap('gray'))
  
  fig2.show()


# ## Step 5: Display the resulted image using matplotlib

# In[13]:



sobel_filter(grayscale_image,input_image)


# In[ ]:




