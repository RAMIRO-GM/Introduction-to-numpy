# -*- coding: utf-8 -*-
"""
Numpy examples
"""
import pandas as pd
import numpy as np

# random matrix 2x4
arr =  10 * np.random.randn(2,5)
print(arr)
# mean of the array
print(arr.mean())
# mean by row
print(arr.mean(axis = 1))
# mean by column
print(arr.mean(axis = 0))
# sum all the elements
print(arr.sum())
# median
print(np.median(arr, axis = 1))


#created a 10 element array of arandoms
unsorted = np.random.randn(10)
print(unsorted)

#sorted array
sorted = np.array(unsorted)
sorted.sort()
print (sorted)

#finding numpy values that exist in the array
array =  np.array([1,2,1,4,2,2])
print(np.unique(array))

s1 = np.array(['desk', 'chair','bulb'])
s2 = np.array(['lamp','bulb','chair'])

#intersection the same elements that are in the arrays
print(np.intersect1d(s1,s2))
# elements in s1 that are not in s2
print(np.setdiff1d(s1,s2))
# which element in s1 is also in s2 boolean
print(np.in1d(s1,s2))

#Broadcasting arithmetic with different size 

start = np.zeros((4,3))
print(start)
add_rows = np.array([1,0,2])
print (add_rows)
y = start + add_rows
print (y)
# now another array  4x1
add_cols = np.array([0,1,2,30])
#apply the transpose to compare the resuls once added to the matrix
add_cols = add_col.T
print (add_cols)
y = start + ad_cols 
print(y)

# Basic image processing
# reading an image
from skimage import data
from scipy import misc
import matplotlib.pyplot as plt

photo_data = misc.imread('./wifire/sd-3layers.jpg')
type(photo_data)

plt.figure()
plt.imshow(photo_data)
photo_data.shape
photo_data.size
photo_data.min()
photo_data.mean()

print("Shape of photo data:", photo_data.shape)
low_value_filter = photo_data < 50
print("Shape of low value:", low_value_filter.shape)
photo_data[low_value_filter] = 0
plt.figure()
plt.imshow(photo_data)

rows_range =  np.arange(len(photo_data))
cols_range = rows_range
print(type(rows_range))
photo_data[rows_range, cols_range] =255
plt.figure()
plt.imshow(photo_data)

total_rows, total_cols, total_layers = photo_data.shape
# ogrid allows the creating of multidimensional n-arrays opereations
X, Y = np.ogrid[:total_rows, :total_cols]
print("X = ",X.shape, "and Y:", Y.shape)

center_row, center_col = total_rows /2 , total_cols/2
dist_from_center = (X - center_row)**2 + (Y - center_col)**2
radious = (total_rows/2)**2
circular_mask = (dist_from_center > radious)
print (circular_mask)
# applying the mask
photo_data = misc.imread('./wifire/sd-3layers.jpg')
photo_data[circular_mask]=0
plt.figure()
plt.imshow(photo_data)

# Boolean result with the same shape as x1 and x2 of the logical AND operation
# on corresponding elements of x1 and x2
X, Y = np.ogrid[:total_rows, :total_cols]
half_upper = X <center_row
half_upper_mask = np.logical_and(half_upper, circular_mask)
photo_data = misc.imread('./wifire/sd-3layers.jpg')
photo_data[half_upper_mask] = 255
plt.figure()
plt.imshow(photo_data)

#RED pixels
photo_data = misc.imread('./wifire/sd-3layers.jpg')
red_mask = photo_data[:,:,0]<150
photo_data[red_mask]=0
plt.figure()
plt.imshow(photo_data)

# composite mask taking the thresholds RGB
photo_data = misc.imread('./wifire/sd-3layers.jpg')
red_mask = photo_data[:,:,0] < 150
green_mask = photo_data[:,:,1] < 100
blue_mask = photo_data[:,:,2] < 100
final_mask = np.logical_and(red_mask, green_mask, blue_mask)
photo_data[final_mask] = 0
plt.figure()
plt.imshow(photo_data)


