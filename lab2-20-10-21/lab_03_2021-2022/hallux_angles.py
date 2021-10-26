#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image

from skimage import io
from skimage.morphology import skeletonize, thin
from skimage.util import invert

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.color import rgba2rgb

import networkx as nx
import cv2

mpl.rc('image', cmap='gray')


data = "./input"
images = []

for i in os.listdir(data):
    if '.png' in i:
        image = io.imread(os.path.join(data,i))
        grayscale = rgb2gray(rgba2rgb(image))
        thresh = threshold_otsu(grayscale)
        binary = grayscale > thresh
        images.append(binary)



def display_images(images, columns = 5, rows = 2):
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(rows, columns, i+1)
        img = images[i]
        plt.axis('off')
        plt.imshow(img)
    plt.show()
display_images(images)


skeletonized = []
for i in images:
    skeletonized.append(invert(skeletonize(i)))
display_images(skeletonized[:3], columns = 3, rows = 1)


skeletonized_lee = []
for i in images:
    #skeletonized_lee.append(invert(skeletonize(i, method='lee')))
    skeletonized_lee.append(invert(skeletonize(i)))    
display_images(skeletonized_lee[:3], columns = 3, rows = 1)


skeletonized_thin = []
for i in images:
    skeletonized_thin.append(invert(thin(i)))
display_images(skeletonized_thin[:3], columns = 3, rows = 1)


for i in range(3):
    plt.figure(figsize=(50, 30))
    plt.subplot(1, 4, 1)
    plt.axis('off')
    plt.imshow(images[i])
    plt.subplot(1, 4, 2)
    plt.axis('off')
    plt.imshow(skeletonized[i])
    plt.subplot(1, 4, 3)
    plt.axis('off')
    plt.imshow(skeletonized_lee[i])
    plt.subplot(1, 4, 4)
    plt.axis('off')
    plt.imshow(skeletonized_thin[i])
    plt.show()




def dfs(i, j, visited, tree):
    if (i,j) in visited:
        return 
    visited.add((i,j))
    m = [-1,0,1]
    for k in m:
        for l in m:
            if not image[i+k][j+l]:
                if not (i+k,j+l) in visited:
                    tree.add_node((i+k,j+l))
                    tree.add_edge((i,j), ((i+k,j+l)))
                    dfs(i+k,j+l, visited, tree)

def find_first(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not image[i][j]:
                return i, j
            
def create_tree(image):
    tree = nx.DiGraph()
    start_x, start_y = find_first(image)
    tree.add_node((start_x, start_y))
    dfs(start_x, start_y, set(), tree)
    return tree

def get_longest_path(image):
    tree = create_tree(image)
    return nx.dag_longest_path(tree)

def get_image_with_longest(image, path):
    new_image = np.ones(image.shape, dtype = np.bool)
    for i in path:
        new_image[i[0]][i[1]] = False
    return new_image

def plot_longest(image, path):
    new_image = get_image_with_longest(image, path)
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(new_image)
    plt.show()

def to_binary(image):
    grayscale = rgb2gray(image)
    thresh = threshold_otsu(grayscale)
    binary = grayscale > thresh
    return binary

longest_paths = []
longest_skelets = []
for i in range(len(images)):
    image = to_binary(skeletonized_lee[i])
    print(image)
    lp = get_longest_path(image)
    longest_paths.append(lp)
    longest_skelets.append(get_image_with_longest(image, lp))

for i, p in zip(skeletonized_lee[:5],longest_paths[:5]):
    plot_longest(i,p)


def get_lines(image):
    img = image.copy()
    edges = cv2.Canny(np.uint8(img),0,1,apertureSize = 3)
    minLineLength = 200
    maxLineGap = 10
    return cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 25,minLineLength = 100,maxLineGap = 50)

import math
def get_smallest_angle(lines):
    highest = 100000
    lowest = 0
    a =[]
    b= []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y1 < highest:
                highest = y1
                a = [x1,y1,x2,y2]
            if y2 < highest:
                highest = y2
                a = [x1,y1,x2,y2]
            if y1 > lowest:
                lowest = y1
                b = [x1,y1,x2,y2]
            if y2 > lowest:
                lowest = y2
                b = [x1,y1,x2,y2]
                
    a = [a[2] - a[0], a[3] - a[1]]
    b = [b[2] - b[0], b[3] - b[1]]
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cosine_angle)
    degrees = math.degrees(angle)
    if degrees < 90:
        return 180 - degrees
    if degrees > 180:
        return 360 - degrees
    return degrees

lines = []
for i in range(len(longest_skelets)):
    lines = get_lines(longest_skelets[i])
    img_lines= np.uint8(longest_skelets[i])
    for line in lines: 
        for x1,y1,x2,y2 in line:
            cv2.line(img_lines,(x1,y1),(x2,y2),(255,0,0),2)
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(images[i])
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(longest_skelets[i])
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(img_lines)
    plt.show()
    print("Angle = {}".format(get_smallest_angle(lines)))

