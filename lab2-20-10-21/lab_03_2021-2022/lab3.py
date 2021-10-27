import os

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import skeletonize
from skimage.util import invert
import math


def display_image(image, title):
    plt.axis('off')
    plt.title(title)
    plt.imshow(image)
    plt.show()


def remove_green_color(image):
    green_pixels_mask = np.any(image != [0, 255, 0], axis=-1)
    black_pixels_mask = np.any(image != [32, 32, 32], axis=-1)
    image[~green_pixels_mask] = [0, 0, 0]
    image[~black_pixels_mask] = [0, 0, 0]
    return image


def greyscale(image):
    grayscale = rgb2gray(image)
    binary = grayscale > 0
    return binary


def dfs(i, j, visited, tree, image):
    if (i, j) in visited:
        return
    visited.add((i, j))
    m = [-1, 0, 1]
    for k in m:
        for l in m:
            if not image[i + k][j + l]:
                if not (i + k, j + l) in visited:
                    tree.add_node((i + k, j + l))
                    tree.add_edge((i, j), (i + k, j + l))
                    dfs(i + k, j + l, visited, tree, image)


def find_first(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not image[i][j]:
                return i, j


def create_tree(image):
    tree = nx.DiGraph()
    start_x, start_y = find_first(image)
    tree.add_node((start_x, start_y))
    dfs(start_x, start_y, set(), tree, image)
    return tree


def get_longest_path(image):
    tree = create_tree(image)
    return nx.dag_longest_path(tree)


def get_image_with_longest(image, path):
    new_image = np.ones(image.shape, dtype=bool)
    for i in path:
        new_image[i[0]][i[1]] = False
    return new_image


def get_lines(image):
    img = image.copy()
    edges = cv2.Canny(np.uint8(img), 0, 1, apertureSize=3)
    return cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=25, minLineLength=100, maxLineGap=50)


def line_function(lines):
    # y = m*x + b
    line_functions = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y1 - y2) / (x1 - x2)
            b = (x1 * y2 - x2 * y1) / (x1 - x2)
            line_functions.append((m, b))

    return line_functions


def filter_lines(lines):
    threshold = 1
    filtered = {}
    result = []
    for m, b in lines:
        a = int(m)
        a_set = a
        for key in filtered.keys():
            if a - threshold <= int(key) <= a + threshold:
                a_set = key
        if a_set in filtered:
            filtered[a_set].append((m, b))
        else:
            filtered[a_set] = [(m, b)]

    for value in filtered.values():
        m_acc = 0
        b_acc = 0
        i = 0
        for m, b in value:
            m_acc = m_acc + m
            b_acc = b_acc + b
            i = i + 1
        result.append((m_acc / i, b_acc / i))

    return result

def get_angle(lines):
    m1, b1 = lines[0]
    m2, b2 = lines[1]

    angle_in_degrees1 = math.degrees(math.atan(m1)) if math.degrees(math.atan(m1)) >= 0 else 180 + math.degrees(math.atan(m1))
    angle_in_degrees2 = math.degrees(math.atan(m2)) if math.degrees(math.atan(m2)) >= 0 else 180 + math.degrees(math.atan(m2))

    return angle_in_degrees1 - angle_in_degrees2 if angle_in_degrees1 > angle_in_degrees2 else angle_in_degrees2 - angle_in_degrees1


mpl.rc('image', cmap='gray')
data = "./bone_seg"

for i in os.listdir(data):
    if '.png' in i:
        image = io.imread(os.path.join(data, i))

        # 1. raw image
        display_image(image, 'raw image')

        # 2. segment fingers based on colors
        no_green_image = remove_green_color(image)
        # display_image(no_green_image, 'no green color')

        # 3. convert to binary image
        binary = greyscale(no_green_image)
        # display_image(binary, 'binary')

        # 4. skeletonize
        skeletonized = invert(skeletonize(binary))
        # display_image(skeletonized, 'skeletonize')

        # 5. skeletonize lee
        skeletonized_lee = invert(skeletonize(binary, method='lee'))
        # display_image(skeletonized_lee, 'skeletonize lee')

        # 6. skeletonize thin
        # skeletonized_thin = invert(thin(binary))
        # display_image(skeletonized_thin, 'skeletonized thin')

        # 7. longest paths
        first_longest_path = get_longest_path(skeletonized_lee)
        first_longest_skelets = get_image_with_longest(skeletonized_lee, first_longest_path)
        # display_image(first_longest_skelets, 'first longest path')

        binary_skeletonized_lee = skeletonized_lee > 0
        second_path = binary_skeletonized_lee + ~first_longest_skelets
        second_longest_path = get_longest_path(second_path)
        second_longest_skelets = get_image_with_longest(second_path, second_longest_path)
        # display_image(second_longest_skelets, 'second longest path')

        two_longest_paths = first_longest_skelets * second_longest_skelets
        # display_image(two_longest_paths, 'two longest paths')

        # 8. detect lines
        lines = get_lines(two_longest_paths)
        img_lines = np.uint8(two_longest_paths)
        line_functions = line_function(lines)

        for m, b in line_functions:
            # y = m*x + b
            cv2.line(img_lines, (int((0 - b) / m), 0), (int((1024 - b) / m), 1024), (255, 0, 0), 1)
        # display_image(img_lines, 'detected lines')

        # 9. filter lines
        new_lines = filter_lines(line_functions)
        print(str.format('Kat: {}', get_angle(new_lines)))
        new_img_lines = np.uint8(np.ones(binary.shape))
        new_img_lines = invert(new_img_lines * 255).round().astype(np.uint8)
        for m, b in new_lines:
            cv2.line(new_img_lines, (int((0 - b) / m), 0), (int((1024 - b) / m), 1024), (255, 0, 0), 1)
        # display_image(new_img_lines, 'filtered lines')

        # 10. combine images together
        binary = (binary * 255).round().astype(np.uint8)
        rgb_lines = cv2.cvtColor(new_img_lines, cv2.COLOR_GRAY2RGB)
        rgb_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        white_pixels_mask = np.any(rgb_lines != [255, 255, 255], axis=-1)
        rgb_binary[~white_pixels_mask] = [255, 0, 0]

        display_image(rgb_binary, 'final segmentation')

