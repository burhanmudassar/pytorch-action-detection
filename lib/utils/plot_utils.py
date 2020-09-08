import numpy as np
import math
import random
from itertools import product as product
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import PIL.ImageFont as ImageFont

color_dict = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
    'pink': (255, 153, 255),
    'orange': (255, 128, 0)
}


def format_display_str(draw, box, label, score):
    base_font_size = 20
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', base_font_size)
    except IOError:
        font = ImageFont.load_default()

    if label is None:
        return
    display_str_list = ['{}'.format(label)]
    if score is not None:
        display_str_list[0] += '{:.3f}'.format(score)
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if box[1] > total_display_str_height:
        text_bottom = box[1]
    else:
        text_bottom = box[3] + total_display_str_height

    for display_str in display_str_list[::-1]:
        current_size = base_font_size
        text_width, text_height = font.getsize(display_str)
        box_width = (box[2] - box[0]) + 1
        while text_width > box_width and current_size > 10:
            current_size -= 1
            font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', current_size)
            text_width, text_height = font.getsize(display_str)

        total_display_str_height = (1 + 2 * 0.05) * sum([text_height])
        if box[1] > total_display_str_height:
            text_bottom = box[1]
        else:
            text_bottom = box[3] + total_display_str_height

        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(box[0], text_bottom - text_height - 2 * margin), (box[0] + text_width,
                                                                text_bottom)],
            fill=(0, 0, 255, 128))
        draw.text(
            (box[0] + margin, text_bottom - text_height - margin),
            display_str,
            fill='white',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_box_with_str(im_to_plot, box, label, score, color):
    im_to_plot = Image.fromarray(np.uint8(im_to_plot))
    draw = ImageDraw.Draw(im_to_plot, 'RGBA')
    # Draw the box
    draw.line([(box[0], box[1]), (box[0], box[3]),
               (box[2], box[3]),
               (box[2], box[1]), (box[0], box[1])], width=10,
              fill=color_dict[color])
    # Draw the string
    format_display_str(draw, box, label, score)
    im_to_plot = np.array(im_to_plot)
    return im_to_plot

def draw_centroid_with_str(im_to_plot, box, label, score, color, opacity=1.0):
    im_to_plot = Image.fromarray(np.uint8(im_to_plot))
    draw = ImageDraw.Draw(im_to_plot, 'RGBA')

    center_point = (box[:2] + box[2:])//2
    width_center_point = 10

    # fillcolor = tuple((np.asarray(color_dict[color]) * opacity).astype(np.int32))
    fillcolor = tuple(list(color_dict[color]) + [opacity])

    # Draw the box
    draw.ellipse((center_point[0]-width_center_point, center_point[1]-width_center_point,
               center_point[0]+width_center_point, center_point[1]+width_center_point),
                fill=fillcolor, outline=fillcolor)
    # Draw the string
    format_display_str(draw, box, label, score)
    im_to_plot = np.array(im_to_plot)
    return im_to_plot


def plot_boxes_on_image(im_to_plot, box, label, score, color=None):
    for ind, box_ in enumerate(box):
        label_ = label[ind]
        if score is not None:
            score_ = score[ind]
        else:
            score_ = None

        if color is None:
            color = random.choice(['green', 'blue', 'yellow', 'pink', 'white', 'red', 'orange'])
        im_to_plot = draw_box_with_str(im_to_plot, box_, label_, score_, color[ind])

    return im_to_plot

def plot_centroid_on_image(im_to_plot, box, label, score, color=None, opacity=None):
    for ind, box_ in enumerate(box):
        if label is not None:
            label_ = label[ind]
        else:
            label = None
        if score is not None:
            score_ = score[ind]
        else:
            score_ = None

        if color is None:
            color = random.choice(['green', 'blue', 'yellow', 'pink', 'white', 'red', 'orange'])
        im_to_plot = draw_centroid_with_str(im_to_plot, box_, label_, score_, color[ind])

    return im_to_plot

def plot_centroid_tube_on_image(im_to_plot, box, color, time_opacity=True):
    for ind, box_ in enumerate(box):

        if time_opacity is True:
            opacity = int((len(box) - ind) / len(box) * 255.0)
        else:
            opacity = 255
        im_to_plot = draw_centroid_with_str(im_to_plot, box_, None, None, color, opacity)

    return im_to_plot