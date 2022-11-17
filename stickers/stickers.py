# overlay stickers to achieve maximum similarity to reference
from PIL import Image, ImageOps
import numpy as np
from scipy.cluster.vq import kmeans
import tqdm
from matplotlib import pyplot as plt
import sys
import os


def map_img_to_colors(img, colors):
    # flatten the image
    recolored = np.array(img.convert('LA'), dtype=np.float32).reshape(-1, 2)
    visible = recolored[:, 1] > 0

    # recolor not transparent pixels
    recolored[visible, 0] = np.array(
        [min(colors, key=lambda x:abs(x-i)) for i in recolored[visible, 0]])

    # remove transparency if below threshold
    recolored[recolored[:, 1] < 100, 1] = 0

    # resize and cast to ints
    return np.uint8(recolored.reshape((img.height, img.width, 2)))


def load_reference(filename, min_dim_size):
    reference = Image.open(filename).convert('RGBA')
    ratio = reference.width / reference.height
    reference = reference.resize(
        (min_dim_size, int(min_dim_size / ratio)), Image.Resampling.LANCZOS)
    return reference


def load_sticker(filename, size):
    sticker = Image.open(filename).convert('RGBA')
    max_dim = max(sticker.size)
    padding = (max_dim - sticker.size[0]) // 2
    sticker = ImageOps.expand(sticker, border=(padding, 0, padding, 0), fill=0)
    sticker = sticker.resize((size, size), Image.Resampling.NEAREST)
    return sticker


def calculate_colors(img, n_colors):
    img = np.array(img, dtype=np.float32).reshape(-1, 2)
    visible = img[:, 1] > 0
    colors, _ = kmeans(img[visible, 0], n_colors)
    return sorted(round(a) for a in colors)


def construct_rotations(sticker):
    img = Image.fromarray(np.array(sticker))
    rotations = []
    for i in range(0, 360, 5):
        rotations.append(np.array(img.rotate(i)))
    return rotations


def calculate_closeness(reference, result, sticker, y, x):
    '''smaller is better'''
    reference_window = reference[y:y+sticker.shape[0], x:x+sticker.shape[1]]
    result_window = np.array(
        result[y:y+sticker.shape[0], x:x+sticker.shape[1]], dtype=np.float32)
    result_window_copy = result_window.copy()

    sticker_mask = sticker[:, :, 1] > 0
    result_window_copy[sticker_mask] = sticker[sticker_mask]

    error_with_sticker = np.sum(
        (reference_window[:, :, 0] - result_window_copy[:, :, 0]) ** 2)
    error_original = np.sum(
        (reference_window[:, :, 0] - result_window[:, :, 0]) ** 2)

    return error_with_sticker - error_original


def find_best_match_random(reference, result, rotations, num_iterations=1000):
    reference_height, reference_width, _ = reference.shape
    sticker_height, sticker_width, _ = rotations[0].shape

    cur_best = (None, None, float('inf'))
    for _ in range(num_iterations):
        reference_y = np.random.randint(0, reference_height - sticker_height)
        reference_x = np.random.randint(0, reference_width - sticker_width)

        for i, orientation in enumerate(rotations):
            score = calculate_closeness(
                reference, result, orientation, reference_y, reference_x)
            if score < cur_best[2]:
                cur_best = (i, (reference_y, reference_x), score)

    return cur_best


def place_sticker(result, sticker, y, x, alpha_axis=1):
    visible = sticker[:, :, alpha_axis] > 0
    result_window = result[y:y+sticker.shape[0], x:x+sticker.shape[1]]
    result_window[visible] = sticker[visible]


def fill_image(reference, stickers: list[list[np.array]], num_stickers: list[int]):
    result = np.zeros_like(reference)

    placements = []

    # place sticker one at a time until num_stickers is 0
    total = sum(num_stickers)
    progress_bar = tqdm.tqdm(total=total, desc="Placing stickers")
    i = 0
    while any(s != 0 for s in num_stickers):
        sticker_rotations = stickers[i]
        if num_stickers[i] == 0:
            i = (i + 1) % len(stickers)
            continue

        # find best match
        best_match = find_best_match_random(
            reference, result, sticker_rotations, num_iterations=300)
        best_rotation, (y, x), _ = best_match

        # place sticker
        place_sticker(result, sticker_rotations[best_rotation], y, x)
        num_stickers[i] -= 1
        placements.append((i, best_rotation, (y, x)))

        progress_bar.update()

        i = (i + 1) % len(stickers)

    return result, placements


def save_placements(placements, filename):
    with open(filename, 'w') as f:
        for i, r, (y, x) in placements:
            f.write(f"{i} {r} {y} {x}\n")


def load_placements(filename):
    placements = []
    with open(filename, 'r') as f:
        for line in f:
            i, r, y, x = line.split()
            placements.append((int(i), int(r), (int(y), int(x))))
    return placements


def save_final(reference, stickers: list[list[np.array]], placements, filename):
    result = np.zeros_like(reference)
    for i, r, (y, x) in placements:
        place_sticker(result, stickers[i][r], y, x)
    Image.fromarray(result).save(filename)


def show_placements(reference, stickers: list[list[np.array]], placements, out_dir):
    '''
    reference and stickers must both be rgba
    '''

    # make every visible pixel red
    for sticker_list in stickers:
        for sticker in sticker_list:
            visible = sticker[:, :, 3] > 0
            sticker[visible, 0] = 0
            sticker[visible, 1] = 255
            sticker[visible, 2] = 255

    index = 0
    for i, r, (y, x) in tqdm.tqdm(placements):
        result = np.zeros_like(reference)
        place_sticker(result, stickers[i][r], y, x, alpha_axis=3)
        # draw bounding box around sticker
        result[y, x:x+stickers[i][r].shape[1], :] = [255, 0, 0, 255]
        result[y+stickers[i][r].shape[0]-1, x:x +
               stickers[i][r].shape[1], :] = [255, 0, 0, 255]
        result[y:y+stickers[i][r].shape[0], x, :] = [255, 0, 0, 255]
        result[y:y+stickers[i][r].shape[0], x+stickers[i]
               [r].shape[1]-1, :] = [255, 0, 0, 255]

        img = Image.fromarray(result)
        # save image
        img.save(f"{out_dir}/{index}.png")
        index += 1


def main():
    sticker_size = 32
    reference_size = 300
    num_stickers = 88

    fire_sticker = load_sticker('fire.png', sticker_size)
    # 0 because background color is black
    fire_colors = calculate_colors(fire_sticker, 3)

    top_left_sticker_original = load_sticker('topleft.png', sticker_size)
    top_left_rotations_original = construct_rotations(
        top_left_sticker_original)
    top_left_sticker_recolored = map_img_to_colors(
        top_left_sticker_original, fire_colors)
    top_left_rotations_recolored = construct_rotations(
        top_left_sticker_recolored)

    bottom_right_sticker_original = load_sticker(
        'bottomright.png', sticker_size)
    bottom_right_rotations_original = construct_rotations(
        bottom_right_sticker_original)
    bottom_right_sticker_recolored = map_img_to_colors(
        bottom_right_sticker_original, fire_colors)
    bottom_right_rotations_recolored = construct_rotations(
        bottom_right_sticker_recolored)

    reference_original = load_reference('alma.png', reference_size)
    reference_recolored = map_img_to_colors(
        reference_original, [0] + fire_colors)

    result = fill_image(
        reference_recolored, [top_left_rotations_recolored, bottom_right_rotations_recolored], [num_stickers, num_stickers])

    _, placements = result

    out_dir = sys.argv[1]

    try:
        os.mkdir(out_dir)
    except:
        pass

    save_placements(placements, 'placements.txt')
    save_final(reference_original, [top_left_rotations_original,
               bottom_right_rotations_original], placements, f'{out_dir}/final_original.png')
    show_placements(reference_original, [
                    top_left_rotations_original, bottom_right_rotations_original], placements, out_dir)


if __name__ == '__main__':
    main()
