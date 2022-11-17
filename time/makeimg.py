from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Choose color pallete. Have 4 colors. Each color represented by 2 bits. Each byte of data is represented by 4 colors.
# change brightness


hues = np.linspace(0, 256, 5, dtype=np.uint8)
# hues = (hues + 50) % 256


def gen_hand_coords(rgb_img):
    h, w = rgb_img.shape[:2]
    for y in range(h):
        for x in range(w):
            if rgb_img[y, x, 0] != 0 or rgb_img[y, x, 1] != 0 or rgb_img[y, x, 2] != 0:
                yield y, x


def get_color(img, message, y, x, i):
    message_byte = message[i // 4]
    message_bits = message_byte >> (6 - (i % 4) * 2) & 0b11
    hue = hues[message_bits]
    _, saturation, value = img[y, x]

    def mapRange(value, inMin, inMax, outMin, outMax):
        return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))

    saturation = mapRange(saturation, 0, 255, 20, 235)
    value = mapRange(value, 0, 255, 20, 235)

    return [hue, saturation, value]


def fill_image(message: bytes):
    pic = Image.open('hand_no_bg.png')
    pic = pic.resize((pic.width // 10, pic.height // 10))

    # find best size by binary search
    current_scale = 1
    change = 0.5
    prev_diff = 0
    while True:
        print(f'current scale: {current_scale}')
        w = int(current_scale * pic.width)
        h = int(current_scale * pic.height)

        hands = np.array(pic.resize((w, h)))
        # set background to black
        hands[hands[:, :, 3] < 20] = [0, 0, 0, 255]
        hand_coords = list(gen_hand_coords(hands))

        if len(hand_coords) >= len(message) * 4:
            diff = len(hand_coords) - len(message) * 4
            print(f'difference is {diff}')
            if diff == prev_diff:
                break
            prev_diff = diff
            current_scale -= change
        else:
            current_scale += change

        change /= 2

    hands = np.delete(hands, 3, axis=2)

    hand_coords = list(gen_hand_coords(hands))
    hands = np.array(Image.fromarray(hands).convert('HSV'))

    i = 0
    for y, x in hand_coords:
        if i >= len(message) * 4:
            break
        hands[y, x] = get_color(hands, message, y, x, i)
        i += 1

    smallest_y = min(y for y, x in hand_coords)
    largest_y = max(y for y, x in hand_coords)

    width_of_img = hands.shape[1]
    padding = (width_of_img - (largest_y - smallest_y)) // 2
    # hands = hands[smallest_y:largest_y + 1]
    # hands = hands[smallest_y - padding:largest_y + padding + 1]
    hands = hands[smallest_y - padding:largest_y +
                  padding]

    result = Image.fromarray(hands, 'HSV').convert('RGB')

    # scale result by 200%, do not blur
    result = result.resize(
        (result.width * 8, result.height * 8), resample=Image.NEAREST)
    result.save('result.png')


def main():
    # generate 100000 random bytes of data
    # message = np.random.randint(0, 256, 100000, dtype=np.uint8)
    message = open('msg.zip', 'rb').read()
    fill_image(message)


if __name__ == '__main__':
    main()
