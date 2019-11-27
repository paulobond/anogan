from PIL import Image, ImageDraw
import os

import random


def generate_normal_moon():
    w, h = 100, 100

    center_1 = (50 + random.randint(-5, 5), 50 + random.randint(-5, 5))
    r_1 = random.randint(37, 42)
    x1_1 = center_1[0] - r_1
    x2_1 = center_1[0] + r_1
    y1_1 = center_1[1] - r_1
    y2_1 = center_1[1] + r_1

    center_2 = center_1
    r_2 = random.randint(17, 23)
    x1_2 = center_2[0] - r_2
    x2_2 = center_2[0] + r_2
    y1_2 = center_2[1] - r_2
    y2_2 = center_2[1] + r_2

    shape_1 = [(x1_1, y1_1), (x2_1, y2_1)]
    shape_2 = [(x1_2, y1_2), (x2_2, y2_2)]

    random_angle_offet = random.randint(-30, 30)
    angle_start = 0 + random_angle_offet
    angle_end = 180 + random_angle_offet

    # creating new Image object
    img = Image.new("1", (w, h), color='white')

    # create pieslice image
    img1 = ImageDraw.Draw(img)
    img1.pieslice(shape_1, start=angle_start, end=angle_end, fill="black", outline="black")
    img1.pieslice(shape_2, start=angle_start, end=angle_end, fill="white", outline="white")

    return img


def generate_random_moon_with_anomaly_1():
    """
    Generates a moon that is too long (angle more than 180, excroissance)
    :return:
    """
    w, h = 100, 100

    center_1 = (50 + random.randint(-5, 5), 50 + random.randint(-5, 5))
    r_1 = random.randint(37, 42)
    x1_1 = center_1[0] - r_1
    x2_1 = center_1[0] + r_1
    y1_1 = center_1[1] - r_1
    y2_1 = center_1[1] + r_1

    center_2 = center_1
    r_2 = random.randint(17, 23)
    x1_2 = center_2[0] - r_2
    x2_2 = center_2[0] + r_2
    y1_2 = center_2[1] - r_2
    y2_2 = center_2[1] + r_2

    shape_1 = [(x1_1, y1_1), (x2_1, y2_1)]
    shape_2 = [(x1_2, y1_2), (x2_2, y2_2)]

    random_angle_offet = random.randint(-30, 30)
    angle_start = 0 + random_angle_offet
    angle_end = 180 + random_angle_offet + 45

    # creating new Image object
    img = Image.new("1", (w, h), color='white')

    # create pieslice image
    img1 = ImageDraw.Draw(img)
    img1.pieslice(shape_1, start=angle_start, end=angle_end, fill="black", outline="black")
    img1.pieslice(shape_2, start=angle_start, end=angle_end, fill="white", outline="white")

    return img


def generate_random_moon_with_anomaly_2():
    """
    Generates a moon with a circle next to it
    :return:
    """
    img = generate_normal_moon()
    img1 = ImageDraw.Draw(img)

    img1.ellipse([(10, 10), (20, 20)], fill="black", outline="black")

    return img


def generate_random_moon_with_anomaly_3():
    """
    Generates a moon that is filled in the middle
    :return:
    """
    w, h = 100, 100

    center_1 = (50 + random.randint(-5, 5), 50 + random.randint(-5, 5))
    r_1 = random.randint(37, 42)
    x1_1 = center_1[0] - r_1
    x2_1 = center_1[0] + r_1
    y1_1 = center_1[1] - r_1
    y2_1 = center_1[1] + r_1

    center_2 = center_1
    r_2 = random.randint(17, 23)
    x1_2 = center_2[0] - r_2
    x2_2 = center_2[0] + r_2
    y1_2 = center_2[1] - r_2
    y2_2 = center_2[1] + r_2

    shape_1 = [(x1_1, y1_1), (x2_1, y2_1)]
    shape_2 = [(x1_2, y1_2), (x2_2, y2_2)]

    random_angle_offet = random.randint(-30, 30)
    angle_start = 0 + random_angle_offet
    angle_end = 180 + random_angle_offet

    # creating new Image object
    img = Image.new("1", (w, h), color='white')

    # create pieslice image
    img1 = ImageDraw.Draw(img)
    img1.pieslice(shape_1, start=angle_start, end=angle_end, fill="black", outline="black")
    img1.pieslice(shape_2, start=angle_start, end=angle_end, fill="black", outline="black")

    return img


if __name__ == '__main__':

    N_NORMAL_MOONS = 5000
    N_ANORMAL_MOONS_1 = 2
    N_ANORMAL_MOONS_2 = 2
    N_ANORMAL_MOONS_3 = 2

    os.makedirs("./data/moons", exist_ok=True)
    for i in range(N_NORMAL_MOONS):
        img = generate_normal_moon()
        img.save(f"./data/moons/normal_moon_{i}.png")
    os.makedirs("./test_data/moons", exist_ok=True)
    for i in range(N_ANORMAL_MOONS_1):
        img = generate_random_moon_with_anomaly_1()
        img.save(f"/Users/paulbonduelle/MVA/medical image/AnoGAN/test_data/moons/anormal_moon_1_{i}.png")
    for i in range(N_ANORMAL_MOONS_2):
        img = generate_random_moon_with_anomaly_2()
        img.save(f"/Users/paulbonduelle/MVA/medical image/AnoGAN/test_data/moons/anormal_moon_2_{i}.png")
    for i in range(N_ANORMAL_MOONS_3):
        img = generate_random_moon_with_anomaly_3()
        img.save(f"/Users/paulbonduelle/MVA/medical image/AnoGAN/test_data/moons/anormal_moon_3_{i}.png")
