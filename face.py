import json
import math

import cv2 as cv
import numpy as np
import os
import numpy
from tha3.util import resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
# Load the models.
import torch
from tha3.poser.modes.pose_parameters import get_pose_parameters
from tqdm import trange

# 加载我们的设备
MODEL_NAME = "standard_float"
DEVICE_NAME = 'cuda'
device = torch.device(DEVICE_NAME)
FRAME_RATE = 30.0
last_torch_input_image = None
torch_input_image = None
default_img = None


# 加载姿势模型
def load_poser(model: str, device: torch.device):
    print("Using the %s model." % model)
    if model == "standard_float":
        from tha3.poser.modes.standard_float import create_poser
        return create_poser(device)
    elif model == "standard_half":
        from tha3.poser.modes.standard_half import create_poser
        return create_poser(device)
    elif model == "separable_float":
        from tha3.poser.modes.separable_float import create_poser
        return create_poser(device)
    elif model == "separable_half":
        from tha3.poser.modes.separable_half import create_poser
        return create_poser(device)
    else:
        raise RuntimeError("Invalid model: '%s'" % model)


poser = load_poser(MODEL_NAME, DEVICE_NAME)
poser.get_modules()

# 获取姿势参数
pose_parameters = get_pose_parameters()
pose_size = poser.get_num_parameters()
last_pose = torch.zeros(1, pose_size, dtype=poser.get_dtype()).to(device)

iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
head_x_index = pose_parameters.get_parameter_index("head_x")
head_y_index = pose_parameters.get_parameter_index("head_y")
neck_z_index = pose_parameters.get_parameter_index("neck_z")
body_y_index = pose_parameters.get_parameter_index("body_y")
body_z_index = pose_parameters.get_parameter_index("body_z")
breathing_index = pose_parameters.get_parameter_index("breathing")


# 修改透明背景为白色背景图
def transparent2white(img):
    black_pixels = np.where(
        (img[:, :, 0] == 0) &
        (img[:, :, 1] == 0) &
        (img[:, :, 2] == 0)
    )
    # set those pixels to white
    img[black_pixels] = [255, 255, 255]
    # sp = img.shape  # 获取图片维度
    # width = sp[0]  # 宽度
    # height = sp[1]  # 高度
    # for yh in range(height):
    #     for xw in range(width):
    #         color_d = img[xw, yh]  # 遍历图像每一个点，获取到每个点4通道的颜色数据
    #         if len(color_d) == 3 and color_d[0] == 0 and color_d[1] == 0 and color_d[2] == 0:
    #             img[xw, yh] = [255,255,255]
            # if len(color_d) > 3:
            #     if color_d[3] == 0:  # 最后一个通道为透明度，如果其值为0，即图像是透明
            #         img[xw, yh] = [255, 255, 255, 255]  # 则将当前点的颜色设置为白色，且图像设置为不透明

    return img


def get_pose(eye_left_slider, eye_right_slider, mouth_left_slider, head_x_slider, head_y_slider):
    pose = torch.zeros(1, pose_size, dtype=poser.get_dtype())
    # 0-1
    # 左边眉毛
    eyebrow_left_slider = 0.1
    # 右边眉毛
    # 0-1
    eyebrow_right_slider = 0.1
    # 眉毛下降类型
    # "troubled", "angry", "lowered", "raised", "happy", "serious"
    eyebrow_dropdown = "troubled"
    # ["wink", "happy_wink", "surprised", "relaxed", "unimpressed", "raised_lower_eyelid"]
    # 眼睛下架类型
    eye_dropdown = "wink"
    # 左眼
    # 0-1
    # eye_left_slider = 0.2
    # 右眼
    # 0-1
    # eye_right_slider = 0.2
    # 嘴巴下降类型
    # ["aaa", "iii", "uuu", "eee", "ooo", "delta", "lowered_corner", "raised_corner", "smirk"]
    mouth_dropdown = "aaa"
    # 嘴巴只调左边即可
    # mouth_left_slider = 0.2
    mouth_right_slider = 0.3
    # 瞳孔大小
    # 0-1
    iris_small_left_slider = 0.1
    iris_small_right_slider = 0.2
    # 眼睛往上还是往下
    # -1-1
    iris_rotation_x_slider = 0.0
    iris_rotation_y_slider = 0.1
    # 头往上还是往下
    # head_x_slider = 0.1
    # 头往左还是往右
    # head_y_slider = 0.2
    # 左右摇头
    neck_z_slider = 0.1
    # 身体往左还是往右
    body_y_slider = 0.2
    # 身体旋转
    body_z_slider = 0.0
    # 0-1
    breathing_slider = 0.2

    eyebrow_name = f"eyebrow_{eyebrow_dropdown}"
    eyebrow_left_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_left")
    eyebrow_right_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_right")

    pose[0, eyebrow_left_index] = eyebrow_left_slider
    pose[0, eyebrow_right_index] = eyebrow_right_slider
    # 0-1
    eye_name = f"eye_{eye_dropdown}"
    eye_left_index = pose_parameters.get_parameter_index(f"{eye_name}_left")
    eye_right_index = pose_parameters.get_parameter_index(f"{eye_name}_right")
    pose[0, eye_left_index] = eye_left_slider
    pose[0, eye_right_index] = eye_right_slider

    mouth_name = f"mouth_{mouth_dropdown}"
    if mouth_name == "mouth_lowered_corner" or mouth_name == "mouth_raised_corner":
        mouth_left_index = pose_parameters.get_parameter_index(f"{mouth_name}_left")
        mouth_right_index = pose_parameters.get_parameter_index(f"{mouth_name}_right")
        pose[0, mouth_left_index] = mouth_left_slider
        pose[0, mouth_right_index] = mouth_right_slider
    else:
        mouth_index = pose_parameters.get_parameter_index(mouth_name)
        pose[0, mouth_index] = mouth_left_slider

    pose[0, iris_small_left_index] = iris_small_left_slider
    pose[0, iris_small_right_index] = iris_small_right_slider
    pose[0, iris_rotation_x_index] = iris_rotation_x_slider
    pose[0, iris_rotation_y_index] = iris_rotation_y_slider
    pose[0, head_x_index] = head_x_slider
    pose[0, head_y_index] = head_y_slider
    pose[0, neck_z_index] = neck_z_slider
    pose[0, body_y_index] = body_y_slider
    pose[0, body_z_index] = body_z_slider
    pose[0, breathing_index] = breathing_slider

    return pose.to(device)


def update(pose):
    global last_pose
    global last_torch_input_image

    if torch_input_image is None:
        return

    needs_update = False
    if last_torch_input_image is None:
        needs_update = True
    else:
        if (torch_input_image - last_torch_input_image).abs().max().item() > 0:
            needs_update = True

    if (pose - last_pose).abs().max().item() > 0:
        needs_update = True

    if not needs_update:
        return

    output_image = poser.pose(torch_input_image, pose)[0]
    output_image = output_image.detach().cpu()
    img = cv.cvtColor(numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0)),
                      cv.COLOR_RGB2BGR)
    last_torch_input_image = torch_input_image
    last_pose = pose
    return img


# 计算两个点的距离
def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


# 计算两条线的距离
def cal_line_distance(p11, p12, p21, p22):
    return math.sqrt(math.pow((((p11[0] + p12[0]) / 2) - ((p21[0] + p22[0]) / 2)), 2) + math.pow(
        (((p11[1] + p12[1]) / 2) - ((p21[1] + p22[1]) / 2)), 2))


# 获取左眼的差值
def get_eye_left_slider(face):
    return cal_line_distance(face[47], face[46], face[43], face[44])


# 获取右眼的差值
def get_eye_right_slider(face):
    return cal_line_distance(face[41], face[40], face[37], face[38])


# 计算嘴巴的差值
def get_mouth_slider(face):
    return cal_distance(face[66], face[62])


# 获取头左右方向
def get_head_y_slider(face):
    return cal_distance(face[28], [1076, 329])


# 获取上下方向
def get_head_x_slider(face):
    return cal_distance(face[27], [1076, 329])


# 读取所有的脸部信息
def face_process():
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    videoWriter = cv.VideoWriter("res.avi", fourcc, 25, (512, 512))

    faces = []
    # 左右眼最大张开值
    max_eye_left_slider = 0
    max_eye_right_slider = 0
    max_mouth_slider = 0
    # 头左右偏的最小值和最大值
    max_head_y_slider = 0
    min_head_y_slider = 0
    # 头上下偏的最大值和最小值
    max_head_x_slider = 0
    min_head_x_slider = 0
    for name in os.listdir("all"):
        with open("all/{}".format(name)) as f:
            data = json.loads(f.read())
            if len(data["people"]) > 0:
                # 提取人脸信息
                point = data["people"][0]["face_keypoints_2d"]
                # 每隔三个取一个，取出所有点坐标
                face = [point[i:i + 3] for i in range(0, len(point), 3)]
                # 分别提取出最大的人脸坐标
                # 如果不足68个点就不管
                if len(face) >= 68:
                    # print(face[28], face[29])
                    faces.append(face)
                    if max_eye_left_slider < get_eye_left_slider(face):
                        max_eye_left_slider = get_eye_left_slider(face)
                    if max_eye_right_slider < get_eye_right_slider(face):
                        max_eye_right_slider = get_eye_right_slider(face)
                    # print(get_mouth_slider(face))
                    if max_mouth_slider < get_mouth_slider(face):
                        max_mouth_slider = get_mouth_slider(face)
                    if max_head_y_slider < get_head_y_slider(face):
                        max_head_y_slider = get_head_y_slider(face)
                    if min_head_y_slider > get_head_y_slider(face):
                        min_head_y_slider = get_head_y_slider(face)
                    if max_head_x_slider < get_head_x_slider(face):
                        max_head_x_slider = get_head_x_slider(face)
                    if min_head_x_slider > get_head_x_slider(face):
                        min_head_x_slider = get_head_x_slider(face)
                    continue
            faces.append([])

    # faces = faces[:500]
    for i in trange(len(faces)):
        face = faces[i]
        img = transparent2white(default_img)
        if len(face) >= 68:
            ## 计算左眼闭合程度，就是计算两条直线的中点距离 43-44 47-46
            eye_left_slider = 1 - get_eye_left_slider(face) / max_eye_left_slider
            eye_right_slider = 1 - get_eye_right_slider(face) / max_eye_right_slider
            mouth_slider = get_mouth_slider(face) / max_mouth_slider
            if get_head_y_slider(face) > 0:
                head_y_slider = get_head_y_slider(face) / max_head_y_slider
            else:
                head_y_slider = -abs(get_head_y_slider(face) / min_head_y_slider)
            if get_head_x_slider(face) > 0:
                head_x_slider = get_head_x_slider(face) / max_head_x_slider
            else:
                head_x_slider = -abs(get_head_x_slider(face) / min_head_x_slider)
            # print("左眼,{}".format(eye_left_slider))
            # print("右眼,{}".format(eye_right_slider))
            # print("嘴巴,{}".format(mouth_slider))
            # print("头左右,{}".format(head_y_slider))
            # print("头上下,{}".format(head_x_slider))
            img = update(get_pose(eye_left_slider, eye_right_slider, mouth_slider, head_x_slider, head_y_slider))
            img = transparent2white(img)
            # print(img.shape)
            # cv.imshow('img1', img)
            # cv.waitKey(-1)
            # img = np.zeros((1080, 1920, 3), np.uint8)
            # img.fill(255)
            # point_color = (0, 0, 255)  # BGR
            # for r in face:
            #     cv.circle(img, (int(r[0]), int(r[1])), 1, point_color, 0)
        videoWriter.write(img)
    videoWriter.release()

    # 图片显示

    # # 浅灰色背景
    # cv.imshow('img1', img)
    # cv.waitKey(-1)


if __name__ == '__main__':
    path = "./data/images/crypko_02.png"
    # 读取图片，先使用opencv去读取一个默认的图片
    with open(path, "rb") as f:
        default_img = cv.imread(path)
        # 把我们的图片转换为pytorch的格式
        pil_image = resize_PIL_image(extract_PIL_image_from_filelike(f), size=(512, 512))
        w, h = pil_image.size
        if pil_image.mode != 'RGBA':
            raise "Image must have an alpha channel!!!"
        else:
            torch_input_image = extract_pytorch_image_from_PIL_image(pil_image).to(device)
            if poser.get_dtype() == torch.half:
                torch_input_image = torch_input_image.half()
            # 开始处理人脸信息
            face_process()
