from scenedetect import open_video
from scenedetect.detectors import ContentDetector
from scenedetect import SceneManager
from scenedetect.video_splitter import split_video_ffmpeg
import cv2
import os

video_path = './data/video/poppyes.mp4'
# video_path = './data/video/vllo.mp4'
video = open_video(video_path)

content_detector = ContentDetector(threshold=30, min_scene_len=10)
scene_manager = SceneManager()
scene_manager.add_detector(content_detector)

scene_manager.detect_scenes(video, show_progress=True)

scene_list = scene_manager.get_scene_list()
for scene in scene_list:
    start, end = scene
    print(start, "~", end)

save_path = './data/video/crop_video/'
# select_list = scene_list[:5]
# select_list = scene_list.copy()
split_video_ffmpeg(video_path, scene_list, show_progress=True)

# 영상을 프레임단위로 자르기
def split_video(input_video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the frames per second (fps) and frame size
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Create an output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read frames and save them as individual images
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        output_path = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
        cv2.imwrite(output_path, frame)

        frame_number += 1

    # Release the video capture object
    cap.release()


video_path = "./data/video/crop_video/poppyes-Scene-004.mp4"
folder_name = video_path.split('/')[-1]
folder_name = folder_name.split('.')[0]
output_folder = f"./data/image/{folder_name}/"  # Replace with the desired output folder
split_video(video_path, output_folder)
