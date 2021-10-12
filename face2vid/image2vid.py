import cv2
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import os

from os.path import join, exists, abspath, dirname
import ffmpeg

audio_path ='../examples/audio/obama2.wav'

video_new='../examples/test_image/test_1.avi'
output = '../examples/test_image/test_1_audio.avi'

finishcode1 ='ffmpeg -i "'+video_new+ '" -i "'+audio_path+'" -c copy '+output
if not os.path.exists(output):
    os.system(finishcode1)