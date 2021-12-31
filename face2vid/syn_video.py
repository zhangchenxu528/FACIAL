import os
import time
import subprocess

audio_path = '../examples/audio/xml.wav'
video_new = '../examples/test_image/xml/test_1.avi'
output = '../examples/test_image/xml/test_1_audio.avi'
output_mp4 = '../examples/test_image/xml/test_1_audio.mp4'
# os.system(ffmpeg -i '$video_new' -i '$audio_path' -c copy '$output')
# os.system(ffmpeg -i '$output'  '$output_mp4')

strsmd = 'ffmpeg -i ' + video_new + ' -i ' + audio_path + ' -c copy ' + output + ''

ffmpeger = subprocess.call(strsmd, shell=True)

time.sleep(2)
# ffmpeger.stdin.write('q'.encode("GBK"))
# ffmpeger.communicate()

strcmd = 'ffmpeg -i ' + output + ' ' + output_mp4 + ''

ffmpeger = subprocess.call(strcmd, shell=True)

time.sleep(2)
# ffmpeger.stdin.write('q'.encode("GBK"))
# ffmpeger.communicate()
