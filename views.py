from __future__ import print_function, division
from django.shortcuts import render_to_response
from django.http.response import StreamingHttpResponse
import os
import numpy as np
import imageio
import io
from PIL import Image,ImageDraw
from edgetpu.classification.engine import ClassificationEngine
from django.shortcuts import render
from django.http import JsonResponse
from glob import glob
import time
a=[]
b=[]
c=[]

input_stats=(1.0,0)
class VideoCamera():
  def __init__(self):
    self.video= imageio.get_reader('<video1>',fps=30)
    #self.video =imageio.get_reader(os.path.dirname(__file__)[:-6] + "static/video/demo4_1s_low_Trim.mp4",'ffmpeg')
    #self.video_check=0
    #self.video_len=len(list(enumerate(self.video)))
    self.engine = ClassificationEngine('LYR_quantized_model_hascamera_edgetpu.tflite')
    self.count=0


  def get_frame(self):
    #img_name=str(self.count).zfill(6)
    #image = Image.open(os.path.dirname(__file__)[:-6] + "static/video/test/"+"img"+img_name+".jpg")
    #self.count+=1
    #pre_image = image.resize((720,720))
    #if(self.count==54):
      #self.count=0
    #image = self.video.get_next_data()
    #self.video_check+=1
    #if(self.video_check==self.video_len):
      #self.video =imageio.get_reader(os.path.dirname(__file__)[:-6] + "static/video/demo4_1s_low_Trim.mp4")
      #self.video_check=0
    #image = Image.fromarray(image)
    #pre_image = image.crop((280, 0, 1000, 720)) #generl_video
    #pre_image = image.crop((280, 25, 1000, 745)) #Trim_video
    image = self.video.get_next_data()
    image = np.rot90(image, 2)
    image = Image.fromarray(image)
    pre_image = image.crop((135,162,405,430)) #webcam
    #pre_image.save(str(self.count)+".jpg")
    #self.count+=1
    #draw = ImageDraw.Draw(pre_image)
    #draw.polygon([(100,10),(100,468),(555,468),(555,10)], outline=(255,0,0))
    #draw = ImageDraw.Draw(pre_image)
    #draw.polygon([(135,160),(135,425),(400,425),(400,160)], outline=(255,0,0))
    img_A = pre_image.resize((100, 100))
    #img_A = image.resize((100, 100))
    img_A = img_A.convert("L")
    img_A = np.reshape(img_A,(100*100*1))
    #time.sleep(0.4)
    a.append(self.run_inference(img_A, self.engine,input_stats))
    img_byte=io.BytesIO()
    pre_image.save(img_byte,format='PNG')
    return img_byte.getvalue()
    
  def run_inference(self,data, engine, input_stats):
    input_scale, input_zero_point = input_stats
    quantized_data = np.uint8(data/input_scale+input_zero_point)
    predicted_class = engine.ClassifyWithInputTensor(quantized_data, top_k=1)
    return int(predicted_class[0][0])



class VideoCamera1():
  def __init__(self):
    #data_type = os.path.dirname(__file__)[:-6]+"static/video"
    #self.path = glob('%s/%s/*' % (data_type,"test"))
    self.video1 =imageio.get_reader(os.path.dirname(__file__)[:-6] + "static/video/demo1_1s_low_Trim.mp4",'ffmpeg')
    self.video1_check=0
    self.video1_len=len(list(enumerate(self.video1)))
    self.engine1 = ClassificationEngine('LYR_quantized_model_edgetpu.tflite')
    #self.count1=69

  def get_frame(self):
    #img_name1=str(self.count1).zfill(6)
    #image1 = Image.open(os.path.dirname(__file__)[:-6] + "static/video/test/"+"img"+img_name1+".jpg")
    #self.count1+=1
    #pre_image1 = image1.resize((720,720))
    #if(self.count1==143):
      #self.count1=69
    #indices1 = (len(self.path) * np.random.rand(1)).astype(int)
    #image=Image.open(self.path[indices1[0]])
    image1 = self.video1.get_next_data()
    self.video1_check+=1
    if(self.video1_check==self.video1_len):
      self.video1 =imageio.get_reader(os.path.dirname(__file__)[:-6] + "static/video/demo1_1s_low_Trim.mp4")
      self.video1_check=0
    image1 = Image.fromarray(image1)
    #pre_image1 = image1.crop((280, 0, 1000, 720)) #generl_video
    pre_image1 = image1.crop((280, 25, 1000, 745)) #Trim_video
    img_A1 = pre_image1.resize((100, 100))
    #img_A1 = image1.resize((100, 100))
    img_A1 = img_A1.convert("L")
    img_A1 = np.reshape(img_A1,(100*100*1))
    #time.sleep(0.4)
    b.append(self.run_inference1(img_A1, self.engine1, input_stats))
    img_byte1=io.BytesIO()
    pre_image1.save(img_byte1,format='PNG')
    return img_byte1.getvalue()
    
  def run_inference1(self,data1, engine1, input_stats):
    input_scale, input_zero_point = input_stats
    quantized_data1 = np.uint8(data1/input_scale+input_zero_point)
    predicted_class1 = engine1.ClassifyWithInputTensor(quantized_data1, top_k=1)
    return int(predicted_class1[0][0])
    
class VideoCamera2():
  def __init__(self):
    self.video2 =imageio.get_reader(os.path.dirname(__file__)[:-6] + "static/video/demo2_1s_low_Trim.mp4",'ffmpeg')
    self.video2_check=0
    self.video2_len=len(list(enumerate(self.video2)))
    self.engine2 = ClassificationEngine('LYR_quantized_model_edgetpu.tflite')
    #self.count2=227

  def get_frame(self):
    #img_name2=str(self.count2).zfill(6)
    #image2 = Image.open(os.path.dirname(__file__)[:-6] + "static/video/test/"+"img"+img_name2+".jpg")
    #self.count2+=1
    #pre_image2 = image2.resize((720,720))
    #if(self.count2==339):
      #self.count2=227

    image2 = self.video2.get_next_data()
    self.video2_check+=1

    if(self.video2_check==self.video2_len):
      self.video2 =imageio.get_reader(os.path.dirname(__file__)[:-6] + "static/video/demo2_1s_low_Trim.mp4")
      self.video2_check=0

    image2 = Image.fromarray(image2)
    #pre_image2 = image2.crop((280, 0, 1000, 720)) #generl_video
    pre_image2 = image2.crop((280, 25, 1000, 745)) #Trim_video
    img_A2 = pre_image2.resize((100, 100))
    #img_A2 = image2.resize((100, 100))
    img_A2 = img_A2.convert("L")
    img_A2 = np.reshape(img_A2,(100*100*1))
    #time.sleep(0.1)
    c.append(self.run_inference2(img_A2, self.engine2, input_stats))
    img_byte2=io.BytesIO()
    pre_image2.save(img_byte2,format='PNG')
    return img_byte2.getvalue()
    
  def run_inference2(self,data2, engine2, input_stats):
    input_scale, input_zero_point = input_stats
    quantized_data2 = np.uint8(data2/input_scale+input_zero_point)
    predicted_class2 = engine2.ClassifyWithInputTensor(quantized_data2, top_k=1)
    return int(predicted_class2[0][0])
    
def gen(camera):
  while True:
    frame = camera.get_frame()
    yield (b'--frame\n'b'Content-Type: image/jpeg\n\n' +frame+ b'\n\n')
    
def gen1(camera1):
  while True:
    frame1 = camera1.get_frame()
    yield (b'--frame1\n'b'Content-Type: image/jpeg\n\n' +frame1+ b'\n\n')
    
def gen2(camera2):
  while True:
    frame2 = camera2.get_frame()
    yield (b'--frame2\n'b'Content-Type: image/jpeg\n\n' +frame2+ b'\n\n')

def data_fresh(request):
    context = {"data1": a,
               "data2": b,
               "data3": c, }
    #print("1"+str(a))
    #print("2"+str(b))
    #print("3"+str(c))
    return JsonResponse(context)


def video_feed(request):
  return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')


def video_feed1(request):
  return StreamingHttpResponse(gen1(VideoCamera1()), content_type='multipart/x-mixed-replace; boundary=frame1')


def video_feed2(request):
  return StreamingHttpResponse(gen2(VideoCamera2()), content_type='multipart/x-mixed-replace; boundary=frame2')



def index1(request):
    path1 = 'http://'+request.get_host()+ '/video_feed'
    path2 = 'http://' + request.get_host() + '/video_feed1'
    path3 = 'http://' + request.get_host() + '/video_feed2'
    return render_to_response('index.html', locals())


