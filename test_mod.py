# -*- coding: utf-8 -*-

from time import time
import numpy as np
import cv2
import tensorflow as tf


## parameters ##
video=0
num_top_predictions=5
graph='./output_graph.pb'
labels='./output_labels.txt'
output_layer='final_result:0'
input_layer='input:0'
frame_threshold=5
accuracy_threshold=0.3

##################

x1,y1,x2,y2=390,0,640,250
      
consonant = {'aa':'ㅇ','ba':'ㅂ','cha':'ㅊ','da':'ㄷ','ga':'ㄱ','ha':'ㅎ',
             'ja':'ㅈ','ka':'ㅋ','ma':'ㅁ','na':'ㄴ','pa':'ㅍ','ra':'ㄹ',
             'sa':'ㅅ','ta':'ㅌ' }

prev_consonant=''
count =0 
text='ga'
bgrm=cv2.createBackgroundSubtractorMOG2()
 
def put_char_on_image(img,prediction,consonant):
    #put char image on left upper cam image
    global prev_consonant
    global count
    global text
    
    if consonant == prev_consonant:
        count +=1
    else:
        prev_consonant=consonant
        count =0
        
    if count > frame_threshold and prediction > accuracy_threshold:
        text = consonant
        
    char_img=cv2.imread('./char/'+text+'.png')
    char_img=cv2.resize(char_img,(60,60))
    img[0:60,0:60]=char_img
    
    return img

def load_labels(filename):
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def FPS(prev_time,num_of_img):
    #show the frame per second
    cur_time = time()
    total_time=cur_time-prev_time
    fps=num_of_img/total_time
    return fps

def load_graph(filename):
    #load graph from ,pb file
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def image_preprocessing(image_data_org):
    #remove background and normalization
    image_data=bgrm.apply(image_data_org)      
    crop_img = image_data[y1:y2,x1:x2]
    img_zero=np.zeros((y2-y1,x2-x1,3))
    img_zero[:,:,0]=crop_img
    img_zero[:,:,1]=crop_img
    img_zero[:,:,2]=crop_img
    crop_img=img_zero
    image_data_org[y1:y2,x1:x2]=img_zero

    crop_img=cv2.resize(crop_img,(224,224))      
    crop_img=np.expand_dims(crop_img,0)
    crop_img=(crop_img -127.5)/127.5
    
    return image_data_org,crop_img
    
    
def run_graph( labels, input_layer_name, output_layer_name,
              num_top_predictions):
    #testing
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    ground_truth= cv2.imread('./hand.jpg')
    ground_truth=cv2.resize(ground_truth,(640,480))
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    cap = cv2.VideoCapture(video)

    while True:
        prev_time=time()
        read,image_data_org=cap.read()
      
        if read == False:
            print("Couldn't find Cam or Video in path")
            break
            #return -1
        image_data_org=cv2.flip(image_data_org,1)
        image_data_org=cv2.resize(image_data_org,(640,480))       
        image_data_org,crop_img =image_preprocessing(image_data_org)
        
        predictions, = sess.run(softmax_tensor, {input_layer_name: crop_img})
      
    # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        top_prediction=predictions[top_k[0]]
        top_label=labels[top_k[0]]
        
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (consonant[human_string], score))
              
        image_data_org=put_char_on_image(image_data_org,top_prediction,top_label) 
        
        final=np.hstack([ground_truth,image_data_org])
        cv2.imshow('frame',final)
        # press q on keyborad to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
        fps=FPS(prev_time,num_of_img=1)
        print("Frame per second : ",int(fps))
      
  cap.release()
  cv2.destroyAllWindows()
  return 0

def main():
  
  #check necessery files
    if not tf.gfile.Exists(labels):
        tf.logging.fatal('labels file does not exist %s',labels)

    if not tf.gfile.Exists(graph):
        tf.logging.fatal('graph file does not exist %s',graph)

    if not tf.gfile.Exists('./hand.jpg'):
        tf.logging.fatal('Sign language picture does not exist '+'./hand.jpg')

    if not tf.gfile.Exists('./char'):
        tf.logging.fatal('consonant file does not exist %s'+'./char')
         
    label = load_labels(labels)    
  # load graph, which is stored in the default session
    load_graph(graph)
    
    run_graph(label,input_layer,output_layer,num_top_predictions)
     

if __name__ == '__main__':
  main()