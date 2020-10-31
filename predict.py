import argparse
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import json

pars = argparse.ArgumentParser(description='predict image names -->clasification')
pars.add_argument('--image',default = './test_images/orange_dahlia.jpg',action = "store", help='image_path')
pars.add_argument('--top_k',default = 5 , dest ="top_k", action = "store",type = int, help='top k probability names')
pars.add_argument('--model' , default ='./best_model.h5' ,action = "store" ,type = str, help = 'file name of the model')         
pars.add_argument('--category_names',default = 'label_map.json',dest="category_names", action="store", help = 'categories name ')       

store_pars = pars.parse_args()
image_path = store_pars.image
top_k = store_pars.top_k
model = store_pars.model
category_names = store_pars.category_names

def category_name(classes):
    with open('label_map.json', 'r') as f:
        category_names = json.load(f)
    classes = [category_names[str(i+1)] for i in classes]
    return classes

def call_model(model):
    loaded_keras_model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded_keras_model

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,[224,224]).numpy()
    image /= 255
    return image

def predict (image_path, model, top_k):
    
    im = Image.open(image_path)
    test_image = np.asarray(im)
    
    processed_test_image = process_image(test_image)
    expand_image = np.expand_dims(processed_test_image,axis=0)
    prob = model.predict(expand_image)
    top_prob, top_class = tf.math.top_k(prob,top_k)
    return top_prob.numpy()[0] ,top_class.numpy()[0]


def ploting(image_path, model, top_k):
    
    probs,classes  = predict(image_path, model, top_k)
    #print(classes)
    #classes= [category_name[str(i+1)] for i in classes]
    classes = category_name(classes)
    print ("name of the flowers "+ str(classes) +"\n probability "+str(probs) )       
    

if __name__ =="__main__":
    print(" running ...")
    model = call_model(model)
    ploting(image_path ,model , top_k)
    print("end ...")              
    