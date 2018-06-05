import sys,os
import scipy.io as sio
import json, itertools
import numpy as np
#from vgg16 import VGG16
#from keras.preprocessing import image
#from imagenet_utils import preprocess_input
import numpy as np
#from keras.models import Model


images_path= '/users/ud2017/hoavt/data/flickr30k-images/'
def_fnames='image_features/filenames_77512_2.json'
def_feats='image_features/vgg_feats_77512_2.npy'

#def_fnames='/users/ud2017/hoavt/data/flickr8k/filenames_77512.json'
#def_feats='/users/ud2017/hoavt/data/flickr8k/vgg_feats_77512.npy'
#images_path= '/users/ud2017/hoavt/data/flickr8k/Flicker8k_Dataset/'

class ImageFeatures(object):
    def __init__(self, names_files=def_fnames, feats_files=def_feats):
        self.cache = {}
        self.name2idx={}
        self.names_files = names_files
        #the image features file file do exist
        if os.path.exists(self.names_files):
            with open(self.names_files,'rb') as fn:
                self.names=json.load(fn)
            self.feats = np.load(feats_files)
            for img_file in self.names:
                self.name2idx[img_file]= len(self.name2idx)
        else:
            self.names = []
            self.name2idx = {}

    def get_feat(self,img_file):
        if hasattr(self,'name2idx') and img_file in self.name2idx:
            return self.feats[self.name2idx[img_file]]
        return self.get_feat_model(img_file)
    
    def save_feat(self):
        np.save(def_feats, self.feats)
        with open(self.names_files,'wb') as fn:
            json.dump(self.names, fn)
    def get_feat_model(self, img_file):
        #put these imports here since it conflicts with tensorflow
        from keras.preprocessing import image
        from imagenet_utils import preprocess_input 
        if not hasattr(self,'model'):
            from vgg16 import VGG16
            from keras.models import Model
            base_model = VGG16(weights='imagenet')
            self.model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
        img = image.load_img(images_path + img_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc7_features = self.model.predict(x)
        if not hasattr(self, 'names'):
            self.names = []
            self.name2idx = {}
        self.names.append(img_file)
        self.name2idx[img_file] = len(self.name2idx)
        if hasattr(self,'feats'):
            self.feats = np.vstack((self.feats, fc7_features))
        else: self.feats = fc7_features
        return fc7_features[0]

if __name__ == '__main__':
    extractor = ImageFeatures()
    fnames = os.listdir(images_path)
    count=0
    for fname in fnames:
        print fname, count
        count +=1
        #if count <= 20000: continue
        try:
            feats = extractor.get_feat(fname)
        except:
            pass
        
    extractor.save_feat()
