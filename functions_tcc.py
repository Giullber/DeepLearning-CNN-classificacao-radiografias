# Bibliotecas
import os
import cv2 
import albumentations as A
import pickle
import tensorflow as tf

from sklearn.utils import resample
from keras.models import save_model
from tensorflow.keras.models import Model

#######################################

## Definição do caminho da pasta original da aplicação
caminho_origin = os.getcwd()


## Criação da classe com funções espcíficas para uso nas análises do TCC
class func_tcc():
    def __init__(self):
        pass
        
    # Método para realizar uma amostragem em um diretório    
    def amostragem(self, path_or, n):
        list_path = [os.path.join(path_or, i) for i in os.listdir(path_or)] #uni o diretorio da pasta com os nomes das imagens
        amostra = n # quantidade da amostra
        return (resample(list_path,n_samples=amostra, replace=False))  # saída com a relalização da amostragem por bootstrapping

    # Método para a Leitura de uma Imagem
    def leitura(self, img_path, classe):
        self.classe = classe  # variável classe: nome utilizado para represnetar a classe da imagem
        self.path_fold = img_path   # definição do caminho da pasta onde será atribuído a variável path_fold do método "caminho_pasta"
        self.img = cv2.imread(img_path)  # realiza a leitura da imagem pelo pacote cv2
        return(self.img)

    # Método para a aplicação do data argumentation
    def transforma(self):
        crop = [400,420,440,460,480]  # lista com diferentes dimensões para a realizaçõa dos cortes
        crop = resample(crop, n_samples=1)[0]  
        trans = A.Compose([
        A.Resize(512,512),
        A.RandomCrop(width=crop, height=crop,p=1),
        A.RandomBrightnessContrast(p=1)
        ]) 
        return(trans(image = self.img)['image'])
    
    # Método para definir o caminho aonde a imagem será salva
    def caminho_pasta(self):
        path = self.path_fold.split('\\')[:-1]
        self.ext = self.path_fold.split('.')[-1]
        self.path= '\\'.join(path)
        self.quant = len(os.listdir(self.path))
        return(self.path)

    # Método para nomear a imagem criada    
    def nome_img(self):
        self.caminho_pasta()
        # quant = len(os.listdir(self.path))
        self.nome = self.classe + "_" + str(self.quant + 1) + "." + self.ext
        return(self.nome)
    
    # Método para salvar apenas uma imagem
    def criador_img(self):
        nome = self.nome_img()
        os.chdir(self.path)
        cv2.imwrite(img = self.transforma(), filename=nome)
        self.path_origin()

    # Método para redefinir o caminho origem    
    def path_origin(self):
        os.chdir(caminho_origin)

    # Método para criar múltiplas imagens retornando a lista de imagens amostradas
    def criador_mult(self,path_pasta,n,rep,classe):
        lista = self.amostragem(path_pasta,n)
        for i in range(len(lista)):
            self.leitura(lista[i],classe)   
            for r in range(rep):
                self.criador_img()
        return(lista)
        
    # Método para carregar o histórico de treinamento de um modelo
    def load_model(self, path_load, type, weight_ph=None):
        self.path_fold = path_load
        self.caminho_pasta()
        load_name = path_load.split('\\')[-1]
        os.chdir(self.path)

        if type == 'historic':
            with open(load_name, 'rb') as f:
                loaded_hist = pickle.load(f)
            self.path_origin()
            return(loaded_hist)
        
        elif type == 'model':
            load_weight = weight_ph.split('\\')[-1]
            with open (load_name, 'r') as json_file:
                json_saved_model = json_file.read()
            loaded_model = tf.keras.models.model_from_json(json_saved_model)
            loaded_model.load_weights(load_weight)
            loaded_model.compile(loss = 'categorical_crossentropy',optimizer ='Adam',metrics=['accuracy'])
            self.path_origin()
            return(loaded_model)
    
    # Método para salvar o histórico, modelo e pesos
    def save_model(self,model ,path_save, type, nome):
        os.chdir(path_save)

        if type=='historic':
            name = nome + '.pickle'
            with open(name, 'wb') as f:
                pickle.dump(model.history, f)
            self.path_origin()

        elif type=='model':
            model_json_inc = model.to_json()
            name = nome + '.json'
            with open(name,'w') as json_file:
                json_file.write(model_json_inc)
            self.path_origin()

        elif type=='weight':
            name = nome + '.hdf5'
            save_model(model, name)
            self.path_origin()
        





   
        