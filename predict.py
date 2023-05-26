import argparse
import numpy as np
import torch
from scipy import ndimage
from congestion_model import CongestionModel
import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CongestionPrediction():
    def __init__(self,marco_path, RUDY_path, RUDY_pin_path,model_weight_path,device):
        super(CongestionPrediction, self).__init__()
        self.FeaturePathList = [marco_path, RUDY_path, RUDY_pin_path]
        self.feature = self.data_process(self.FeaturePathList).unsqueeze(0).to(device)
        self.model = CongestionModel(device).to(device)
        self.device = device
        checkpoint = torch.load(model_weight_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def resize(self,input):
        dimension = input.shape
        result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
        return result

    def std(self,input):
        if input.max() == 0:
            return input
        else:
            result = (input - input.min()) / (input.max() - input.min())
            return result

    def data_process(self,FeaturePathList):
        features = []
        for path in FeaturePathList:
            feature = np.load(path)
            feature = self.std(self.resize(feature))
            features.append(torch.as_tensor(feature))
        features = torch.stack(features).type(torch.float32)
        return features

    def find_congestion_coord(self,tensor, threshold):
        indices = torch.where(tensor > threshold)
        return np.array(list((indices[1].tolist(), indices[0].tolist()))).T

    def Prediction(self, congestion_threshold):
        self.congestion_threshold = congestion_threshold
        if self.device != 'cpu':
            with torch.cuda.amp.autocast():
                self.pred = self.model(self.feature)
                self.pred = self.model.sigmoid(self.pred)
        if self.device == 'cpu':
            self.pred = self.model(self.feature)
            self.pred = self.model.sigmoid(self.pred)
        self.pred_coord = self.find_congestion_coord(self.pred[0,0], threshold=congestion_threshold)
        return self.pred, self.pred_coord

    def ShowFig(self,fig_save_path):
        if fig_save_path is None:
            raise ValueError("Figure save path is not specified clear.")
        plt.imshow(self.pred[0, 0].detach().cpu().numpy())
        plt.title(f"Congestion > {self.congestion_threshold}")
        pts = plt.scatter(x=self.pred_coord[:,0],y=self.pred_coord[:,1],c='r',s=5)
        plt.legend([pts],["Congestion locate"])
        plt.savefig(f"{fig_save_path}/congestion_{self.congestion_threshold}.png")
        plt.show()

    def save(self,output_path):
        np.save(f"{output_path}/PredArray",self.pred[0,0].detach().cpu().numpy())
        np.save(f"{output_path}/PredCoord",self.pred_coord)


def parse_args():
    description = "Input the Path for Prediction"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--marco_path", default=None, type=str, help='The path of the marco region')
    parser.add_argument("--RUDY_path", default=None, type=str, help='The path of the RUDY')
    parser.add_argument("--RUDY_pin_path", default=None, type=str, help='The path of the RUDY pin')
    parser.add_argument("--fig_save_path", default="./save_img", type=str, help='The path you want to save fingue')
    parser.add_argument("--weight_path", default=None, type=str, help='The path of the model weight')
    parser.add_argument("--output_path", default="./output", type=str, help='The path of the model weight')
    parser.add_argument("--congestion_threshold", default=0.5, type=int, help='congestion_threshold [0,1]')
    parser.add_argument("--device", default='cpu', type=str, help='If you have gpu type "cuda" will be faster!!')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # marco_path = r"C:\Users\user\Desktop\EDA\congestion\data\macro_region\10370-zero-riscy-b-3-c20-u0.9-m3-p7-f1"
    # RUDY_path = r"C:\Users\user\Desktop\EDA\congestion\data\RUDY\10370-zero-riscy-b-3-c20-u0.9-m3-p7-f1"
    # RUDY_pin_path = r"C:\Users\user\Desktop\EDA\congestion\data\RUDY_pin\10370-zero-riscy-b-3-c20-u0.9-m3-p7-f1"
    # fig_save_path = r"C:\Users\user\Desktop\EDA\congestion\save_img"
    # weight_path = r"C:\Users\user\Desktop\IC_image\congestion2_weights.pt"
    args = parse_args()
    predictionSystem = CongestionPrediction(marco_path=args.marco_path,RUDY_path=args.RUDY_path,RUDY_pin_path=args.RUDY_pin_path,
                                model_weight_path=args.weight_path,device=args.device)
    pred,pred_coord = predictionSystem.Prediction(congestion_threshold=args.congestion_threshold)
    predictionSystem.save(args.output_path)
    if args.fig_save_path !=None:
        predictionSystem.ShowFig(args.fig_save_path)
