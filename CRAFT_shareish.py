
import os
import argparse
from test import copyStateDict
import string
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import cv2
import numpy as np
import test
import imgproc
import file_utils
import itertools

from utils import CTCLabelConverter, AttnLabelConverter
from model import Model
from craft import CRAFT
from dataset import AlignCollate, ShareishDataset
import torch.nn.functional as F
import cv2
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")



def Boxes():
    #CRAFT
    
    model_pretrained = "c:/Master's thesis/CRAFT-pytorch-master/weights/craft_mlt_25k.pth"
    text_threshold = 0.7
    low_text = 0.4
    link_threshold = 0.4
    cuda = True
    canvas_size = 1280
    mag_ratio = 1.5
    Image_folder = "c:/Master's thesis/CRAFT-pytorch-master/Image_shareish/"


    image_path, _, _ = file_utils.get_files(Image_folder)


    image_name = os.path.basename(image_path[0])

    model = CRAFT()    

    if cuda:
        model = model.cuda()
        model.load_state_dict(copyStateDict(torch.load(model_pretrained)))
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = False
    else:
        model.load_state_dict(copyStateDict(torch.load(model_pretrained, map_location='cpu')))

    model.eval()

    image = imgproc.loadImage(image_path[0])

    image = np.ascontiguousarray(image)
    def str2bool(v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    #CRAFT
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default="c:/Master's thesis/CRAFT-pytorch-master/weights/craft_mlt_25k.pth", type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default="/Master's thesis/CRAFT-pytorch-master/Images/", type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()
 
    bboxes, polys, score_text, det_scores = test.test_net(model, image, text_threshold, link_threshold, low_text, cuda, False, args)

    bbox_score = []

    for box_num in range(len(bboxes)):
          key = det_scores[box_num]
          bbox = bboxes[box_num]
          item = [key, bbox]

          bbox_score.append(item)

    image = cv2.imread(image_path[0])
    return image_name, image, bbox_score, polys, score_text, det_scores

def crop(pts, image):
  """
  Takes inputs as 8 points
  and Returns cropped, masked image with a white background
  """
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  cropped = image[y:y+h, x:x+w].copy()
  # show cropped image


  pts = pts - pts.min(axis=0)
  mask = np.zeros(cropped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
  dst = cv2.bitwise_and(cropped, cropped, mask=mask)
  bg = np.ones_like(cropped, np.uint8)*255
  cv2.bitwise_not(bg,bg, mask=mask)
  dst2 = bg + dst

  return dst2

def generate_words(image_name, score_bbox, image):
  words = []                  
  for bbox_coords in score_bbox:

    if bbox_coords[0] != 0:
      l_t = bbox_coords[1][0][0]
      t_l = bbox_coords[1][0][1]
      r_t = bbox_coords[1][1][0]
      t_r = bbox_coords[1][1][1]
      r_b = bbox_coords[1][2][0]
      b_r = bbox_coords[1][2][1]
      l_b = bbox_coords[1][3][0]
      b_l = bbox_coords[1][3][1]

      pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
      
      if np.all(pts) > 0:
        
        word = crop(pts, image)
      words.append(word)
  return words

def get_words():

    image_name, image, bbox_score, polys, score_text, det_scores = Boxes()

    # disct to dataframe

    image_name = image_name.strip('.jpg')

    words = generate_words(image_name, bbox_score, image)
    return words, image

def generate_text(image, words):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default= "*", help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default="c:/Master's thesis/None-VGG-BiLSTM-CTC.pth", help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,  default="None", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="VGG", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if opt.sensitive:
        opt.character = string.printable[:-6] 

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)

    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = ShareishDataset(words, image, opt=opt)  
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for image_tensors, _ in demo_loader:

            batch_size = image_tensors.size(0)

            image = image_tensors.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:

                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                #preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open(f'./result_1file.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for  pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{"blbl":25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{"blbl":25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()
def read_text():
    result2_path = "result_1file.txt"
    result2 = open(result2_path, "r")
    result_text2 = result2.readlines()
    result2.close()

    #if line starts with --- or image_path remove it
    result_text2 = [x for x in result_text2 if not x.startswith("---") and not x.startswith("image_path")]


    # split with tab
    result_text2 = [x.split('\t') for x in result_text2]
    
    


    
    #group line with same 77 first characters as first element of the list
    result_text2 = [list(g) for k, g in itertools.groupby(result_text2, lambda x: x[0][:77])]

    for i in range(len(result_text2)):
        for j in range(len(result_text2[i])):
            result_text2[i][j] = result_text2[i][j][1]

    join_result2 = result_text2
    return join_result2

if __name__ == '__main__':
    words, image = get_words()
    res = generate_text(image, words)
    result = read_text()
    #print(result)