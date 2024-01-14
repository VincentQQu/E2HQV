import numpy as np
import torch
from models import count_parameters, TimeEmbedMini

from PIL import Image

from new_models import EffMultiWNet
from conv_lstm import UNetConvLSTM

import os




device = torch.device("cuda:0")
print(device)

model_dir = "./versions/"
ver_no = "1.0.0"
datafolder = "./dataset/desk_fast_voxelgrid_5bins_examples/"
pred_save_dir = "./predictions"


is_recurrent = True #new

init_frame = 0.0
max_gt_interval = 20
gt_interval = max_gt_interval

embed_h, embed_w = 180, 240

lstm_n_lys = 3
max_lstm_len = 40

out_depth = 1
bilinear = True

sep = '_'

n_bins = 5



models = {

}



def create_dir(folder_p):
    if not os.path.exists(folder_p):
        os.makedirs(folder_p)
    return folder_p


def construct_predth_model():

    inc_f0, use_embedr, add_lstm = 1, 1, 1
    input_n_channels = n_bins + inc_f0 + use_embedr + add_lstm
    ch1 = 12

    model_name = f"EffMultiWNet{sep}v{ver_no}" #small_

    print(model_name)

    fresh_model = EffMultiWNet(n_channels=input_n_channels, out_depth=out_depth, inc_f0=inc_f0, inc_e2=0, inc_f2=0, bilinear=bilinear, n_lyr=4, ch1=ch1, c_is_const=False, c_is_scalar=False, device=device)

    num_params = count_parameters(fresh_model)
    print(f"num of parameters: {num_params:,}")

    return {"model": fresh_model, "name": model_name}



def construct_embedr():
  
    h, w = embed_h, embed_w
    embed_rf0 = 1

    model_name = f"EffEmbedr{sep}v{ver_no}" #small_

    print(model_name)

    fresh_model = TimeEmbedMini(h=h, w=w, seq_len=max_gt_interval, embed_rf0=embed_rf0)

    num_params = count_parameters(fresh_model)
    print(f"num of parameters: {num_params:,}")

    return {"model": fresh_model, "name": model_name}



def construct_lstmcn():
  
    model_name = f"EffLstmcn{sep}v{ver_no}" #small_
    print(model_name)


    fresh_model = UNetConvLSTM(input_size=n_bins, output_size=1, n_lyr=lstm_n_lys, decode_lstm=1)

    num_params = count_parameters(fresh_model)

    return {"model": fresh_model, "name": model_name}




def load_data_path(datafolder):
    
    return sorted( [os.path.join(datafolder, p) for p in os.listdir(datafolder)] )




def reconstruct_core(models, e0, previous_f, last_gt, distance_from_gt, p_states):
    b_E0_cuda = torch.from_numpy(np.load(e0)).unsqueeze(dim=0).to(device=device, dtype=torch.float32)


    if previous_f == None:
        frame_size = [1, 1] + list(b_E0_cuda.shape[2:])
        previous_f = init_frame * torch.ones(frame_size).to(device=device, dtype=torch.float32)

        last_gt = previous_f
    


    rf0 = previous_f
    lstm_out, current_states = models["lstmcn"]["model"](b_E0_cuda, p_states)


    last_gt = lstm_out
    embed_channel, post_embed_x, embed_to_add = models["embedr"]["model"](rf0, last_gt, distance_from_gt, 0, device=device)

    rf0_e0 = [post_embed_x, embed_channel, b_E0_cuda, lstm_out]

    rf0_e0 = torch.concat(rf0_e0, axis=1)

    all_output = models["predth"]["model"](rf0_e0)

    f01 = all_output[:,0,:,:].unsqueeze(dim=1)

    f01 = torch.clamp(f01, min=0, max=1)

    return rf0, f01.detach(), last_gt, p_states, all_output



def reconstruct_iter(models, data_paths):

    rfn1s, cf0s = [], []

    cf0 = None
    last_gt = None
    p_states = [None] * (lstm_n_lys*2)
      
    for b_i, e0 in enumerate(data_paths, 1):
        distance_from_gt = (b_i-1) % gt_interval
        rfn1, cf0, last_gt, p_states, all_output = reconstruct_core(models, e0, previous_f=cf0, last_gt=last_gt, distance_from_gt=distance_from_gt, p_states=p_states)


        rfn1s.append(rfn1)
        cf0s.append(cf0)


        if gt_interval and (b_i % gt_interval == 0):
            cf0 = None
        if b_i % max_lstm_len == 0:
            p_states = [None] * (lstm_n_lys*2)
    

    return rfn1s, cf0s



def np2img(pred_f1, pred_f1_p):
  img = Image.fromarray(np.uint8(pred_f1))

  img.save(pred_f1_p)

        
        
def export_f1_recurrent(data_paths, preds, pred_save_dir):
    
    subfolder_name = data_paths[0].split(os.path.sep)[-2]

    save_dir = os.path.join(*[pred_save_dir, subfolder_name])
    save_dir = create_dir(save_dir)


    preds = np.squeeze(preds)

    if len(preds.shape) < 3:
      preds = np.expand_dims(preds, axis=0)
    
      
    for data_path, f1_hat in zip(data_paths, preds):
      file_name = os.path.split(data_path)[-1]

      pred_f1 = f1_hat*255

      save_path = os.path.join(save_dir, f"{file_name}.png")

      np2img(pred_f1, save_path)


    



def main():
    
    create_dir(pred_save_dir)

    models["predth"] = construct_predth_model()
    models["embedr"] = construct_embedr()
    models["lstmcn"] = construct_lstmcn()


    for mk in models:
        model_name = models[mk]["name"]
        mp = f"{model_dir}v{ver_no}/{model_name}.pth"
        models[mk]['model'].load_state_dict(torch.load(mp, map_location=device))
        models[mk]['model'].to(device=device)
  

    data_paths = load_data_path(datafolder)

    rfn1s, cf0s = reconstruct_iter(models, data_paths)

    # preds = [rfn1s[0]] + cf0s

    preds = cf0s
    preds = torch.concat(preds).to(device="cpu", dtype=torch.float32)


    export_f1_recurrent(data_paths[1:], preds, pred_save_dir)


        


if __name__ == "__main__":
    main()