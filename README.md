# E2HQV
Official Implementation for "E2HQV: High-Quality Video Generation from Event Camera via Theory-Inspired Model-Aided Deep Learning" - **AAAI 2024** [arxiv](https://arxiv.org/abs/2401.08117) [aaai](https://ojs.aaai.org/index.php/AAAI/article/download/28263/28517)

## E2HQV Generated Video Frames for Benchmarking
To benchmark with our method without processing your own data, you can find E2HQV-generated frames for evaluation on [Google Drive](https://drive.google.com/file/d/1pZRhDOfx5A7w-KZpPsOc3bq4okR9-58X/view?usp=sharing). Below are the model's statistics on each dataset and scene:

### [Overall]
| Method       | IJRR MSE↓ | IJRR SSIM↑ | IJRR LPIPS↓ | MVSEC MSE↓ | MVSEC SSIM↑ | MVSEC LPIPS↓ | HQF MSE↓ | HQF SSIM↑ | HQF LPIPS↓ |
|--------------|-----------|------------|-------------|------------|-------------|--------------|----------|-----------|------------|
| E2VID        | 0.212     | 0.424      | 0.350       | 0.337      | 0.206       | 0.705        | 0.127    | 0.540     | 0.382      |
| FireNet      | 0.131     | 0.502      | 0.320       | 0.292      | 0.261       | 0.700        | 0.094    | 0.533     | 0.441      |
| E2VID+       | 0.070     | 0.560      | 0.236       | 0.132      | 0.345       | 0.514        | 0.036    | 0.643     | 0.252      |
| FireNet+     | 0.063     | 0.555      | 0.290       | 0.218      | 0.297       | 0.570        | 0.040    | 0.614     | 0.314      |
| SPADE-E2VID  | 0.091     | 0.517      | 0.337       | 0.138      | 0.342       | 0.589        | 0.077    | 0.521     | 0.502      |
| SSL-E2VID    | 0.046     | 0.364      | 0.425       | 0.062      | 0.345       | 0.593        | 0.126    | 0.295     | 0.498      |
| ET-Net       | 0.047     | 0.617      | 0.224       | 0.107      | 0.380       | 0.489        | 0.032    | 0.658     | 0.260      |
| E2HQV (Ours) | 0.028     | 0.682      | 0.196       | 0.032      | 0.421       | 0.460        | 0.019    | 0.671     | 0.261      |

### [IJRR]
|        | boxes_6dof | calibration | dynamic_6dof | office_zigzag | poster_6dof | shapes_6dof | slider_depth |
|--------|------------|-------------|--------------|---------------|-------------|-------------|--------------|
| MSE↓   | 0.0354     | 0.0206      | 0.0278       | 0.0214        | 0.0345      | 0.0407      | 0.0129       |
| SSIM↑  | 0.5638     | 0.6471      | 0.7185       | 0.6802        | 0.5552      | 0.8194      | 0.7879       |
| LPIPS↓ | 0.2574     | 0.1639      | 0.1965       | 0.2239        | 0.1978      | 0.1712      | 0.1623       |

### [MVSEC]
|        | indoor_flying1 | indoor_flying2 | indoor_flying3 | outdoor_day1 | outdoor_day2 |
|--------|----------------|----------------|----------------|--------------|--------------|
| MSE↓   | 0.0235         | 0.0194         | 0.0224         | 0.0518       | 0.0403       |
| SSIM↑  | 0.4495         | 0.4249         | 0.4484         | 0.3343       | 0.4462       |
| LPIPS↓ | 0.4381         | 0.4444         | 0.4262         | 0.5802       | 0.4086       |

### [HQF]
|        | bike_bay_hdr | boxes | desk | desk_fast | desk_hand_only | desk_slow | engineering_posters | high_texture_plants | poster_pillar_1 | poster_pillar_2 | reflective_materials | slow_and_fast_desk | slow_hand | still_life |
|--------|--------------|-------|------|-----------|----------------|-----------|---------------------|---------------------|-----------------|-----------------|----------------------|--------------------|-----------|------------|
| MSE↓   | 0.0306       | 0.0139| 0.0146| 0.0087   | 0.0135         | 0.0223    | 0.0207              | 0.0280              | 0.0108          | 0.0084          | 0.0147               | 0.0246             | 0.0304    | 0.0225     |
| SSIM↑  | 0.5689       | 0.7571| 0.7358| 0.7781   | 0.7485         | 0.6867    | 0.6537              | 0.5559              | 0.6195          | 0.6543          | 0.6924               | 0.6737             | 0.5779    | 0.6878     |
| LPIPS↓ | 0.3532       | 0.1850| 0.1808| 0.1771   | 0.2842         | 0.2711    | 0.2444              | 0.2166              | 0.2746          | 0.2651          | 0.2403               | 0.2531             | 0.3629    | 0.2087     |

## Generate Video Frames with the Trained E2HQV



**Fix on 06/27/2024:** app.py line 144 replace the `p_states` to `current_states`: return rf0, f01.detach(), last_gt, current_states, all_output

**Note:** Due to the size limitation on GitHub, the complete code along with the model weights is stored on [Google Drive](https://drive.google.com/drive/folders/1h_Xq-VcwIIa4xWXhhFAHjZ_z6jSkIUwc?usp=drive_link).

* On Google Drive, we provide minimal code to predict video frames using event-streams represented as voxel grids with 5 temporal bins. This representation was proposed by Alex et al. in their [CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.pdf).

* An example sequence of voxel grids can be found in `./dataset/desk_fast_voxelgrid_5bins_examples`. To generate the corresponding frames, simply run `python3 app.py` in the terminal.

* If you wish to use E2HQV with your own event data, place your event temporal bins in the form of a 5xHxW numpy array saved in `.npy` format (to ./dataset/desk_fast_voxelgrid_5bins_examples). Then, execute `python3 app.py` to process your data. In the **Dataset Preparation** section, we will provide detailed instructions and the necessary code to convert raw event data into voxel format.

**Known Issue:** The training process did not utilize optical flow, unlike other methods such as E2VID. As a result, the temporal consistency is suboptimal.


### To Cite
<pre>
@inproceedings{qu2024e2hqv,
  title={E2HQV: High-Quality Video Generation from Event Camera via Theory-Inspired Model-Aided Deep Learning},
  author={Qu, Qiang and Shen, Yiran and Chen, Xiaoming and Chung, Yuk Ying and Liu, Tongliang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4632--4640},
  year={2024}
}
</pre>



## Dataset Preparation

You can find the `e2voxel_grid.py` script for converting events to voxel grids in [Google Drive](https://drive.google.com/drive/folders/1h_Xq-VcwIIa4xWXhhFAHjZ_z6jSkIUwc?usp=drive_link).
### To Be Updated
