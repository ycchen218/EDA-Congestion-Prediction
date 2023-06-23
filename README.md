# EDA Congestion Prediction
## Update
- 2023/6/24

  Add the train.py file.
## Introduce
This is a deep-learning-based model used to predict the location of congestion. We achieved an SSIM score of 0.863. During the prediction process, we incorporate the macro region feature, and Rectangular Uniform Wire Density (RUDY)**[1]**, which serves as an early estimation of routing demand after placement. and also consider the RUDY pins, which are calculated based on individual pins and the nets connected to them.
## Requirement
1. python3.8
2. scipy
3. matplotlib
4. numpy
5. pandas
6. pytorch 1.12.0
## Model Overview
![image](https://github.com/ycchen218/EDA-Congestion-Prediction/blob/master/git_image/Congestion_overview.png)
## Train
```markdown
python train.py
```
--root_path: The path of the data file <br>
--batch_size: The batch size figure <br>
--num_epochs: The training epochs <br>
--learning_rate: learning rate [0,1] <br>
--weight_path: The path to save the model weight <br>
--fig_path: The path of the figure file <br>
## Predict
```markdown
python predict.py
```
--data_path: The path of the data file <br>
--fig_save_path: The path you want to save figure <br>
--weight_path: The path of the model weight <br>
--output_path: The path of the predict output with .npy file <br>
--congestion_threshold: congestion_threshold [0,1] <br>
--device: If you want to use gpu type "cuda" <br>
## Predict result
1. Tune your own congestion_threshold, the defalt is 0.5 as shown in following figure.
2. The output coordinate csv file and image array npy file are in the ./output file.
3. The model predict cost time is **0.71 ~ 1.8 sec**.
4. Congestion value: $normalize(\frac{overflow}{totaltracks})$.


![image](https://github.com/ycchen218/EDA-Congestion-Prediction/blob/master/save_img/congestion_0.2.png)
## Compare with ground truth
![image](https://github.com/ycchen218/EDA-Congestion-Prediction/blob/master/git_image/compare1.png)
## Cross validation while evalulate the model
SSIM score: 0.863 <br>
by pytorch_msssim.SSIM
## Reference
```markdown
[1] P. Spindler and F. M. Johannes, "Fast and accurate routing demand estimation for efficient routability-driven placement," *Design, Automation & Test in Europe Conference & Exhibition*, pp. 1-6, 2007.
```
