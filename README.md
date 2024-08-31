# Inertial Proximal Difference-of-Convex Algorithm with Convergent Bregman Plug-and-Play
[**Tsz Ching Chow**](https://github.com/nicholechow), [**Chaoyan Huang**](https://scholar.google.com/citations?user=Sun7dRgAAAAJ&hl=en&oi=ao), Zhongming Wu, Tieyong Zeng, Angelica I. Aviles-Rivero

<!-- xxxxx  needs to be filled in -->
[[paper arxiv](https://arxiv.org/pdf/xxxxx.pdf)]  

## News
- 2024.09.03 : Initial release of the codebase for the paper "Inertial Proximal Difference-of-Convex Algorithm with Convergent Bregman Plug-and-Play". Code will be released soon after the paper is accepted.

## Prerequisites:
This code requires `torch>= 1.8.0` and `pytorch-lightning`. Please install dependencies by running the following command:
```
pip install -r requirements.txt
```



Rician Noise Removal 
----------
[<img src="./raw/PDw.gif" width="300px"/>](https://imgsli.com/MjkyNzc1) 
[<img src="./raw/T2w.gif" width="300px"/>](https://imgsli.com/MjkyNzcz) 

- Denoising Results on BrainWeb dataset

<img src="raw/rician_table.png" width="600px"/> 



Phase Retrieval
----------
[<img src="./raw/Pollen.gif" width="300px"/>](https://imgsli.com/MjkyNzc1) 
[<img src="./raw/TadpoleGalaxy.gif" width="300px"/>](https://imgsli.com/MjkyNzcz) 
<img src="raw/pr_table.png" width="600px"/> 
* for visual demonstration purpose only! 




Citation
----------
```
@article{chow2024inertial,
  title={Inertial Proximal Difference-of-Convex Algorithm with Convergent Bregman Plug-and-Play},
  author={Chow, Tsz Ching and Huang, Chaoyan and Wu, Zhongming and Zeng, Tieyong and Aviles-Rivero, Angelica I.},
  journal={arXiv preprint xxxxx},
  year={2024}
}

```