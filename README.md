# MDPS

## CCF
**ICME belongs to CCF-B.**

![image](https://user-images.githubusercontent.com/19371493/189510285-5d88b69f-bb8d-45b3-9870-02c7fbcebb9d.png)

## Idea
![image](https://user-images.githubusercontent.com/19371493/189364274-4fc4b756-13d5-4b0f-8487-4d188bf415a0.png)

## Code
[Unofficial] [pytorch] the implementation of "A Rolling Bearing Fault Diagnosis Method Using Multi-Sensor Data and Periodic Sampling."

[official code](https://github.com/IWantBe/MDPS)   ---tensorflow version （Thanks to [Dr. Zheng](https://github.com/IWantBe) for the open source code）

- ReadData_2.py : Channel fusion and periodic sampling.
- model_2_view.py : the code of training and testing, and it provides functions to evaluate the performance changes under different stripe and sample length.

## Cited
```html
@inproceedings{DBLP:conf/icmcs/Zheng0Z022,
  author    = {Jianbo Zheng and
               Chao Yang and
               Fangrong Zheng and
               Bin Jiang},
  title     = {A Rolling Bearing Fault Diagnosis Method Using Multi-Sensor Data and
               Periodic Sampling},
  booktitle = {{IEEE} International Conference on Multimedia and Expo, {ICME} 2022,
               Taipei, Taiwan, July 18-22, 2022},
  pages     = {1--6},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/ICME52920.2022.9859658},
  doi       = {10.1109/ICME52920.2022.9859658},
  timestamp = {Wed, 31 Aug 2022 11:49:15 +0200},
  biburl    = {https://dblp.org/rec/conf/icmcs/Zheng0Z022.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

