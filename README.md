LabelInspect
======================
Codes for the interactive analysis tool, LabelInspect, described in our paper "An Interactive Method to Improve Crowdsourced Annotations."

Requirements
----------
```
tqdm==4.36.1
numpy==1.19.0
matplotlib==2.1.0
pandas==0.25.1
scipy==1.5.0
Flask==1.1.1
scikit_learn==0.21.3
anytree==2.8.0
Flask-Cors==3.0.9
Flask-Script==2.0.6
Flask-Session==0.3.2
ipython==7.16.1
joblib==0.16.0
```

Quick Start with Demo Data
-----------------
Step 1: download demo data from [here](https://cloud.tsinghua.edu.cn/f/c397fedb3bf849208e4a/), and unpack it in the root folder LabelInspect.

Step 2: setup the system:
```python server.py```

Step 3: visit http://localhost:8181/ in a browser.

## Citation
If you use this code for your research, please consider citing:
```
@article{liu2018crowsourcing,
  title={An Interactive Method to Improve Crowdsourced Annotations},
  author={Liu, Shixia and Chen, Changjian and Lu, Yafeng and Ouyang, Fangxin and Wang, Bin},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  volume={25},
  number={1},
  pages ={235--245},
  year={2019},
}
```

## Contact
If you have any problem about this code, feel free to contact
- ccj17@mails.tsinghua.edu.cn

or describe your problem in Issues.
