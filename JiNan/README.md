This is the code of COGCN in JiNan dataset!

This is a document for this code++++++++++++++++++++++++++++++++++++++++++++++++++++++++=>

***First, the structure of the code:

util.py: The reading of data and other functions except of Model

utils.py: all base blocks and functions of neural network models in our paper

model.py: construct model based on utils.py

engine.py: the program of training model in our paper

train.py: main document of this code

***How to run these files?

In jupyter ,you should write:

Firstly, you should run generate_data.py and genenrate_adjacent.py to generate all data and adjacent matrix of JiNan datasets! 

secondly,

run train.py --force True --model (you can select from COGCN or COGCN_noloss(=COGCNNL in paper)) --CL True (or False) 

--CL: True is COGCN and False is COGCN_noloss

or 

python train.py --force True --model (you can select from COGCN or COGCN_noloss(=COGCNNL in paper)) --CL True (or False) 

The dataset and saving model can be found in BaiduYun: website：https://pan.baidu.com/s/1uXSDLngQMeQv__-UsBCDeQ key：mrcj 

