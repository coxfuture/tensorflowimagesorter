#Tensorflow based image sorter

This program expects you have your data set in the following scheme:

```
-main folder
--Data
---unsorted images
--Dataset
---class 1
---class 2
---class 3
---etc
--Sorted
---class 1
---class 2
---class 3
---etc```

To run, first run:
	
	`pip install -r requirements-cpu.txt`
	
	or:
	
	`pip install -r requirements-gpu.txt`
	
	if you have a GPU and CUDA installed. 
	
then run:

`python train.py --num_classes <number of image classes> --batch_size <recommended 32> --epochs <10-20 recommended>`

and wait for training to complete. This will create a classification model and save it under ./Model/1

then run: 

`python sort.py `

And it will copy the images from ./Data into their respective folders. 

Tested with the Google flower_photos dataset (5 classes, 3600 images), and a dataset for a project I was working on (20 classes, 40,000 images) and produced quite good results. 
