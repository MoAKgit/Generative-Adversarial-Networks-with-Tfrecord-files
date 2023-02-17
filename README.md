
# Generative Adversarial Networks with Tfrecord files

This is a tensorflow implementation of GAN with tfrecord files. 
To use the code first install following requirments packages:

- tensorflow-gpu
- glob
- os
- pillow
- matplotlib


Download the flower dataset from the link below:
https://www.tensorflow.org/datasets/catalog/tf_flowers
and collect the five classes (daisy, dandelion, roses, sunflowers, tulips) in one folder.


Use the file 'Convert_jpg_to_tfrecord.py' to convert all images to a trfecord file named 'flower.trfecord'.


Copy the flowers.tfrecord to the directory of 'tfrecord files'.


Run the gan_tfrecord.py to start training the GAN model using the tfrecord file.


Notice that, tensorboard has been provided in the code for further analysis.

The trained weights are saved in the directory 'saved_model/'

For the sake of the time, I just uploaded the tf record file in the directory of 'tfrecord files'

The number of epochs and learning rate are set to 200 and 0.0005 , respectively.

I printed out the details of the Generator and discriminator as follows:

![My Image](G.PNG)  

![My Image](D.PNG)  


For testing the model, change the defualt of the 'mode' in the arguments to 'test' and run the 'gan_tfrecord.py' agian. In this time, the code will automatically load the pretrained weights of genrator and then generate random flowers.
Here are some generated images:

![My Image](flower1.png)  
![My Image](flower2.png) 


