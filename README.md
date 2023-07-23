# Attractiveness-Methods-Evaluation
This is a repository for my Master's Thesis. In it, I evaluated whether deep learning methods are sufficiently accurate at predicting human attractiveness to be used in scientific research. 

My motivation for this came when reading a study in which researchers used a computational method to determine how attractive a sample of politicians are, and then to check how that related to their vote share. I realized that they'd never shown how effective their method actually was, only that it produced a statistically significant result. Upon checking other papers, I found this was a widespread problem in the field. While the methods used varied widely, I chose to focus on one in particular: neural networks. Several published studies in the past few years have used neural networks to rate human attractiveness, but have not checked how accurate these methods are. I tackled that subject for my thesis, evaluating neural network accuracy on new images "from the wild" over three studies. To do so, I performed image preparation in Python, trained neural networks using Pytorch and Amazon products like Sagemaker and S3, and conducted statistical analysis in R.

Samples of the code used have been provided in this repository. Other parts I could not share publically due to privacy concerns. Below I will explain the three studies I conducted and the methods used on each. 

# Study 1

Numerous deep learning methods (primarily neural networks) already exist for predicting human attractiveness. Largely trained on the SCUT-FBP5500 dataset, a collection of 5500 facial images, they perform exceptionally well on test sets. The best performing I could find, the [ComboLoss](https://github.com/lucasxlu/ComboLoss) model based on SEResNeXt50 backbone from Xu & Xiang, achieves an average Pearson Correlation of 0.9199, with a Root Mean Square Error of only 0.27 (Scores are from 1-5) across a 5-Fold Cross Validation on the SCUT-FBP5500 dataset. Scores from other models are not far behind. If these models perform as well on other sets of images, their accuracy would almost certainly be high enough to produce good research. As a first step, therefore, I found eight publically available pre-trained models and set them up in Python, using GPUs from Google Colab, to conduct inference. 

The images I planned to conduct inference on came from four published papers whose authors generously shared their data with me. In each paper, the authors modeled the vote share a politician received on the attractiveness, and found a positive correlation. To see the accuracy on the data generated from these deep learning methods, then, I chose to conduct two tests: firstly, I would calculate the RMSE, MAE, and PC of the generated data versus the original data (after both were Z-standardized), and secondly, I would replicate each of the studies. The only difference between the original study and the replication would be that the attractiveness variable data would come from a neural network instead os human survey respondents. 

Before doing this, however, I had one final step. The training images for the models were standardized, with completely vertical faces, no backgrounds, and fairly regular cropping. As I didn't know the original method used to produce the images, I made several different modifications ([removing backgrounds](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/Background%20Remover.py), [rotating heads](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/image_preparation.py), cropping the images closely) to each set of images before conducting inference on each set. As several image modifications mixed poorly with some of the datasets, this left me with a total of 171 image-modification-methods combinations to test. 

<img src="[[http://....jpg](images/image augmentation.JPG)](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/image%20augmentation.JPG)" width="200" height="200" />


![Image Preparation](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/dfdc3c6b012a88735c74f961a9a1a57d8292dfe0/images/image%20augmentation.JPG "Logo Title Text 1")



