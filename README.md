# Attractiveness-Methods-Evaluation
This is a repository for my Master's Thesis. In it, I evaluated whether deep learning methods are sufficiently accurate at predicting human attractiveness to be used in scientific research. 

My motivation for this came when reading a study in which researchers used a computational method to determine how attractive a sample of politicians are, and then to check how that related to their vote share. I realized that they'd never shown how effective their method actually was, only that it produced a statistically significant result. Upon checking other papers, I found this was a widespread problem in the field. While the methods used varied widely, I chose to focus on one in particular: neural networks. Several published studies in the past few years have used neural networks to rate human attractiveness, but have not checked how accurate these methods are. I tackled that subject for my thesis, evaluating neural network accuracy on new images "from the wild" over three studies. To do so, I performed image preparation in Python, trained neural networks using Pytorch and Amazon products like Sagemaker and S3, and conducted statistical analysis in R.

Samples of the code used have been provided in this repository. Other parts I could not share publically due to privacy concerns. 

#The State of the Field#
