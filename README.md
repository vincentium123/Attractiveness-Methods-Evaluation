This repository contains the code for my Master’s Thesis in Political Science at the University of Mannheim. For my thesis, I chose a methods-oriented project: whether a practice in social science research, using neural networks to gauge the attractiveness of people in images, gives sufficiently accurate results to be used. 

To answer this question, I used Python to prepare images, R to conduct studies, and Pytorch, alongside Amazon Sagemaker and S3, to train deep learning computer vision models. 

## A Brief Summary

Some academic researchers have started using neural networks to judge the attractiveness of people in photographs and then using those values in the research they publish. This sounds bizarre, but there’s logic behind it- research has shown that humans have a consistent bias towards attractive people. The more attractive someone is, the more votes they win in elections, the more money they receive from investors, and the more positively people think of them overall. Typically, research in this field has been done using large-scale surveys, but they’re expensive and time-consuming. Properly trained neural networks, on the other hand, can grade thousands of images in minutes. 

There’s a problem, however, that no researchers have yet dealt with- seeing how accurate these neural networks are. Often the trained neural networks are trained on the bulk of a dataset (80-90%) then tested on the remainder. That’s fine, but what it doesn’t cover is how well the neural networks do on a new dataset- in other words, how generalizable they are. 

That’s the task I set for myself. I first tested eight publicly available pre-trained neural networks by attempting to replicate the results of five published studies (the authors kindly shared their data with me). As they largely failed to replicate the studies, I then trained two new neural networks on a newer dataset that I hoped corrected the flaws of earlier datasets. With these two new networks, I then ran the five studies again and conducted a new study on my own. Results were better, but far from perfect. 

In the end, I concluded that researchers should not use neural networks to judge attractiveness in their research without making some verifiable improvements to their predictive ability. 

## Technical Details

The section before covered the basics of my project. Now, I’ll dive into the technical details. Many parts of my code can be found in this repository, but some I’ve had to hold back for privacy reasons (the authors of the papers I replicated asked me not to share any data). 

With that, let’s get started. 

### Existing Neural Networks

A surprising number of neural networks for gauging attractiveness can already be found on public Githubs or are accessible via APIs. I set out to test as many as possible. Here are the model architectures. 

They are: 
1. [ComboLoss](https://github.com/lucasxlu/ComboLoss)
2. [ResNet-18](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
3. [BeholderNet](https://github.com/beholdergan/Beholder-GAN)
4. [CRNet](https://github.com/lucasxlu/CRNet)
5. [HMTNet](https://github.com/lucasxlu/HMTNet)
6. [Face++](https://www.faceplusplus.com/)
7. [Baidu](https://github.com/miracleyoo/Face-Recognition-Using-Baidu-API)

The last two are available via APIs- unfortunately there’s no public information about the training methods or network architectures. 

To test how accurate these methods are at predicting scores on new images from outside their training/testing set, I attempted to replicate five published scientific studies. Each study used attractiveness, as measured by human raters, as an independent variable. 

Before conducting inferences on the images, I did a variety of image manipulation techniques- removing backgrounds, rotating them, using [MTCNN](https://github.com/timesler/facenet-pytorch/tree/master) to crop and center them- in order to better match the training images. I applied them separately and in combination with each other, as I didn’t know beforehand which combination would produce the best results.

<h3><img align="center" height="300" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/image%20augmentation.JPG"></h3>

This left me with 171 combinations of studies-models-images. I used each model to conduct inference on each set of images, then used the results in the original regression models for each study. I counted a study as successfully replicated if I saw two things: the attractiveness variable was significant at the 5% level and the attractiveness coefficient had approximately the same real-world effect. 

To replicate the studies, I first conducted inference on the images in Google Colab (the neural networks run best on GPUs, and my laptop doesn’t have one). To do so, I created a folder with numerous subfolders, each other containing a different combination of study-images (i.e. photos with no background from Study 1). I wrote a script in Python that would conduct inference for each set of images and save the results as a separate csv file. Once I had all my data, I fed it into scripts in R that cleaned it, standardized the values, merged it with the original dataset from the study, conducted whatever regression model was used in the original study, then outputted the results into two csv files. The first showed the results of the regression and the second the Pearson Correlation, Root Mean Square Error, and Mean Absolute Error between the data generated from the neural network and the original data. 

Unfortunately, the models struggled to produce good real-world results. 

| Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11 | #12 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269 | 254 |

Only a minority of combinations produced significant results in all studies, and three studies showed no significant results at all. Of the combinations that produced significant results, most did produce coefficients in-line with the original studies. In this case, I defined this as of the same approximate real-world implications. Three, however, did not, meaning that only 8.1% of the model-study-image combinations replicated successfully- hardly a rousing success. 

Much of the trouble can be seen in the table below. On their test set of their original dataset, many of these models performed extremely well- Pearson Correlations of around 0.9, RMSEs of 0.25-0.35 (Scores were typically from 1-5). On these new images, however, results were often quite poor.

INSERT IMAGE

I developed a theory as to why- these models were trained on very rigid datasets. The most common dataset, [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) is quite well-made, but the images are highly similar to each other- front-facing, good lighting, no background, cropped in about the same way. The images used in these studies, however, are much more variable and naturalistic. There are differences in lighting, cropping, the tilt of the head, and other factors. There were other potential differences as well. The images from the studies have many older people; SCUT-FBP is largely young people. The studies I replicated were conducted in Europe, the US, and Australia; SCUT-FBP’s ratings come from Chinese people. The studies used many average-looking people; SCUT-FBP had disproportionately many highly attractive people. All of these were potential confounders, which I hoped to avoid. 

INSERT EXAMPLE IMAGES

### Training my own networks

To train my own neural networks, I selected a newly created dataset: [MEBeauty](https://github.com/fbplab/MEBeauty-database). Previous datasets tended to be largely formal, and had little diversity in age and ethnicity. MEBeauty, however, contains a wide variety of images from around the world, including of older people. The images are also more naturalistic(different lighting, different poses, etc.) and were created in a standardized way that was easy to copy. 

I trained several new neural networks on it, and selected the two best performing ones. Both were versions of the [ComboLoss](https://github.com/lucasxlu/ComboLoss) model. They had identical hyperparameters, except one had a batch size of 16 while the other had a batch size of 32. 
SPECS

These models performed worse on their test sets than models trained on the SCUT-FBP5500 dataset, which I believe is due to the greater variety of the images. Their performance was similar to models created by the developers of the MEBeauty dataset. 

Once I had my models, I set out to test them in two ways. Firstly, I attempted to replicate the five studies again, using three scores: one from each of the models and a third composite score of their averages on each image. 

My results were mixed. Most studies, once again, did not replicate. Furthermore, the results were confusing. The two models’ performances diverged even though on the test sets they’d been extremely similar. What’s more, it was different studies that replicated this time. 

TABLE

Following these replications, I attempted one more test of my neural networks. A consistent finding in the literature has been that right-wing politicians are slightly more attractive than left-wing ones. No one knows why, but it’s been found in studies from Finland, the US, Germany, Australia, and the UK. 

I therefore conducted a quick test on images of members of the US Congress from 2011 to the present, using members of the Democratic Party as left-wing politicians and Republicans as right-wing. All images came from [Voteview’s Member Photos Repository](https://github.com/voteview/member_photos). 

As a first step, I did a t-test, which did not show a significant difference. As, however, there are several major difference between the parties- Republican politicians are older and more likely to be white and male, I conducted a linear regression as well. For a dependent variable, I used the attractiveness scores from my models. For my independent and control variables, I chose party, age, chamber (House/Senate), and gender. 

STARGAZER

Results from the OLS models largely line up with previous findings. Both neural networks find that Republican legislators are positively correlated with attractiveness at the 10% level, while the average finds the effect at the 5% level. This is even more suggestive than at first glance- I ran my neural networks over the same images three times, each time generating somewhat different results. In the other two instances, the neural network with batch size 16 found the Republican-attractiveness correlation at the 5% level. The effect size, however, is fairly marginal. Attractiveness in this study is measured from 1-10, and the effect of being a Republican is only 0.065- hardly important in real life. This is smaller than the effect found in other studies.

The control variables also contain some interesting information. Previous studies have found no difference in attractiveness between male and female politicians, but in this study, male politicians are significantly less attractive than female politicians. Unlike with party, this effect is quite large- between 0.6 and 0.8 on a 10-point scale. Likewise, age has not been found to be significantly correlated with attractiveness in some studies that use human raters. Here, however, it is, although the difference is fairly small. A thirty-year age gap would only be associated with a difference of around 0.3, on average. Finally, that Senators are more attractive than Representatives is in line with existing theories. As more attractive politicians win more votes and Senators have typically gone through more (and more difficult) elections than Representatives, it’s logical that selection pressure would result in Senators being more attractive on average. Studies from business also cast another angle- attractive founders can raise funds more easily, and a key part of an American politician’s job is fundraising. Better fundraisers, meanwhile, are more likely to become Senators.

## Conclusion

Throughout these studies, one thing has been clear: deep learning methods, at least at their present state, are not accurate enough at consistently gauging human attractiveness to be used in scientific research. They do sometimes produce accurate results, but it is difficult to tell beforehand if they will. Throughout these studies, deep learning methods which performed extremely well on one dataset failed completely on another, despite clear similarities. This extends even to those which performed well on test datasets post-training. Even worse, two highly similar deep learning methods can have similar results on a test set, but diverge when applied to real-world questions. The black-box nature of deep learning makes it nearly impossible to judge beforehand. 
