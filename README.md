# An Evaluation of Using Deep Learning Techniques in Social Science Research to Judge Attractiveness

**This is the English version. Die deutsche Version finden Sie weiter unten.**

This repository contains the code for my Master’s Thesis in Political Science at the University of Mannheim. For my thesis, I chose a methods-oriented project: whether a practice in social science research, using neural networks to gauge the attractiveness of people in images, gives sufficiently accurate results to be used. 

To answer this question, I used Python to prepare images, R to conduct studies, and Pytorch, alongside Amazon Sagemaker and S3, to train deep learning computer vision models. 

## A Brief Summary

Some academic researchers have started using neural networks to judge the attractiveness of people in photographs and then using those values in the research they publish. This sounds bizarre, but there’s logic behind it- research has shown that humans have a consistent bias towards attractive people. The more attractive someone is, the more votes they win in elections, the more money they receive from investors, and the more positively people think of them overall. Typically, research in this field has been done using large-scale surveys, but they’re expensive and time-consuming. Properly trained neural networks, on the other hand, can grade thousands of images in minutes. 

There’s a problem, however, that no researchers have yet dealt with- seeing how accurate these neural networks are. Often the trained neural networks are trained on the bulk of a dataset (80-90%) then tested on the remainder. That’s useful information, but what it doesn’t cover is how well the neural networks do on a new dataset- in other words, how generalizable they are. 

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
4. [CRNet](https://github.com/lucasxlu/CRNet)- two versions, trained on two different datasets
5. [HMTNet](https://github.com/lucasxlu/HMTNet)
6. [Face++](https://www.faceplusplus.com/)
7. [Baidu](https://github.com/miracleyoo/Face-Recognition-Using-Baidu-API)

The last two are available via APIs- unfortunately there’s no public information about the training methods or network architectures. 

To test how accurate these methods are at predicting scores on new images from outside their training/testing set, I attempted to replicate five published scientific studies. Each study used attractiveness, as measured by human raters, as an independent variable. 

Before conducting inferences on the images, I did a variety of [image manipulation techniques](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/Background%20Remover.py)- removing backgrounds, rotating them, using [MTCNN](https://github.com/timesler/facenet-pytorch/tree/master) to crop and center them- in order to better match the training images. I applied them separately and in combination with each other, as I didn’t know beforehand which combination would produce the best results.

<h3><img align="center" height="300" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/image%20augmentation.JPG"></h3>

This left me with 171 combinations of studies-models-images. I used each model to conduct inference on each set of images, then used the results in the original regression models for each study. I counted a study as successfully replicated if I saw two things: the attractiveness variable was significant at the 5% level and the attractiveness coefficient had approximately the same real-world effect. 

To replicate the studies, I first conducted inference on the images in Google Colab (the neural networks run best on GPUs, and my laptop doesn’t have one). To do so, I created a folder with numerous subfolders, each containing a different combination of study-images (i.e. photos with no background from Study 1). I wrote a [script in Python](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/inference.py) that would conduct inference for each set of images and save the results as a separate csv file. Once I had all my data, I fed it into [scripts in R](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/paper%20replication.Rmd) that cleaned it, standardized the values, merged it with the original dataset from the study, conducted whatever regression model was used in the original study, then outputted the results into two csv files. The first showed the results of the regression and the second the Pearson Correlation, Root Mean Square Error, and Mean Absolute Error between the data generated from the neural network and the original data. 

Unfortunately, the models struggled to produce good real-world results. 

| Study | #1 | #2 | #3 | #4 | #5 | #6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Attempts Replicated (%) | 0 | 0 | 0 | 2.7 | 10.8 | 27.0 |

Only a minority of combinations produced significant results in all studies, and three studies showed no significant results at all. Of the combinations that produced significant results, most did produce coefficients in-line with the original studies. In this case, I defined this as of the same approximate real-world implications. Three, however, did not, meaning that only 8.1% of the model-study-image combinations replicated successfully- hardly a rousing success. 

Much of the trouble can be seen in the table below. On their test set of their original dataset, many of these models performed extremely well- Pearson Correlations of around 0.9, RMSEs of 0.25-0.35 (Scores were typically from 1-5). On these new images, however, results were often quite poor.

| Measure | RMSE | PC | MAE |
| :---: | :---: | :---: | :---: |
| Average (Across All Datasets) | 1.24 | 0.23 | 1.00 |

I developed a theory as to why- these models were trained on very rigid datasets. The most common dataset, [SCUT-FBP5500,](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) is quite well-made, but the images are highly similar to each other- front-facing, good lighting, no background, cropped in about the same way. The images used in these studies, however, are much more variable and naturalistic. There are differences in lighting, cropping, the tilt of the head, and other factors. There were other potential differences as well. The images from the studies have many older people; SCUT-FBP is largely young people. The studies I replicated were conducted in Europe, the US, and Australia; SCUT-FBP’s ratings come from Chinese people. The studies used many average-looking people; SCUT-FBP had disproportionately many highly attractive people. All of these were potential confounders, which I hoped to avoid. 

### Training my own networks

To train my own neural networks, I selected a newly created dataset: [MEBeauty](https://github.com/fbplab/MEBeauty-database). Previous datasets tended to be largely formal, and had little diversity in age and ethnicity. MEBeauty, however, contains a wide variety of images from around the world, including of older people. The images are also more naturalistic (different lighting, different poses, etc.) and were created in a standardized way that was easy to copy. 

<h3><img align="center" height="300" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/ME3.png"></h3>


I [trained](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/training.ipynb) several new neural networks on it, and selected the two best performing ones. Both were versions of the [ComboLoss](https://github.com/lucasxlu/ComboLoss) model. They had identical hyperparameters, except one had a batch size of 16 while the other had a batch size of 32. 

| Model | RMSE | MAE | PC |
| :---: | :---: | :---: | :---: |
| ComboLoss (MEBeauty) (BS 16) | 1.07 | 0.82 | 0.64 |
| ComboLoss (MEBeauty) (BS 32) | 1.05 | 0.84 | 0.67 |
| ComboLoss (SCUT-FBP5500) | 0.21 | 0.27 | 0.92 |

These models performed worse on their test sets than models trained on the SCUT-FBP5500 dataset, which I believe is due to the greater variety of the images. Their performance was similar to models created by the developers of the MEBeauty dataset. 

Once I had my models, I set out to test them in two ways. Firstly, I attempted to replicate the studies again, using three scores: one from each of the models and a third composite score of their averages on each image. 

My results were mixed. Most studies, once again, did not replicate. Furthermore, the results were confusing. The two models’ performances diverged even though on the test sets they’d been extremely similar. What’s more, it was different studies that replicated this time. 

<h3><img align="center" height="200" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/mymodel_pvalues.JPG"></h3>

Following these replications, I attempted one more test of my neural networks. A consistent finding in the literature has been that right-wing politicians are slightly more attractive than left-wing ones. No one knows why, but it’s been found in studies from Finland, the US, Germany, Australia, and the UK. 

I therefore conducted a test on images of members of the US Congress from 2011 to the present, using members of the Democratic Party as left-wing politicians and Republicans as right-wing. All images came from [Voteview’s Member Photos Repository](https://github.com/voteview/member_photos). 

As a first step, I did a t-test, which did not show a significant difference. As, however, there are several major difference between the parties- Republican politicians are older and more likely to be white and male- I conducted a linear regression as well. For a dependent variable, I used the attractiveness scores from my models. For my independent and control variables, I chose party, age, chamber (House/Senate), and gender. 

<h3><img align="center" height="400" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/stargazer%20congress.JPG"></h3>

Results from the OLS models largely line up with previous findings. Both neural networks find that Republican legislators are positively correlated with attractiveness at the 10% level, while the average finds the effect at the 5% level. This is even more suggestive than at first glance- I ran my neural networks over the same images three times, each time generating somewhat different results. In the other two instances, the neural network with batch size 16 found the Republican-attractiveness correlation at the 5% level. The effect size, however, is fairly marginal. Attractiveness in this study is measured from 1-10, and the effect of being a Republican is only 0.065- hardly important in real life. This is smaller than the effect found in other studies.

The control variables also contain some interesting information. Previous studies have found no difference in attractiveness between male and female politicians, but in this study, male politicians are significantly less attractive than female politicians. Unlike with party, this effect is quite large- between 0.6 and 0.8 on a 10-point scale. Likewise, age has not been found to be significantly correlated with attractiveness in some studies that use human raters. Here, however, it is, although the difference is fairly small. A thirty-year age gap would only be associated with a difference of around 0.3, on average. Finally, that Senators are more attractive than Representatives is in line with existing theories. As more attractive politicians win more votes and Senators have typically gone through more (and more difficult) elections than Representatives, it’s logical that selection pressure would result in Senators being more attractive on average. Studies from business also cast another angle- attractive founders can raise funds more easily, and a key part of an American politician’s job is fundraising. Better fundraisers, meanwhile, are more likely to become Senators.

## Conclusion

Throughout these studies, one thing has been clear: deep learning methods, at least at their present state, are not accurate enough at consistently gauging human attractiveness to be used in scientific research. They do sometimes produce accurate results, but it is difficult to tell beforehand if they will. Throughout these studies, deep learning methods which performed extremely well on one dataset failed completely on another, despite clear similarities. This extends even to those which performed well on test datasets post-training. Even worse, two highly similar deep learning methods can have similar results on a test set, but diverge when applied to real-world questions. The black-box nature of deep learning makes it nearly impossible to judge beforehand. 

***

# Eine Bewertung des Einsatzes von Deep Learning-Techniken in der sozialwissenschaftlichen Forschung zur Beurteilung der Attraktivität

Dieses Repository enthält den Code für meine Masterarbeit in Politikwissenschaft an der Universität Mannheim. Für meine Arbeit wählte ich ein methodenorientiertes Projekt: ob eine Praxis in der sozialwissenschaftlichen Forschung, die Verwendung neuronaler Netze zur Bewertung der Attraktivität von Menschen auf Bildern, ausreichend genaue Ergebnisse liefert, um verwendet zu werden. 

Um diese Frage zu beantworten, habe ich Python zur Aufbereitung von Bildern, R zur Durchführung von Studien und Pytorch zusammen mit Amazon Sagemaker und S3 zum Training von Deep-Learning-Computer-Vision-Modellen verwendet. 

## Eine kurze Zusammenfassung

Einige akademische Forscher haben damit begonnen, neuronale Netze einzusetzen, um die Attraktivität von Menschen auf Fotos zu beurteilen und diese Werte dann in ihren Veröffentlichungen zu verwenden. Das hört sich bizarr an, hat aber eine gewisse Logik: Die Forschung hat gezeigt, dass der Mensch attraktive Menschen immer bevorzugt. Je attraktiver jemand ist, desto mehr Stimmen erhalten sie bei Wahlen, desto mehr Geld erhalten sie von Investoren und desto positiver denken die Menschen insgesamt über sie. In der Regel wurde die Forschung auf diesem Gebiet mit groß angelegten Umfragen durchgeführt, die jedoch teuer und zeitaufwändig sind. Richtig trainierte neuronale Netze hingegen können Tausende von Bildern in wenigen Minuten bewerten. 

Es gibt jedoch ein Problem, mit dem sich noch kein Forscher befasst hat: Wie genau sind diese neuronalen Netze? Häufig werden die trainierten neuronalen Netze mit dem Großteil eines Datensatzes (80-90 %) trainiert und dann mit dem Rest getestet. Das ist eine nützliche Information, aber sie sagt nichts darüber aus, wie gut die neuronalen Netze in einem neuen Datensatz funktionieren - mit anderen Worten, wie verallgemeinerbar sie sind. 

Das ist die Aufgabe, die ich mir gestellt habe. Zunächst testete ich acht öffentlich verfügbare vortrainierte neuronale Netze, indem ich versuchte, die Ergebnisse von fünf veröffentlichten Studien zu replizieren (die Autoren stellten mir ihre Daten zur Verfügung). Da es ihnen nicht gelang, die Studien zu replizieren, trainierte ich zwei neue neuronale Netze auf einem neueren Datensatz, von dem ich hoffte, dass er die Fehler der früheren Datensätze korrigierte. Mit diesen beiden neuen Netzen habe ich dann die fünf Studien erneut durchgeführt und eine neue Studie auf eigene Faust durchgeführt. Die Ergebnisse waren besser, aber noch lange nicht perfekt. 

Am Ende kam ich zu dem Schluss, dass Forscher keine neuronalen Netze zur Beurteilung der Attraktivität in ihrer Forschung einsetzen sollten, ohne ihre Vorhersagefähigkeit nachweislich zu verbessern. 

## Technische Details

Im vorigen Abschnitt habe ich die Grundlagen meines Projekts erläutert. Jetzt werde ich auf die technischen Details eingehen. Viele Teile meines Codes sind in diesem Repository zu finden, aber einige musste ich aus Datenschutzgründen zurückhalten (die Autoren der Papiere, die ich repliziert habe, haben mich gebeten, keine Daten weiterzugeben). 

Also, fangen wir an. 

### Bestehende neuronale Netze

Eine überraschende Anzahl von neuronalen Netzen zur Bewertung der Attraktivität findet sich bereits auf öffentlichen Githubs oder ist über APIs zugänglich. Ich habe mir vorgenommen, so viele wie möglich zu testen. Hier sind die Modellarchitekturen. 

Sie sind: 
1. [ComboLoss](https://github.com/lucasxlu/ComboLoss)
2. [ResNet-18](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
3. [BeholderNet](https://github.com/beholdergan/Beholder-GAN)
4. [CRNet](https://github.com/lucasxlu/CRNet)- zwei Versionen, trainiert auf zwei verschiedenen Datensätzen
5. [HMTNet](https://github.com/lucasxlu/HMTNet)
6. [Face++](https://www.faceplusplus.com/)
7. [Baidu](https://github.com/miracleyoo/Face-Recognition-Using-Baidu-API)

Die letzten beiden sind über APIs verfügbar - leider gibt es keine öffentlichen Informationen über die Trainingsmethoden oder Netzwerkarchitekturen. 

Um zu testen, wie genau diese Methoden bei der Vorhersage von Bewertungen für neue Bilder sind, die nicht aus ihrer Trainings-/Testgruppe stammen, habe ich versucht, fünf veröffentlichte wissenschaftliche Studien zu replizieren. In jeder Studie wurde die von menschlichen Bewertern gemessene Attraktivität als unabhängige Variable verwendet. 

Bevor ich Rückschlüsse auf die Bilder gezogen habe, habe ich eine Reihe von [Bildmanipulationstechniken}(https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/Background%20Remover.py) angewandt - Entfernen von Hintergründen, Drehen, Zuschneiden und Zentrieren mit [MTCNN](https://github.com/timesler/facenet-pytorch/tree/master), um sie besser an die Trainingsbilder anzupassen. Ich wandte sie einzeln und in Kombination miteinander an, da ich vorher nicht wusste, welche Kombination die besten Ergebnisse erzielen würde.

<h3><img align="center" height="300" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/image%20augmentation.JPG"></h3>

Damit blieben 171 Kombinationen von Studien, Modellen und Bildern übrig. Ich verwendete jedes Modell, um Inferenzen für jeden Satz von Bildern durchzuführen, und verwendete dann die Ergebnisse in den ursprünglichen Regressionsmodellen für jede Studie. Ich betrachtete eine Studie als erfolgreich repliziert, wenn ich zwei Dinge feststellen konnte: Die Attraktivitätsvariable war auf dem 5 %-Niveau signifikant und der Attraktivitätskoeffizient hatte ungefähr den gleichen Effekt in der realen Welt. 

Um die Studien zu replizieren, führte ich zunächst Inferenzen auf den Bildern in Google Colab durch (die neuronalen Netzwerke laufen am besten auf GPUs, und mein Laptop hat keinen). Dazu erstellte ich einen Ordner mit zahlreichen Unterordnern, die jeweils eine andere Kombination von Studienbildern enthielten (d. h. Fotos ohne Hintergrund aus Studie 1). Ich schrieb ein [Skript in Python] (https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/inference.py), das die Inferenz für jede Gruppe von Bildern durchführte und die Ergebnisse in einer separaten CSV-File speicherte. Sobald ich alle Daten hatte, fütterte ich sie mit [Skripten in R](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/paper%20replication.Rmd), die sie bereinigten, die Werte standardisierten, sie mit dem ursprünglichen Datensatz der Studie zusammenführten, das Regressionsmodell durchführten, das in der ursprünglichen Studie verwendet wurde, und die Ergebnisse dann in zwei CSV-Filen ausgaben. Die erste zeigte die Ergebnisse der Regression und die zweite die Pearson-Korrelation, den Root Mean Square Error und den Mean Absolute Error zwischen den vom neuronalen Netzwerk generierten Daten und den Originaldaten. 

Leider hatten die Modelle Schwierigkeiten, gute Ergebnisse in der Praxis zu erzielen. 

| Studien | #1 | #2 | #3 | #4 | #5 | #6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Replizierte Versuche (%) | 0 | 0 | 0 | 2.7 | 10.8 | 27.0 |

Nur eine Minderheit der Kombinationen führte in allen Studien zu signifikanten Ergebnissen, und drei Studien zeigten überhaupt keine signifikanten Ergebnisse. Von den Kombinationen, die signifikante Ergebnisse erbrachten, ergaben die meisten Koeffizienten, die mit denen der ursprünglichen Studien übereinstimmten. In diesem Fall definierte ich dies als eine annähernd gleiche Auswirkung in der Praxis. In drei Fällen war dies jedoch nicht der Fall, was bedeutet, dass nur 8,1 % der Modell-Studien-Bild-Kombinationen erfolgreich repliziert werden konnten - ein kaum zu übertreffender Erfolg. 

Ein Großteil der Probleme ist aus der nachstehenden Tabelle ersichtlich. Auf dem Testsatz ihres ursprünglichen Datensatzes schnitten viele dieser Modelle sehr gut ab: Pearson-Korrelationen von etwa 0,9, RMSEs von 0,25-0,35 (die Punktzahlen lagen in der Regel zwischen 1-5). Bei den neuen Bildern waren die Ergebnisse jedoch oft recht schlecht.

| Maßnahme | RMSE | PC | MAE |
| :---: | :---: | :---: | :---: |
| Durchschnitt (über alle Datensätze hinweg) | 1.24 | 0.23 | 1.00 |

Ich habe eine Theorie entwickelt, warum das so ist: Diese Modelle wurden auf sehr rigiden Datensätzen trainiert. Der gebräuchlichste Datensatz, [SCUT-FBP5500,](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), ist recht gut gemacht, aber die Bilder sind einander sehr ähnlich - nach vorne gerichtet, gut beleuchtet, ohne Hintergrund, in etwa gleich beschnitten. Die in diesen Studien verwendeten Bilder sind jedoch viel variabler und naturalistischer. Es gibt Unterschiede in der Beleuchtung, im Bildausschnitt, in der Neigung des Kopfes und in anderen Faktoren. Es gab auch andere potenzielle Unterschiede. Auf den Bildern aus den Studien sind viele ältere Menschen zu sehen, beim SCUT-FBP sind es überwiegend junge Menschen. Die Studien, die ich repliziert habe, wurden in Europa, den USA und Australien durchgeführt; die Bewertungen von SCUT-FBP stammen von Chinesen. In den Studien wurden viele durchschnittlich aussehende Personen verwendet; bei SCUT-FBP waren es unverhältnismäßig viele sehr attraktive Personen. All dies waren potenzielle Störfaktoren, die ich zu vermeiden hoffte. 

### Trainieren meiner eigenen Netzwerke

Um meine eigenen neuronalen Netze zu trainieren, habe ich einen neu erstellten Datensatz ausgewählt: [MEBeauty(https://github.com/fbplab/MEBeauty-database). Frühere Datensätze waren in der Regel eher formal und wiesen nur eine geringe Vielfalt in Bezug auf Alter und ethnische Zugehörigkeit auf. MEBeauty hingegen enthält eine große Vielfalt an Bildern aus der ganzen Welt, auch von älteren Menschen. Die Bilder sind auch natürlicher (unterschiedliche Beleuchtung, unterschiedliche Posen usw.) und wurden auf eine standardisierte Weise erstellt, die leicht zu kopieren war. 

<h3><img align="center" height="300" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/ME3.png"></h3>


Ich [trainierte](https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/training.ipynb) mehrere neue neuronale Netze darauf und wählte die beiden leistungsstärksten aus. Beide waren Versionen des Modells [ComboLoss](https://github.com/lucasxlu/ComboLoss). Sie hatten identische Hyperparameter, mit der Ausnahme, dass eines eine Batch Size von 16 hatte, während das andere eine Batch Size von 32 hatte. 

| Modell | RMSE | MAE | PC |
| :---: | :---: | :---: | :---: |
| ComboLoss (MEBeauty) (BS 16) | 1.07 | 0.82 | 0.64 |
| ComboLoss (MEBeauty) (BS 32) | 1.05 | 0.84 | 0.67 |
| ComboLoss (SCUT-FBP5500) | 0.21 | 0.27 | 0.92 |

Diese Modelle schnitten in ihren Testsätzen schlechter ab als die Modelle, die mit dem SCUT-FBP5500-Datensatz trainiert wurden, was meiner Meinung nach auf die größere Vielfalt der Bilder zurückzuführen ist. Ihre Leistung war ähnlich wie die der Modelle, die von den Entwicklern des MEBeauty-Datensatzes erstellt wurden. 

Sobald ich meine Modelle hatte, wollte ich sie auf zwei Arten testen. Erstens habe ich versucht, die Studien zu wiederholen, wobei ich drei Bewertungen verwendet habe: eine von jedem der Modelle und eine dritte zusammengesetzte Bewertung ihrer Durchschnittswerte für jedes Bild. 

Meine Ergebnisse waren gemischt. Die meisten Studien konnten wiederum nicht wiederholt werden. Außerdem waren die Ergebnisse verwirrend. Die Leistungen der beiden Modelle wichen voneinander ab, obwohl sie bei den Testgruppen extrem ähnlich waren. Außerdem waren es dieses Mal andere Studien, die sich wiederholten. 

<h3><img align="center" height="200" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/mymodel_pvalues.JPG"></h3>

Nach diesen Replikationen habe ich einen weiteren Test meiner neuronalen Netze durchgeführt. Ein durchgängiges Ergebnis in der Literatur ist, dass rechte Politiker etwas attraktiver sind als linke. Niemand weiß warum, aber es wurde in Studien aus Finnland, den USA, Deutschland, Australien und dem Vereinigten Königreich festgestellt. 

Ich habe daher einen Test mit Bildern von Mitgliedern des US-Kongresses von 2011 bis heute durchgeführt und dabei Mitglieder der Demokratischen Partei als linke Politiker und Republikaner als rechte Politiker verwendet. Alle Bilder stammen aus [Voteview's Member Photos Repository](https://github.com/voteview/member_photos). 

In einem ersten Schritt habe ich einen t-Test durchgeführt, der keinen signifikanten Unterschied ergeben hat. Da es jedoch einige wesentliche Unterschiede zwischen den Parteien gibt - republikanische Politiker sind älter und eher weiß und männlich - habe ich auch eine lineare Regression durchgeführt. Als abhängige Variable verwendete ich die Attraktivitätswerte aus meinen Modellen. Als unabhängige und Kontrollvariablen wählte ich die Partei, das Alter, die Kammer (Repräsentantenhaus/Senat) und das Geschlecht. 

<h3><img align="center" height="400" src="https://github.com/vincentium123/Attractiveness-Methods-Evaluation/blob/main/images/stargazer%20congress.JPG"></h3>

Die Ergebnisse der OLS-Modelle stimmen weitgehend mit den früheren Ergebnissen überein. Beide neuronalen Netze stellen fest, dass republikanische Abgeordnete auf dem 10 %-Niveau positiv mit Attraktivität korreliert sind, während der Durchschnitt den Effekt auf dem 5 %-Niveau findet. Dies ist sogar noch aussagekräftiger als auf den ersten Blick - ich habe meine neuronalen Netze dreimal über dieselben Bilder laufen lassen und jedes Mal etwas andere Ergebnisse erhalten. In den beiden anderen Fällen fand das neuronale Netz mit der Batch Size 16 die Korrelation zwischen Republikanern und Attraktivität auf dem 5 %-Niveau. Die Größe des Effekts ist jedoch relativ gering. Die Attraktivität wird in dieser Studie auf einer Skala von 1-10 gemessen, und die Auswirkung eines Republikaners beträgt nur 0,065 - im wirklichen Leben kaum von Bedeutung. Dieser Wert ist geringer als der in anderen Studien festgestellte Effekt.

Die Kontrollvariablen enthalten ebenfalls einige interessante Informationen. Frühere Studien haben keinen Unterschied in der Attraktivität zwischen männlichen und weiblichen Politikern festgestellt, aber in dieser Studie sind männliche Politiker deutlich weniger attraktiv als weibliche Politiker. Anders als bei der Partei ist dieser Effekt ziemlich groß - zwischen 0,6 und 0,8 auf einer 10-Punkte-Skala. Auch für das Alter wurde in einigen Studien mit menschlichen Beurteilern keine signifikante Korrelation mit der Attraktivität festgestellt. In dieser Studie ist dies jedoch der Fall, obwohl der Unterschied recht gering ist. Ein Altersunterschied von dreißig Jahren würde im Durchschnitt nur mit einem Unterschied von etwa 0,3 einhergehen. Schließlich steht die Tatsache, dass Senatoren attraktiver sind als Abgeordnete, im Einklang mit bestehenden Theorien. Da attraktivere Politiker mehr Stimmen erhalten und Senatoren in der Regel mehr (und schwierigere) Wahlen hinter sich haben als Abgeordnete, ist es logisch, dass der Selektionsdruck dazu führen würde, dass Senatoren im Durchschnitt attraktiver sind. Studien aus der Wirtschaft werfen noch einen anderen Blickwinkel auf: Attraktive Gründer können leichter Geldmittel beschaffen, und ein wichtiger Teil der Arbeit eines amerikanischen Politikers ist die Mittelbeschaffung. Bessere Spendensammler werden mit größerer Wahrscheinlichkeit Senatoren.

## Fazit

Bei all diesen Studien wurde eines deutlich: Deep-Learning-Methoden sind, zumindest in ihrem derzeitigen Stadium, nicht genau genug, um die menschliche Attraktivität durchgängig zu beurteilen und in der wissenschaftlichen Forschung eingesetzt zu werden. Manchmal liefern sie genaue Ergebnisse, aber es ist schwierig, im Voraus zu sagen, ob sie das tun werden. In diesen Studien versagten Deep-Learning-Methoden, die bei einem Datensatz extrem gut abschnitten, bei einem anderen völlig, obwohl sie sich deutlich ähneln. Dies gilt sogar für Methoden, die nach dem Training auf Testdatensätzen gut abschnitten. Noch schlimmer ist, dass zwei sehr ähnliche Deep-Learning-Methoden auf einem Testdatensatz ähnliche Ergebnisse erzielen können, die jedoch bei der Anwendung auf reale Fragestellungen voneinander abweichen. Die Blackbox-Natur des Deep Learning macht es fast unmöglich, dies im Voraus zu beurteilen. 

