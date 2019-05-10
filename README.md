# A Convolutional Neural Network to Diagnose Pneumonia from Frontal Chest X-Ray Images

## Abstract

Pneumonia is the leading cause of death among children under 5 years old, and early diagnosis in critical for proper treatment. In this paper, we show that transfer learning applied to large convolutional neural networks can be used to diagnose pneumonia using a Kaggle dataset of 5,858 frontal chest X-ray images [[1]](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from 5,232 unique children. We were able to replicate previous work and build on this by developing a multi-class model that could differentiate viral and bacterial pneumonia

We developed two models, for differentiating between (1) normal and infected images, and (2) normal, bacterial, and viral pneumonia images. Our binary algorithm achieved a best F1 test score of 0.914 and accuracy 88.288\% with three hidden layers, batch size of 64, learning rate of 0.001, and 50 epochs. This performance was within a similar range as previous work [[2]](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). Our multi-classifying algorithm achieved a best F1 test score of 0.636 and accuracy 83.784\% with three hidden layers, batch size of 1000, learning rate of 0.001, and 400 epochs.

We have shown that our model is capable of binary and 3-class classification of chest X-ray images for pneumonia, but more can be done to further improve performance.


## Authors

Alex Kim (agk2144) and Kevin Mao (km3290)
