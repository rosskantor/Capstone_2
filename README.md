###Sign Language Detection

The ability of a visual detection system to recognize and translate sign language would be a major breakthrough. This system would allow deaf people with limited ability to speak to more easily communicate with individuals with limited understanding of sign language.

###Data

The data used for this model was provided by Kaggle.  Each letter of the alphabet is represented by 3000 training images of similar size, orientation, and resolution (200 X 200).  Surprisingly, the test dataset consists of a single image per letter.

The single biggest limitation to a swift solution appears to be computing power.  Model training frequently taxed the processor to 300%.

![Processor](Processor.png)

https://www.kaggle.com/grassknoted/asl-alphabet.

###Training

Image detection was the theme I wanted to pursue.  The predictive model used, Convolutional Neural Networks, was chosen because they tend to do well on image detection projects.  

Most models were trained at an image resolution of 64 X 64 or 96 X 96.  An image size greater than 96 takes a prohibitively long time (2+ hours per epoch) to train.  Visually, 96 X 96 is the lowest resolution still detectable to the human eye.

The final model utilized a images at 128 by 128, 3 by 3 filters and a slide of 1.  The model consisted of one input layer, four hidden layers and an output layer.  Hyperbolic tangent was used for the first two layers, relu for the next 3 and softmax for categorical prediction.

After numerous training iterations CNN models appeared to crater at around 60 to 65% accuracy.  The two charts detailed below indicate a model stopping after four epochs because validation loss increased.

![128_Accuracy](128_Accuracy.png)

![128_Loss](Img Size 128 Loss.png)

### The Experiment

Can I improve model results by eliminating all background colors.
![Letter Y at 128](Y128.png)

Below is a distribution of color intensities.  Perhaps if I can black out all non-skin color the model will improve.
![Y 128 Distribution](Y_128_Dist.png)

Distribution of color intensities between 25 and 60.
![Y 128 Between 25 and 60](Y128_25_60.png)

Distribution of color intensities between 80 and 105.
![Y 128 Between 80 and 105](Y_128_80_105.png)

Distribution of color intensities between 170 and 195.
![Y 128 Between 170 and 195](Y_128_170_195.png)

I created a second model where all pictures were altered.  Pixel intensity greater than 85 was set to zero in an effort to remove noise.  Altered images were run through the same model as non altered images.  Unfortunately, the model terminated early because of thermal issues.

![Altered Picture](A371.png)

### Results

What was once a promising model erodes when predicting on the test set proved to be very disappointing when fed the fest set.  Two letters, P and C, were predicted 97 percent of the time.

![Letter Distributions](Letter_Dist.png)

### P

![Letter P](P3.png)

### C

![Letter C](C111.png)

### Lessons Learned

* Predict on holdout set sooner.
* I must have made an error.

### Future Work

* I may have to abandon this initiative.
