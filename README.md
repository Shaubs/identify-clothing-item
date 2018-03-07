# identify-clothing-item
The program takes image inputs and identifies if it is a shirt or a pant

As a college student who has no access to a washing machine, doing laundry seems like the biggest task we achieve that day. If that wasn't enough of a burden, hanging them out to dry and turning that pile of dry clothes into neat stacks of folded articles become another pain in the neck. As Sheldon from Big Bang Theory showed us the folding machine, it still required you to take out a shirt or a pant and have the machine fold it accordingly. I'm lazier than Sheldon. I just want to dump the pile of clothes on the machine and let it fold it for me. Hence, I decided to come up with a code that distinguishes between different kinds of clothing and assorts them accordingly. Feel free to check it out and make your lives easier! 

Here there are two ways of classifying.
1) Extacting image parameters and loading it into a Naive Bayes classifier
          Extracting_parameters_from_image.py : use this to get the parameters
          Classifying_type_of_clothing.py :load the parameters to classsify
          
          
2) Using CNN trained model on Region of interest
          training_cnn_for_shirt_pant.py : this will create model "pant-shirt-model.h5" which will be run on the region of interest for                                              classification
          pant-shirt-classify-with-cnn.py : classify image
