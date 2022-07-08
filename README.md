# Polyimide/Visualizing-CNN-to-investigate-EMI-Porous-Nanocomposites

data_sem: the folder containing train and test sets.

origin_image: the folder containing the images for visualizing resnet.

save_models: the folder containing the saved model and optimizer after training.

visualizing_model_images: the folder containg the output images from visualizing resnet.

main.py: train the ResNet.

visulizing_model.py: output the images from visualizing resnet.

Note: the output imagesfrom visualizing in manuscript are based on the saved model in the folder 'save_models'. 
      Since the training set is shuffled during training, the obtained models will be slightly different, resulting in different details of the output           visualization pictures, but these do not affect the discussion and results in manuscript.
