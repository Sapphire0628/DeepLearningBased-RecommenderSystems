# Deep learning-based Recommender systems Using Amazon Reviews

## Description

Deep learning based Recommender systems Using Amazon Reviews

This project involves using a large dataset containing customer reviews and product information from Amazon to construct a recommender system. The aim is to leverage deep learning techniques to predict consumer preferences based on their consumption habits and product metadata. 

In this project, a hybrid recommendation model is implemented, combining a neural matrix factorization model with content-based models. This hybrid model integrates user-item interactions and item attributes (product image and product review) to predict a user's preference for a particular Amazon fashion product based on past interactions with other products, following the approach proposed by Zhang et al. (2017) for joint representation learning. The neural matrix factorization model's architecture is based on the framework proposed by Chakrabarti and Das (2019). 

## Data Source

[Amazon review data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)


## Overview of the Context-Aware Hybrid model

<img src='.overview.png' width='700'>

## Architecture of a Hybrid model
```
def hybirdModel(user_ids, unique_product_ids,):
    # Setting the size of embeddings
    embeddings_size = 32
    # Getting the number of unique users and products
    usr, prd = user_ids.shape[0], unique_product_ids.shape[0]

    # Defining input layers
    x_users_in = Input(name="User_input", shape=(1,))
    x_products_in = Input(name="Product_input", shape=(1,))
    
    # A) Matrix Factorization
    ## Embeddings and Reshape layers for user ids
    cf_xusers_emb = Embedding(name="MF_User_Embedding", input_dim=usr, output_dim=embeddings_size)(x_users_in)
    cf_xusers = Reshape(name='MF_User_Reshape', target_shape=(embeddings_size,))(cf_xusers_emb)
    ## Embeddings and Reshape layers for product ids
    cf_xproducts_emb = Embedding(name="MF_Product_Embedding", input_dim=prd, output_dim=embeddings_size)(x_products_in)
    cf_xproducts = Reshape(name='MF_Product_Reshape', target_shape=(embeddings_size,))(cf_xproducts_emb)
    ## Dot product layer
    cf_xx = Dot(name='MF_Dot', normalize=True, axes=1)([cf_xusers, cf_xproducts])

    # B) Neural Network
    ## Embeddings and Reshape layers for user ids
    nn_xusers_emb = Embedding(name="NN_User_Embedding", input_dim=usr, output_dim=embeddings_size)(x_users_in)
    nn_xusers = Reshape(name='NN_User_Reshape', target_shape=(embeddings_size,))(nn_xusers_emb)
    ## Embeddings and Reshape layers for product ids
    nn_xproducts_emb = Embedding(name="NN_Product_Embedding", input_dim=prd, output_dim=embeddings_size)(x_products_in)
    nn_xproducts = Reshape(name='NN_Product_Reshape', target_shape=(embeddings_size,))(nn_xproducts_emb)
    ## Concatenate and dense layers
    nn_xx = Concatenate()([nn_xusers, nn_xproducts])
    nn_xx = Dense(name="NN_layer", units=16, activation='relu')(nn_xx)
    nn_xx = Dropout(0.1)(nn_xx)
    ######################### TEXT BASED ############################
    text_in = Input(name="title_input", shape=(64,))
    text_x = Dense(name="title_layer", units=64, activation='relu')(text_in)

    ######################## IMAGE BASED ###########################
    image_in = Input(name="image_input", shape=(512,))
    image_x = Dense(name="image_layer", units=256, activation='relu')(image_in)
       
    content_xx = Concatenate()([text_x, image_x])
    content_xx = Dense(name="contect_layer", units=128, activation='relu')(content_xx)
    # Merge all
    y_out = Concatenate()([cf_xx, nn_xx, content_xx])

    y_out = Dense(name="CF_contect_layer", units=64, activation='linear')(y_out)
    y_out = Dense(name="y_output", units=1, activation='linear')(y_out)
    model = Model(inputs=[x_users_in,x_products_in, text_in, image_in], outputs=y_out, name="Hybrid_Model")
    adam = optimizers.Adam(lr=0.01, decay = 0.0001)
    model.compile(optimizer=adam, loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.mean_absolute_error])
    
    return model
```

<img src='./Architecture.png' width='700'>

I) Matrix Factorization Component (Black)
Matrix Factorization Componen uses matrix factorization to learn user and product embeddings. The embeddings are then reshaped and fed into a dot product layer, which predicts the user's preference for a particular product. The dot product layer is normalized to improve the model's performance.

II) Neural Network Component (Red)
This component uses neural networks to learn additional representations of users and products. The embeddings are again reshaped and concatenated before being fed into a dense layer with 16 units and a ReLU activation function. A dropout layer is added to prevent overfitting.

III) Text-Based Component (Yellow)
The model includes a text-based component that inputs the product's title. The text is fed into a dense layer with 64 units and a ReLU activation function.

IV) Image-Based Component (Blue)
The model includes an image-based component that inputs the product's image. The image is flattened into a 1D vector and fed into a dense layer with 256 units and a ReLU activation function.

## Results and Observation

We employ the Pearson correlation coefficient, RMSE, and MAE to evaluate the hybrid recommendation model's effectiveness.

### Pearson correlation coefficient

Pearson correlation coefficient takes into account the variability of the data. It measures the linear correlation between the predicted and actual ratings. 

<img src='./Pearson_correlation.png' width='400'>

After calculating the Pearson correlation coefficient, we obtained a 0.82009 correlation coefficient. It indicated a strong positive correlation between predicted and real ratings.

### RMSE & MAE

RMSE and MAE provide an intuitive measure of the model's performance in predicting user preferences.
RMSE measures the average squared difference between the predicted and actual ratings.
MAE measures the average absolute difference between the predicted and actual ratings. 

After model training and prediction, we obtained that the RMSE is 0.75259 and the MAE is 0.51566. The Test RMSE of 0.75259 and Test MAE of 0.51566 indicate that the hybrid model performs reasonably well in predicting user preferences for items in the test set. 

## Convergence performance

The hybrid model converges at around 10 epochs, which means that the model has learned to make accurate predictions on the training data after approximately five passes through the training set. They indicate that the model is learning quickly and efficiently.

<img src='./Covergence_preformance.png' width='400'>

## Full Report
See documentation [here](./Project_report.pdf)





