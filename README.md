
## PyTorch FB Scholarship Challenge Flowers Classification
This Project is done using Google Colaboratory.
This dataset contains flower images of 102 categories. 
Training images are 6552, validation images are 818 and test images are 819.
Mount google drive with Colab and place the dataset in zipped format on Google drive to access it. Or alternatively use wget to download the zipped files directly to your current runtime with:
```!wget 'https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip'```

##Checking model's accuracy on test set(This is top-1 accuracy)


```python
def calc_accuracy(model, data):
    model.eval()
    if use_gpu:
      model.cuda()    
    running_corrects = 0
    test_acc = 0
    
    #with torch.no_grad():
    for idx, (inputs, labels) in enumerate(dataloaders[data]):
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        # obtain the outputs from the model
        outputs = model.forward(Variable(inputs))
        # max provides the (maximum probability, max value)
        #_, predicted = outputs.max(dim=1)
        _, predicted = torch.max(outputs.data, 1)
        running_corrects += torch.sum(predicted == labels)
    test_acc = running_corrects / dataset_sizes['test']
    print('Test Accuracy: {:.4f}'.format(test_acc))
    '''# check the 
    if idx == 0:
        print(predicted) #the predicted class
        print(torch.exp(_)) # the predicted probability
    equals = predicted == labels.data
    if idx == 0:
        print(equals)
    print(equals.float().mean())'''
calc_accuracy(model_ft, 'test')
```

    Test Accuracy: 0.9573

#Inference for classification
#Class Prediction
OnceI have got images in the correct format, I have written a function for making predictions with my model. A common practice is to predict the top 5 or so (usually called top- K ) most probable classes. I have calculated the class probabilities then find the  K  largest values.

To get the top  K  largest values in a tensor I have used x.topk(k). This method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes. I have converted these indices to image names using class_names got from cat_to_name.json


```python
def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(Variable(model_input.cuda())))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs, top_labs =top_probs.data, top_labs.data
    top_probs = top_probs.cpu().numpy().tolist()[0] 
    top_labs = top_labs.cpu().numpy().tolist()[0]
    #print(top_labs)
    # Convert indices to classes
    '''idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]'''
    top_flowers = [class_names[lab] for lab in top_labs]
    return top_probs, top_flowers
```
