---
title: TF
mathjax: Tips
categories:
  - true
tags:
  - true
---

This is all about getting data into the right format for a model. The ins and outs.

*Waffle* 

A year into the Masters of A.I. at Monash and so far we've covered plenty of the foundations; probability theory, statistical/shallow learning methods, and domain specific knowledge in NLP and CV. I've really enjoyed learning about latent variable modelling, unsupervised/clustering methods and how different models find structure in data. Next up is deep learning and the research stream. I figured it makes sense to get to know the framework we'll be using before this happens, so I've been running through the excellent [Tensorflow 2 for Deep Learning](https://www.coursera.org/specializations/tensorflow2-deeplearning) by Imperial College London Specialisation on Coursera. 

Here's 5 important topics we have explored. This is intended as a quick reference for myself and any others who might benefit from it. Training/evaluating Models have been left out for now as there is plenty of information on this elsewhere. The (vital) knowledge I lacked has been in preprocessing, pipelines, and predictions.

  
  
## Topics
1. Data pipelines and preprocessing
2. Dataset objects: Inspecting and manipulating objects
3. Predictions: Making them, interpreting them, handling shapes

## Data pipelines and preprocessing
    

### Topics
1. Numpy arrays
2. Pandas Dataframes and CSV
3. Images

(I'm not going to cover the in-built datasets they haven't caused me much grief.)

#### From Numpy arrays
`tf.data.Dataset.from_tensor_slices(input_tensor)`  
`tf.data.Dataset.from_tensors(input_tensor)`

These methods make tf dataset objects from numpy arrays. They convert the input array into a set of tf.constant op's for the tf graph.

The difference between the two is the expected shape: The first assumes that the input array's first dimension is the batch.

Let's create some dummy data for an example:


```python
data = np.empty(shape=(10,4,2),dtype=int) # 10 examples of shape 4,2
labels = np.empty(shape=(10,1),dtype=int) #

batched_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
batched_dataset.element_spec
```




    (TensorSpec(shape=(4, 2), dtype=tf.int64, name=None),
     TensorSpec(shape=(1,), dtype=tf.int64, name=None))



Using `tf.data.Dataset.from_tensor_slices(input_tensor)` generates a dataset with 10 elements, each element of the shape given above.




```python
dataset = tf.data.Dataset.from_tensors((data, labels))
dataset.element_spec
```




    (TensorSpec(shape=(10, 4, 2), dtype=tf.int64, name=None),
     TensorSpec(shape=(10, 1), dtype=tf.int64, name=None))



Wheras this has generated a dataset containing one training example of the shapes given above.

#### From Pandas DataFrames and csv's
`tf.data.Dataset.from_tensor_slices(input_tensor)`  

We can use this method to convert from a pandas df by first converting the df to a dictionary.


```python
# Create a dummy df
df = pd.DataFrame((['a', 1, 0], 
                   ['a', 5, 1], 
                   ['c', 7, 1],
                   ['b', 2, 0]), columns=['Answer', 'Marks', 'Included'])

print(df)

df_dict = dict(df)

print(df_dict.keys())
```

      Answer  Marks  Included
    0      a      1         0
    1      a      5         1
    2      c      7         1
    3      b      2         0
    dict_keys(['Answer', 'Marks', 'Included'])



```python
df_dataset = tf.data.Dataset.from_tensor_slices(df_dict)
df_dataset.element_spec
```




    {'Answer': TensorSpec(shape=(), dtype=tf.string, name=None),
     'Marks': TensorSpec(shape=(), dtype=tf.int64, name=None),
     'Included': TensorSpec(shape=(), dtype=tf.int64, name=None)}



We can also create them directly from csv's using  
`tf.data.experimental.make_csv_dataset(path, batch_size, label_name)`


```python

```
