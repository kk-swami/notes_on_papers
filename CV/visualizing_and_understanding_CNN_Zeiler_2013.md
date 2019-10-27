# Citation  

Visualizing and Understanding Convolutional Networks
Zeiler and Fergus  ACL 2009  

# Tags  

CNN, visualization 

# Significance

A method to visualize how exactly a CNN works. More precisely, at every layer, maps activations of layer back to the original input space
to get a sense of which pixels of the input highlighted that particular layer the most


# Context and summary  

At the time of this paper, Krizhevsky etal's architecture (2012) using CNN's had just come up,  and broke existing records on classification in the Imagenet 2012 dataset. 
But there was little insight into why imagenet worked so well (most previous works tried to understand only the first few layers). In this context, this paper came up with a visualization technique based on deconvnet
to project feature activations back to input space. In addition, the next part of the experiment occludes portions of input image,
revealing which parts are important for classification. Based on insights from this analysis; they modify the architecture of Krizhevsky a bit to get even better performance

# What is a deconvnet ?  

A deconvnet, more formally known as a transpose convulution is kind of the inverse of a convolution operation .
For example, for simplicity's sake, assume a normal convolution operation of a 4*4 matrix with a 3*3 filter, with no padding, and stride = 1
This will yield an output of size 2*2.
If we represent this as a matrix operation, where the input is unrolled into a 16*1 vector, and the output is a 4*1 vector;
the the convolution operation can be captured by a matrix C of dimension 4*16  
ie O<sup>4*1</sup> = C<sup>4*16</sup>*I<sup>16*1</sup>  
If we want to go down the other way during transpose convolution (project output activations in input space) -  
I<sup>'</sup><sup>16*1</sup> = C<sup>T</sup><sup>16*4</sup>*O<sup>'</sup><sup>4*1</sup>  
In otherwords, the transpose of the same set of weighs obtained during a forward convolution operation can be used to perform the transpose convolution operation
(Note - transpose convolution can be performed as a standard direct convolution, by padding a lot more rows and zeros to O<sup>'</sup>
For more details see [link](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)




# Method in more detail  

1) At every intermediate layer j, activations are mapped back to the original input space using a deconvnet. Every cell in any feature map, can be mapped back to a region of the original input space this way  
2) More precisely, while mapping a particular cell/activation a<sub>ik</sub> in a feature map back to input space, in any layer k, all other activations except for a<sub>ik</sub>are set to 0
3) Now , the activation a
        
        

# Reference
[convolutional arithmetic (Dumoulin and Visin 2016)](https://arxiv.org/abs/1603.07285)
