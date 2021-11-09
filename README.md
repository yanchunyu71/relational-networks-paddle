# Relational-Networks-Paddle

Paddle implementation of Relational Networks - [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)

An introduction to the project on AI studio - [relational-networks](https://aistudio.baidu.com/aistudio/projectdetail/2522451)

Implemented & tested on Sort-of-CLEVR task.

## Introduction

Reference Code:  [relational-networks](https://github.com/kimhc6028/relational-networks)

Paper: [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427v1.pdf)

## Reprod Log
Based on 'reprod_log' model, the following documents are produced.
```
log_reprod
├── forward_paddle.npy
├── forward_torch.npy
├── metric_paddle.npy
├── metric_torch.npy
├── loss_paddle.npy
├── loss_torch.npy
├── bp_align_paddle.npy
├── bp_align_torch.npy
├── train_align_paddle.npy
├── train_align_benchmark.npy
```

Based on 'ReprodDiffHelper' model, the following five log files are produced.

```
├── forward_diff.log
├── metric_diff.log
├── loss_diff.log
├── bp_align_diff.log
├── train_align_diff.log
```
## Dataset

Sort-of-CLEVR is simplified version of [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/).This is composed of 10000 images and 20 questions (10 relational questions and 10 non-relational questions) per each image. 6 colors (red, green, blue, orange, gray, yellow) are assigned to randomly chosen shape (square or circle), and placed in a image.

Non-relational questions are composed of 3 subtypes:

1) Shape of certain colored object
2) Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image
3) Vertical location of certain colored object : whether it is on the upside of the image or downside of the image

Theses questions are "non-relational" because the agent only need to focus on certain object.

Relational questions are composed of 3 subtypes:

1) Shape of the object which is closest to the certain colored object
1) Shape of the object which is furthest to the certain colored object
3) Number of objects which have the same shape with the certain colored object

These questions are "relational" because the agent has to consider the relations between objects.

Questions are encoded into a vector of size of 11 : 6 for one-hot vector for certain color among 6 colors, 2 for one-hot vector of relational/non-relational questions. 3 for one-hot vector of 3 subtypes.

I.e., with the sample image shown, we can generate non-relational questions like:

1) What is the shape of the red object? => Circle (even though it does not really look like "circle"...)
2) Is green object placed on the left side of the image? => yes
3) Is orange object placed on the upside of the image? => no

And relational questions:

1) What is the shape of the object closest to the red object? => square
2) What is the shape of the object furthest to the orange object? => circle
3) How many objects have same shape with the blue object? => 3

## Environment

- Frameworks:
* PaddlePaddle 2.1.2

## Usage

	$ ./run.sh

or

  	$ python sort_of_clevr_generator.py

to generate sort-of-clevr dataset
and

 	 $ python main.py 

to train the binary RN model. 

## Modifications

In the original paper, Sort-of-CLEVR task used different model from CLEVR task. However, because model used CLEVR requires much less time to compute (network is much smaller), this model is used for Sort-of-CLEVR task.

## Result

| | RN_Reference (20th epoch) | RN_paddle |
| --- | --- | --- |
| Non-relational question | 99% | 99% |
| Relational question | 83% | 85% |

## Note

We provide training results of 20-25 epochs. The best accuracy of these epochs is as follows:

20-epoch: Ternary accuracy: 55.34% Binary accuracy: 79.93% | Unary accuracy: 98.59%

21-epoch: Ternary accuracy: 55.24% Binary accuracy: 82.66% | Unary accuracy: 99.24%

22-epoch: Ternary accuracy: 54.69% Binary accuracy: 82.56% | Unary accuracy: 99.44%

23-epoch: Ternary accuracy: 55.44% Binary accuracy: 83.82% | Unary accuracy: 99.40%

24-epoch: Ternary accuracy: 56.00% Binary accuracy: 84.93% | Unary accuracy: 99.50%

25-epoch: Ternary accuracy: 56.80% Binary accuracy: 85.18% | Unary accuracy: 99.34%
