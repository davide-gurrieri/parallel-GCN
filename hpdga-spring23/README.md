# High-Performance Data & Graph Analytics - Spring 2023

Repository for the High-Performance Data and Graph Analytics contest.

Main deadlines:

- Register for the contest: 2023-05-02 (yy-mm-dd) @ 23:59 CET
- Code: 2023-06-06 (yy-mm-dd) @ 23:59 CET (NO LATE COMMITS!)
- Report: 2023-06-10 (yy-mm-dd) @ 23:59 CET

For further details follow the information on the tracks' slides. 

## Contest instructions

The contest is organized in two main steps:

1. Accelleration
2. Creativity

You are responsible of completing the two steps and properly organize the time at your disposal!

Each step must be a separate section in your report. 


## Step 1 - Acceleration

In this folder you will find a sequential CPU implementation of a Graph Convolutional Network in C++. This implementation follows directly the original model developed by [Kipf et al.](https://arxiv.org/pdf/1609.02907.pdf). If you are interested, [here](http://tkipf.github.io/graph-convolutional-networks/) you can find an additional blog post by the author of the paper detailing the model that is originally implemented with TensorFlow.

Your goal is to increase the performance of the provided code by using a GPU as an hardware accelerator, maintaining an acceptable level of accuracy of the model compared to the baseline implementation.

Avoid the usage of CUDA libraries (e.g., CuBLAS) in your solution, you are free to use them in order to obtain a good guess regarding the obtainable performances.

*Hint:* a good starting point is to accelerate the layers' functions present in the `.\src\module.cpp` file.

You are not required to run the complete GCN on the GPU side, hypotetically you can accelerate only one layer. Indeed, if a choice of this type is made, you must proprely justify it.

### Build and run
You can build and execute the existing implementation by running the following commands:

```sh
make
./exec/gcn-seq cora # dataset name from the dataset folder: [cora, pubmed, citeseer]
```

You will have to modify the `Makefile` in order to compile the code with `nvcc`, as seen during the lectures.

###  Project structure
In the `.\src\` folder you will find the main components of the implementation.
As usual, the core is the `main.cpp` file that parses the selected dataset and creates and object of type `GCN`.
During the initialization phase all the layers for the model are constructed. 
Consequently, the model is then run by calling the function `GCN::run()`.
This function, based on the parameters set by `GCN::get_default()`, will execute a predefined number of epoch during the training phase.

###  How to use Colab
1) From your *private copy of this repository*, open the `colab.ipynb` file in this folder and click on the "open in Colab" button
2) Create a GihHub fine-grained personal access token just for this repository [link](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) (remember to give the right permission to the token in order to use it)
3) Select the `GPU runtime` in Colab [link](https://www.geeksforgeeks.org/how-to-use-google-colab)
4) Change the parameters accordingly with your account ID, clone repository name and token

We suggest editing the original code on your local machine, commit it on GitHub and then load the changes inside of your Colab environment by executing the cell for pulling the remote commits.

## Step 2 - Creativity

Starting from you accelerated version of the GCN provided you are free to do whatever it takes to increase the accuracy of your model. 

Examples:
- Change the layers in the original model
- Explore the hyperparameters space to tune the model
- Increase the complexity of the network
- Change completely the network :D (in this case you still have to provide an accelerated version)