## Introduction

The top-performing solutions of the challenge surpassed all previous benchmarks trained on similar data. However, the models varied largely in their data handling, preprocessing, loss calculation, and combination of different NN layers (convolution, recurrent, attention, etc.). We plan to identify the key components contributing to these models' performance differences.

To this end, we will create a prix fixe menu of NN blocks from the random promoter DREAM challenge 2022 models. Afterward, we will create models with all possible combinations of the blocks from the Prix Frix menu and train with the same train-validation set. We will use different trainer functions for each model to accommodate the different loss functions used. Our goal is to make a fair comparison of the NN architectures and other components that went into training them. For example, maybe BHI’s solution could be improved by borrowing some components from autosome’s solution. This way, we will ensure we provide some solid takeaways from the DREAM flagship paper. We have discussed the plan with Google TRC, and they have generously agreed to support these experiments with TPU resources.

## To participate

At first, each team has to break its model into multiple blocks so that we can add them to the Prix Fixe menu. Each team should be able to combine their blocks with other models' blocks. 
For this reason, we will use the same library for writing these blocks and maintain a similar flow of information in the blocks. 

As most of the models were written in Pytorch, we will go forward with Pytorch. Each team needs to break down its model into the following blocks:

- **firstLayersBlock**: First layers in the model
- **coreBlock**: the "guts" of the model
- **finalLayersBlock**: The final layers that eventually lead to the prediction
 
Regarding the flow of information, we will use a 3-d tensor (batch, channel, seqLen). And if your model input had a 4-d tensor (batch,channel,seqLen,1) to use conv2d, you will need to change your flow of information in the blocks you provide. If your network needs 4-d tensors for valid reasons, please contact me to discuss how to include your model in the Prix Fixe runs.

The first few layers you use before you go into the core architecture can be written as the firstLayersBlock of your network. While writing the code for this block, remember that you are taking (batch, channel, seqLen) as input and your output is of similar dimension (batch, channel, seqLen). It is okay if you are reducing the length of seqLen in your model, as long as it is in the last axis. This way any coreBlock can plug any firstLayersBlock before itself. The firstLayersBlock can contain your first few convolutional layers, patching of the input, etc. If you do not consider it to be a part of your core architecture, and you think other networks should use it before applying their core architecture, it goes into the firstLayersBlock. 

After the firstLayersBlock, you will have your coreBlock which will contain the core layers of your network (efficientNetBlocks, ResNet, RNNs, transformers, etc.). The output from this block will be of (batch, channel, seqLen) shape. Again, it is okay if you are reducing the length of seqLen in your model; as long as it is in the last axis. This way any finalLayersBlock will be able to plug any coreBlock before itself.

Then you will add your finalLayersBlocks block, leading to the expression prediction. This can be your final pooling layers followed by MLP blocks or just the final fully connected layers. If you do not consider anything to be a part of your core architecture, and/or you think other networks should use it to make prediction from their embedding space, it goes into the finalLayersBlocks.

It should be noted that if a team has any way of integrating auxiliary losses in their network, we will test them at the end. So you will exclude them from the blocks you write for now. We will not optimize architectures for these losses because we can only test so many things.

![](https://github.com/muntakimrafi/random-promoter-DREAM2022/blob/main/prix%20fixe.png)

## First stage

You have to add a script {your team name}_model.py for example, autosome_model.py that will contain all the blocks of your model. For reference, please look at [ResNet_model.py](https://github.com/muntakimrafi/random-promoter-DREAM2022/blob/main/models/ResNet_model.py) script. 
Please comment on all the operations in your block and use a docstring that explains the operation happening inside the block. Also, mention the exact values of the input and output dimensions of the block in your own model’s graph.
After you define all the blocks, call these blocks from {your team name}_model.py and add them to the layer lists of [train](https://github.com/muntakimrafi/random-promoter-DREAM2022/blob/main/train.ipynb) notebook. Print the layers of your model to check whether you have successfully recreated your model graph. Then train your model using the same notebook to verify whether the blocks in {your team name}_model.py are written correctly (download one hot encoded train and validation data from [here](https://drive.google.com/drive/folders/1xaIhWpD9Zwp09VKy-NdWj7wziaOYO0OB?usp=sharing)). We currently have a sample trainer function there. If your model cannot use this trainer function due to the finalLayersBlock performing a classification task or any other reason, please use an appropriate trainer function and add it to the [utils.py](https://github.com/muntakimrafi/random-promoter-DREAM2022/blob/main/utils.py) script) so that we can use this to train models when we stitch your finalLayersBlock with other teams’ blocks.

Afterward, please check whether you can integrate the provided ResNet layers with your model layers using the [train](https://github.com/muntakimrafi/random-promoter-DREAM2022/blob/main/train.ipynb) notebook. You will later follow this procedure to add layers from other teams {team_name}_model.py codes.

If you have an encoding method other than the 4-d onehot encoding, you must create train and validation data (90-10 split) and share it with us (and add the code to [utils.py](https://github.com/muntakimrafi/random-promoter-DREAM2022/blob/main/utils.py) script). We will later use the code from [utils.py](https://github.com/muntakimrafi/random-promoter-DREAM2022/blob/main/utils.py) script) to generate train and validation data in a way that the sequences that go into the train and validation set will be the same for all encoding methods.

If you have any questions, please reach out to me at abdulmuntakim(dot)rafi(at)ubc(dot)ca

## Second stage

We run all possible combinations from the prix fixe menu at this stage using the TRC quota we receive. After we finish the first stage, I will update this section.

## Third stage

Afterward, we will benchmark the improved DREAM models on some human genomics sequence data (both with and without transfer learning) to show how these models can generalize across different genomics sequence datasets. I will update this section when we are in the second stage.
