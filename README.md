# Enhancing Face Recognition with Latent Space Data Augmentation and Facial Posture Reconstruction

Face Representation Augmentation (FRA) @ Expert Systems with Applications 2023 [Arxiv](https://arxiv.org/abs/2301.11986). 

## Objective of FRA

Since data scarcity is a common problem in deep learning-based solutions, it can be very challenging to build up FR systems that are robust enough to recognize face images with extreme diversity. In this paper, we proposed a method that augments the face data in latent space. The proposed method utilizes two major components, one of which is an autoencoder, and the other is a ViT-based model. The former encodes the binarized input images consisting of sparse facial landmarks into a latent space. The latter is used for extracting features from the combined embeddings from a pre-trained FRL approach and the autoencoder part of our model. Lastly, the output of the proposed model is an embedding representing the main identity with the same emotion but with a different posture. This way, we improved the classification accuracy by 9.52, 10.04, and 16.60, in comparison with the based models of MagFace, ArcFace, and CosFace, respectively.

![overview](https://github.com/soroushhashemifar/Augmenting-Face-Representation/assets/24815283/56cbab25-5759-4988-8747-fd1bfc951aa9)

## Requirements
This code is implemented in PyTorch, and we have tested the code on CodeOcean platform. All the required libraries are listed in a dockerfile inside `/environment`.

## A Quick Start - How to use it

For a detailed introduction, please refer to file `/code/run`, which is a bash file and contains all the required steps to run FRA.

#### Step 1: Generate the required dataset from KDEF data
This command will generate 10000 pairs of images needed to train FRA: 

```
python step1_generate_dataset.py --embedding-source magface --num-samples 10000 --data-path /data --results-path /results --checkpoint-path /data/magface_epoch_00025.pth
```

The original dataset is read from `/data` and the generated pairs are written to `/results` directories. Currently, only thee FRL (face representation learning) algorithms are accessible for test: MagFace, CosFace, and ArcFace.

#### Step 2: Train FRA

Then, you need to train FRA using the following command for the required configuration (num of epochs, batch size, learning rate, and dropout rate).

```
python step2_train_FRA.py --embedding-source magface --num-epochs 3 --batch-size 256 --learning-rate 0.001 --dropout-rate 0.4 --results-path /results
```

#### Step 3: Evaluation

Finally, you can evaluate the saved checkpoint using the following command to generate the plots and figures needed to asses the quality of the checkpoint on validation set.

```
python prune_neuron_cifar.py --output-dir './save' --mask-file './save/mask_values.txt' --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th'
```

These saved figures include: ROC and precision-recall curves along with confusion matrices

## Citing this work

If you use our code, please consider cite our work

```bibtex

```

If there is any problem, be free to open an issue or contact: hashemifar.soroush@gmail.com.
