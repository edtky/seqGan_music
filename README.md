# seqGAN for Music Generation
A PyTorch implementation of "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient." (Yu, Lantao, et al.). 

Here we treat music generation as a language modeling problem and use SeqGAN model trained on encoded classical music by Mozart to generate new music.

This project builds on the seqGAN implementation in https://github.com/suragnair/seqGAN that implements policy gradients much simpler than the original work and do not invole rollouts. 

The architectures used are different than those in the orignal work. Specifically, a recurrent bidirectional GRU network is used as the discriminator. 

The code performs the experiment on synthetic data as described in the paper.

To run the code:
```bash 
python main.py
```
main.py should be your entry point into the code.

## Key observations and modifications
- Instead of using an oracle which represents the true distribution, midi files of Mozart music are encoded into text as the data sample representing the true distribution
- Vocabulary size substantially decreased from 5000 to about 149
- Allowing the generator to create samples of flexible length
- Training Discriminator a lot more than Generator (Generator is trained only for one batch of examples, and increasing the batch size hurts stability)
- Using Adam for Generator and Adagrad for Discriminator
- Tweaking learning rate for Generator in GAN phase
- Using dropout in both training and testing phase

- Stablity is extremely sensitive to almost every parameter :/
- The GAN phase may not always lead to massive drops in NLL (sometimes very minimal) - I suspect this is due to the very crude nature of the policy gradients implemented (without rollouts).
