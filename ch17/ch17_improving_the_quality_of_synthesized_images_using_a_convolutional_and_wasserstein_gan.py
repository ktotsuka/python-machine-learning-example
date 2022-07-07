import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from ch17lib2 import make_dcgan_generator
from ch17lib2 import make_dcgan_discriminator
from ch17lib2 import preprocess
from ch17lib2 import Z_SIZE
from ch17lib2 import IMAGE_SIZE
from ch17lib2 import BATCH_SIZE
from ch17lib2 import create_samples

##### Seed random generator for consistent results #####

tf.random.set_seed(1)
np.random.seed(1)

##### Prepare the GPU usage #####

print("GPU Available:", tf.test.is_gpu_available())
if tf.test.is_gpu_available():
    device_name = tf.test.gpu_device_name()

else:
    device_name = 'cpu:0'    
print(device_name)

##### Fetch the MNIST hand-written digit dataset #####

# Fetch the MNIST hand-written digit dataset
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
mnist = mnist_bldr.as_dataset()

# Get training dataset for MINST
# It contains images of size 28 x 28
mnist_train_set = mnist['train']

##### Make the generator and discriminator models (DCGAN) #####

# Make the generator and the discriminator models using the GPU
with tf.device(device_name):
    gen_model = make_dcgan_generator()
    gen_model.build(input_shape=(None, Z_SIZE))
    gen_model.summary()

    disc_model = make_dcgan_discriminator()
    disc_model.build(input_shape=(None, np.prod(IMAGE_SIZE)))
    disc_model.summary()

##### Train the generator and discriminator models #####

# Set up training parameters
num_epochs = 50
mode_z = 'uniform'
lambda_gp = 10.0

# Preprocess the training images
mnist_trainset = mnist['train']
mnist_trainset = mnist_trainset.map(lambda ex: preprocess(ex, mode=mode_z))
mnist_trainset = mnist_trainset.shuffle(10000)
mnist_trainset = mnist_trainset.batch(BATCH_SIZE, drop_remainder=True)

# Train the models using WGAN-GP
z_for_samples = tf.random.uniform(shape=(BATCH_SIZE, Z_SIZE), minval=-1, maxval=1) # z used for creating sample images at different epoch
g_optimizer = tf.keras.optimizers.Adam(0.0002)
d_optimizer = tf.keras.optimizers.Adam(0.0002)
all_losses = []
epoch_samples = []
start_time = time.time()
for epoch in range(1, num_epochs+1): # for each epoch.  This step takes a long time (~60 min)
    epoch_losses = []
    for i,(input_z,input_real) in enumerate(mnist_trainset):
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape: # record operations in the tape so the gradient can be computed
                                                                       # to adjust the parameters afterward
            # Compute outputs of the generator and the discriminator
            g_output = gen_model(input_z, training=True) # fake images generated by the generator            
            d_logits_real = disc_model(input_real, training=True) # output of the discriminator using the real images
            d_logits_fake = disc_model(g_output, training=True) # output of the discriminator using the fake images

            # Compute generator's loss (p662)
            g_loss = -tf.math.reduce_mean(d_logits_fake)

            ## Compute discriminator's losses using gradient penalty (p662)
            d_loss_real = -tf.math.reduce_mean(d_logits_real)
            d_loss_fake =  tf.math.reduce_mean(d_logits_fake)
            d_loss = d_loss_real + d_loss_fake
            
            with tf.GradientTape() as gp_tape:
                alpha = tf.random.uniform(
                    shape=[d_logits_real.shape[0], 1, 1, 1], 
                    minval=0.0, maxval=1.0)
                interpolated = (alpha*input_real + (1-alpha)*g_output)
                gp_tape.watch(interpolated)
                d_logits_interpolated = disc_model(interpolated)
            
            grads_interpolated = gp_tape.gradient(
                d_logits_interpolated, [interpolated,])[0]
            grads_interpolated_l2 = tf.sqrt(
                tf.reduce_sum(tf.square(grads_interpolated), axis=[1, 2, 3]))
            grad_penalty = tf.reduce_mean(tf.square(grads_interpolated_l2 - 1.0))
        
            d_loss = d_loss + lambda_gp*grad_penalty
        
        # Train the generator
        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        g_optimizer.apply_gradients(
            grads_and_vars=zip(g_grads, gen_model.trainable_variables))

        # Train the discriminator
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optimizer.apply_gradients(
            grads_and_vars=zip(d_grads, disc_model.trainable_variables))
        
        # Append losses for this batch
        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(), 
             d_loss_real.numpy(), d_loss_fake.numpy()))
                    
    # Append the losses for this epoch     
    all_losses.append(epoch_losses)
    
    # Print out information for the epoch that has just finished
    print('Epoch {:-3d} | ET {:.2f} min | Avg Losses >>'
          ' G/D {:6.2f}/{:6.2f} [D-Real: {:6.2f} D-Fake: {:6.2f}]'
          .format(epoch, (time.time() - start_time)/60, 
                  *list(np.mean(all_losses[-1], axis=0)))
    )
    
    # Generate some sample fake images using the generator for this epoch
    epoch_samples.append(create_samples(gen_model, z_for_samples).numpy())

##### Plot the training results (losses and the discriminator outputs) #####

# Create a figure
fig = plt.figure(figsize=(8, 6)) # figure size in inches

# Get epoch tick positions
epoch_ticks = [1, 10, 20, 30, 40, 50] # epoch numbers where a tick mark is desired
epoch2iter = lambda e: e*len(all_losses[0]) # function to multiply by the iteration size (training dataset size / batch size)
epoch_tick_positions = [epoch2iter(e) for e in epoch_ticks] # location where epock ticks should be placed

# Plot the losses
ax = fig.add_subplot(1, 1, 1)
g_losses = [item[0] for item in itertools.chain(*all_losses)] # item[0] contains the generator loss
                                                              # itertools.chain iterate for batches and epochs
d_losses = [item[1] for item in itertools.chain(*all_losses)] # item[1] contains the discriminator loss
plt.plot(g_losses, label='Generator loss', alpha=0.95) # alpha: transparency.  0 = transparent ~ 1 = opaque
plt.plot(d_losses, label='Discriminator loss', alpha=0.95)
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15) # iteration contains both batches and epochs
ax.set_ylabel('Loss', size=15)

# Add epoch tick marks (you may have to maximize the plot to see them)
ax2 = ax.twiny() # create a secondary axis sharing the y axis
ax2.set_xticks(epoch_tick_positions) # draw epoch ticks
ax2.set_xticklabels(epoch_ticks) # draw epoch tick labels
ax2.xaxis.set_ticks_position('top') # set epoch ticks to be above the plot
ax2.xaxis.set_label_position('top') # set epoch tick labels to be above the plot
ax2.set_xlabel('Epoch', size=15)
ax2.set_xlim(ax.get_xlim()) # set the limit for the secondary axis to be the same as the first axis for x

plt.show(block=True)

##### Plot the generated images #####

#selected_epochs = [1, 2, 4, 10, 50]
selected_epochs = [1, 2, 3]
num_samples_per_epoch = 5
fig = plt.figure(figsize=(10, 14)) # size in inches
for i,e in enumerate(selected_epochs):
    for j in range(num_samples_per_epoch):
        ax = fig.add_subplot(len(selected_epochs), num_samples_per_epoch, i*num_samples_per_epoch+j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            # Add text for each epoch to the left
            ax.text(
                -0.06, 0.5, 'Epoch {}'.format(e), # x position (-6%, so left of the plot), y position (50%, so middle of the plot), text
                rotation=90, size=18, color='red',
                horizontalalignment='right',
                verticalalignment='center', 
                transform=ax.transAxes) # set text position coordinate so that (0,0) is the left bottom and (1,1) is the right top of the plot
        
        image = epoch_samples[e-1][j]
        ax.imshow(image, cmap='gray_r') # use color map = grayscale reversed (black and white are reversed)
    
plt.show(block=True)
