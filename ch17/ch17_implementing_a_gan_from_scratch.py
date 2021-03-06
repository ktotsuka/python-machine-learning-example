import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from ch17lib import make_generator_network
from ch17lib import make_discriminator_network
from ch17lib import preprocess
from ch17lib import create_samples
from ch17lib import Z_SIZE
from ch17lib import IMAGE_SIZE
from ch17lib import BATCH_SIZE

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

##### Make the generator and discriminator models #####

# Set up model parameters
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers = 1
disc_hidden_size = 100

# Make the generator and the discriminator models using the GPU
with tf.device(device_name):
    gen_model = make_generator_network(
        num_hidden_layers=gen_hidden_layers, 
        num_hidden_units=gen_hidden_size,
        num_output_units=np.prod(IMAGE_SIZE))
    gen_model.build(input_shape=(None, Z_SIZE))

    disc_model = make_discriminator_network(
        num_hidden_layers=disc_hidden_layers,
        num_hidden_units=disc_hidden_size)
    disc_model.build(input_shape=(None, np.prod(IMAGE_SIZE)))

##### Train the generator and discriminator models (GAN) #####

# Set up training parameters
num_epochs = 50
mode_z = 'uniform'

# Preprocess the training images
mnist_trainset = mnist['train']
mnist_trainset = mnist_trainset.map(lambda ex: preprocess(ex, mode=mode_z))
mnist_trainset = mnist_trainset.shuffle(10000)
mnist_trainset = mnist_trainset.batch(BATCH_SIZE, drop_remainder=True)

# Train the models
z_for_samples = tf.random.uniform(shape=(BATCH_SIZE, Z_SIZE), minval=-1, maxval=1) # z used for creating sample images at different epoch
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True) # loss function for binary output
g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()
all_losses = []
all_d_outputs = []
epoch_samples = []
start_time = time.time()
for epoch in range(1, num_epochs+1): # for each epoch.  This step takes a long time (~30 min)
    epoch_losses, epoch_d_vals = [], []
    for i,(input_z,input_real) in enumerate(mnist_trainset): # for each batch (of 64 samples) of the training dataset
        # Compute generator's loss
        with tf.GradientTape() as g_tape: # record operations in the tape so the gradient can be computed to adjust the parameters afterward
            g_output = gen_model(input_z) # fake images generated by the generator
            d_logits_fake = disc_model(g_output, training=True) # output of the discriminator using the fake images
            labels_real = tf.ones_like(d_logits_fake) # ideal output of the discriminator (all 1s) using the fake images.
                                                      # this is what the generator is shooting for
            g_loss = loss_fn(y_true=labels_real, y_pred=d_logits_fake) # loss of the generator
            
        # Train the generator
        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables) # compute the gradient of the loss with respect to the trainable parameters
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads, gen_model.trainable_variables)) # update the trainable parameters

        # Compute discriminator's loss
        with tf.GradientTape() as d_tape: # record operations in the tape so the gradient can be computed to adjust the parameters afterward
            d_logits_real = disc_model(input_real, training=True) # output of the discriminator using the real images
            d_labels_real = tf.ones_like(d_logits_real) # ideal output of the discriminator (all 1s) using the real images.
                                                        # this is what the discriminator is shooting for
            d_loss_real = loss_fn(y_true=d_labels_real, y_pred=d_logits_real) # loss of the discriminator for real images

            d_logits_fake = disc_model(g_output, training=True) # output of the discriminator using the fake images
            d_labels_fake = tf.zeros_like(d_logits_fake) # ideal output of the discriminator (all 0s) using the fake images.
                                                         # this is what the discriminator is shooting for
            d_loss_fake = loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake) # loss of the discriminator for fake images

            d_loss = d_loss_real + d_loss_fake # total loss of the discriminator

        # Train the discriminator
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables) # compute the gradient of the loss with respect to the trainable parameters
        d_optimizer.apply_gradients(grads_and_vars=zip(d_grads, disc_model.trainable_variables)) # update the trainable parameters
                           
        # Append losses for this batch
        epoch_losses.append((g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy()))
        
        # Append the output of the discriminator for this batch
        d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real)) # mean prediction for 64 real images as probability
        d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake)) # mean prediction for 64 fake images as probability
        epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))   

    # Append the losses and the output of the discriminator for this epoch     
    all_losses.append(epoch_losses) # <number of epochs> x <iteration (training dataset size / batch size)>
    all_d_outputs.append(epoch_d_vals) # <number of epochs> x <iteration (training dataset size / batch size)>

    # Print out information for the epoch that has just finished
    print(
        'Epoch {:03d} | ET {:.2f} min | Avg Losses >>'
        ' G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake: {:.4f}]'
        .format(
            epoch, (time.time() - start_time)/60, 
            *list(np.mean(all_losses[-1], axis=0))))

    # Generate some sample fake images using the generator for this epoch
    epoch_samples.append(create_samples(gen_model, z_for_samples).numpy())

##### Plot the training results (losses and the discriminator outputs) #####

# Create a figure
fig = plt.figure(figsize=(16, 6)) # figure size in inches

# Get epoch tick positions
epoch_ticks = [1, 10, 20, 30, 40, 50] # epoch numbers where a tick mark is desired
epoch2iter = lambda e: e*len(all_losses[0]) # function to multiply by the iteration size (training dataset size / batch size)
epoch_tick_positions = [epoch2iter(e) for e in epoch_ticks] # location where epock ticks should be placed

# Plot the losses
ax = fig.add_subplot(1, 2, 1)
g_losses = [item[0] for item in itertools.chain(*all_losses)] # item[0] contains the generator loss
                                                              # itertools.chain iterate for batches and epochs
d_losses = [item[1]/2.0 for item in itertools.chain(*all_losses)] # item[1] contains the discriminator loss. 
                                                                  # Divide by 2 to get the average loss of the real and fake images
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

# Plot the outputs of the discriminator
ax = fig.add_subplot(1, 2, 2)
d_outputs_real = [item[0] for item in itertools.chain(*all_d_outputs)]
d_outputs_fake = [item[1] for item in itertools.chain(*all_d_outputs)]
plt.plot(d_outputs_real, alpha=0.75, label=r'Real: $D(\mathbf{x})$')
plt.plot(d_outputs_fake, alpha=0.75, label=r'Fake: $D(G(\mathbf{z}))$')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Discriminator output', size=15)

# Add epoch tick marks
ax2 = ax.twiny()
ax2.set_xticks(epoch_tick_positions)
ax2.set_xticklabels(epoch_ticks)
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax2.set_xlabel('Epoch', size=15)
ax2.set_xlim(ax.get_xlim())

plt.show(block=True)

##### Plot the generated images #####

selected_epochs = [1, 2, 4, 10, 50]
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
