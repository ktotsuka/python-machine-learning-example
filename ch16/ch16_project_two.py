import numpy as np
import tensorflow as tf

# Seed the random generator for consistent results
tf.random.set_seed(1)

##### What we are tying to do #####
# Given an input character sequence (40 characters), predict the next character

##### Import a text file #####

# Read a text file
with open('1268-0.txt', 'r') as fp:
    text = fp.read()

# Capture only certain section of the text
start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')
print(start_indx, end_indx)
text = text[start_indx:end_indx]
print('Total Length:', len(text))

##### Encode the text #####

# Get the unique character set from the text
char_set = set(text)
print('Unique Characters:', len(char_set))

# Sort the unique character set
chars_sorted = sorted(char_set)

# Create a dictionary that can convert from a character to an integer
char2int = {ch:i for i,ch in enumerate(chars_sorted)}

# Create an array that can convert from an integer to a character
char_array = np.array(chars_sorted)

# Encode the text
text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32)
print('Text encoded shape: ', text_encoded.shape)
print(text[:15], '     == Encoding ==> ', text_encoded[:15])
print(text_encoded[15:21], ' == Reverse  ==> ', ''.join(char_array[text_encoded[15:21]]))

##### Create a tensor dataset from the encoded text #####

# Create a tensor dataset from the encoded text
ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)

# Print out 5 characters (and the encoded integer)
for ex in ds_text_encoded.take(5):
    print('{} -> {}'.format(ex.numpy(), char_array[ex.numpy()]))

##### Create a dataset of input and output text sequences #####

# Break up the text into chunks of 41 characters
# The first 40 characters will be the input
# The last 40 characters will be the output
seq_length = 40
chunk_size = seq_length + 1
ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)

# Define the function to transform a chunk (41 characters) into input (first 40 characters) and output (last 40 characters) sequences
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq

# Transform the dataset from chunks to input sequence and output sequence
ds_sequences = ds_chunks.map(split_input_target)

# Print out samples of input and output sequences
for example in ds_sequences.take(2):
    print(' Input (x):', repr(''.join(char_array[example[0].numpy()])))
    print('Target (y):', repr(''.join(char_array[example[1].numpy()])))
    print()

##### Build a RNN model #####

# Function for building a RNN model
# Not sure why we are using embedding layer.  It is good for reducing the number of input features, but the input feature is only 80 here...
def build_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(
            rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Set up model parameters
charset_size = len(char_array)
embedding_dim = 256
rnn_units = 512

# Build a model
model = build_model(
    vocab_size = charset_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
model.summary()

##### Compile the model #####

model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy( # loss function for multiclass classification and integer (sparse) labels
        from_logits=True
    ))

##### Train the model #####

# Prepare dataset for training
BATCH_SIZE = 64
BUFFER_SIZE = 10000
ds = ds_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Train the model
model.fit(ds, epochs=10) # this step takes a long time (several minutes)

##### Generate a text using the model #####

# Function to generate a text from a starting string using the model 
def sample(model,
           starting_str, 
           len_generated_text=500, 
           max_input_length=40,
           scale_factor=1.0): # If greater than 1, more predictable.  If less than 1, more random.
    # Encode the starting string and turn it into a tensor
    encoded_input = [char2int[s] for s in starting_str]
    encoded_input = tf.reshape(encoded_input, (1, -1))

    # Initialize the generated string as the starting string
    generated_str = starting_str

    # Reset the model states
    model.reset_states()

    # Predict the next character up to the max number of characters specified
    for i in range(len_generated_text):
        # Get logits (can be converted to probabilities) for the output sequence
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0) # dimention is <number of characters in output sequence> x <number of possible characters>

        # Get the predicted encoded characters (output sequence) based on the probability
        scaled_logits = logits * scale_factor
        new_char_indx = tf.random.categorical(
            scaled_logits, num_samples=1)
        
        # Get the new character, which is the last character of the predicted output sequence
        new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()    

        # Add the new character to the generated string
        generated_str += str(char_array[new_char_indx])
        
        # Update the input sequence
        new_char_indx = tf.expand_dims([new_char_indx], 0) # convert to a tensor
        encoded_input = tf.concat( # add the new character to the encoded input
            [encoded_input, new_char_indx],
            axis=1)
        encoded_input = encoded_input[:, -max_input_length:] # limit the length to 40

    return generated_str

# Generate a text
print(sample(model, starting_str='The island'))

