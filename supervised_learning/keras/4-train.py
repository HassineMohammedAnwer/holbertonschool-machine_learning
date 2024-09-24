




def train_model(network, data, labels, batch_size, epochs,
        verbose=True, shuffle=False):
    history = network.fit(
        data,              # Input data
        labels,            # Target data (labels)
        batch_size=batch_size, # Size of each mini-batch
        epochs=epochs,         # Number of epochs to train
        verbose=verbose,       # Print progress of training
        shuffle=shuffle        # Shuffle training data at each epoch
    )
    
    return history