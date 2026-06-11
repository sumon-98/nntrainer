import urllib.request
import gzip
import os
import struct

def download_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        'images': 'train-images-idx3-ubyte.gz',
        'labels': 'train-labels-idx1-ubyte.gz'
    }
    
    for name, filename in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filename)
            
    # Read images
    with gzip.open(files['images'], 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = f.read(num * rows * cols)
        
    # Read labels
    with gzip.open(files['labels'], 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = f.read(num_labels)
        
    return images, labels, num

def create_nntrainer_dat(images, labels, num_samples, output_path):
    print(f"Writing {num_samples} samples to {output_path}...")
    
    with open(output_path, 'wb') as f:
        for i in range(num_samples):
            # Extract image (784 bytes) and label (1 byte)
            img_bytes = images[i * 784 : (i + 1) * 784]
            label = labels[i]
            
            # Write 784 features as float32
            # Values are kept in [0.0, 255.0] (the C++ code divides by 255)
            for b in img_bytes:
                f.write(struct.pack("<f", float(b)))
                
            # Write 10 classes as float32 one-hot vector
            for c in range(10):
                if c == label:
                    f.write(struct.pack("<f", 1.0))
                else:
                    f.write(struct.pack("<f", 0.0))
                    
    print("Done! You can now use this dataset with more samples.")

if __name__ == "__main__":
    images, labels, total_samples = download_mnist()
    # You can specify any number up to 60000 here
    create_nntrainer_dat(images, labels, total_samples, "mnist_trainingSet.dat")
