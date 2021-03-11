import os
import shutil as sh
from tqdm.notebook import tqdm
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

from utils import stratified_split

class AlbMNIST(MNIST):
    """Pytorch MNIST dataset adapted to use albumentaions lib """

    def __init__(self, *args, alb_transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alb_transforms = alb_transforms    
             
    def set_transofrms(self, transforms):
        self.alb_transforms = transforms
        
    def __getitem__(self, idx):
        temp_transform = self.transform
        self.transform = None
        image, label = super().__getitem__(idx)
        self.transform = temp_transform
        image_np = np.array(image)
        
        if self.alb_transforms is not None:
            
            image_alb = self.alb_transforms(image=image_np)['image']

            return {"original": image_np, "augmented": image_alb, "label": label}
        
        else:
            return {"original": image_np, "label": label}
        
        
def compose_array_from_dataloader(dataloader, key="original"):
    """Creates a numpy array from a pytorch dataloader. 
           
    Parameters
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        the initial dataloader providing data to be composed as numpy array 
    key : str
        specifies what is going to be composed. 
        "original" stands for original MNIST dataset,
        "augmented" stands for the aufgmented images,
        "label" stands for annotations
        
    Returns
    -------
    output_array: numpy.array
        The output array
        If "key" param is "original" or "augmented" the array's shape is (N, H, W) where N = len(dataloader), H, W are images width and height
        If "key: param is "label" the output shape is (N,) where N = len(dataloader)      
    """

    
    sample = dataloader.dataset[0][key]
    
    if key == "label":
        dtype = np.int
        output_shape = [len(dataloader.dataset)]
    else:
        dtype = np.float32
        output_shape = [len(dataloader.dataset)] + list(sample.shape)
        
    output_array = np.zeros(output_shape, dtype=dtype)
    output_array.setflags(write=True)
    global_batch_size = dataloader.batch_size
    
         

    with tqdm(total=len(dataloader)) as pbar:
        for idx, batch in enumerate(dataloader):
            array_to_add = batch[key].numpy()
            batch_size = array_to_add.shape[0]
            output_array[global_batch_size*idx : global_batch_size*idx+batch_size] = array_to_add
            pbar.update(1)
            
    return output_array


def save_dataset_as_numpy(dataloader, file_path, key="original", message=""):
    """ Saves the data provided by dataloader as numpy array
    
    Parameters
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        the initial dataloader providing data to be saved as numpy array 
    file_path : str
        path to output file
    key : str
        specifies what is going to be composed. 
        "original" stands for original MNIST dataset,
        "augmented" stands for the aufgmented images,
        "label" stands for annotations
    message : str
        message to print
    """
    
    print(message)
    array_to_save = compose_array_from_dataloader(dataloader, key=key)
    with open(file_path, "wb") as file:
        np.save(file, array_to_save) 
        
        
def create_MNIST_np_files_with_preprocessed_augs(alb_transforms, target_dir = ".", aug_number=10, fraction_to_take = 1., batch_size=4, num_workers=1):
    """creates several np files with original MNIST dataset and its augmented versions
    
    Parameters
    ----------
    alb_transforms : albumentations.core.composition.Compose
        a composition of albumentation transforms
    target_dir : str
        path to directory to save downladed MNIST and np files
        if not present, will be created automatically
    aug_number : int
        number of augmentations to make
        generally the more the numberm the better IIC method should work,
        however the more time it takes to preprocess it and more RAM is consumed
    num_workers : int
        number of workers to utilize. The paralleliztion is done with the help of pytorch dataloader
    fraction_to_take : float
         fraction of the original dataset to take. Must be from 0. to 1.
    batch_size : int
        size of batch to feed into internal dataloader. May affec the performance
    """
    
    if not os.path.exists(target_dir):
        print("%s is not found, creating...", end = " ")
        os.mkdir(target_dir)

        
    dataset_full = AlbMNIST(
        root = os.path.join(target_dir, "MNIST"),
        download=True, 
        alb_transforms=None
    )
    
    
    if fraction_to_take < 1.:
        print("Creating split...", end=" ")
        dataset_alb, _  = stratified_split(dataset_full, train_size=fraction_to_take)
        print("Done!")
    else:
        dataset_alb = dataset_full
    
    dataloader = DataLoader(dataset_alb, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    dataset_full.set_transofrms(None)
    original_mnist_path = os.path.join(target_dir, "mnist_original.np")
    labels_path = os.path.join(target_dir, "mnist_labels.np")
    dataset_full.set_transofrms(alb_transforms)
    
    save_dataset_as_numpy(dataloader, original_mnist_path, key="original", message="Saving original dataset as np file")
    save_dataset_as_numpy(dataloader, labels_path, key="label", message="Saving labels as np file")

    for i in range(aug_number):
        aug_mnist_path = os.path.join(target_dir, "mnist_aug_" + str(i) +".np")
        save_dataset_as_numpy(dataloader, aug_mnist_path, key="augmented", message="Creating augmented dataset #%i"%i)

    
    
def create_MNIST_arrays(alb_transforms=None, aug_number=1, target_dir=".",  batch_size=256, num_workers=1):
    """ Create numpy arrays containing the MNIST dataset, the labels and the augmented images 
            
        Parameters
        ----------
        alb_transforms : albumentations.core.composition.Compose
            a composition of Albumentations transforms
        aug_number : int
            number of augmentations to make
        target_dir : str
            dir to store the dataset
        batch_size : int
            size of batch used in augmentation
        num_workers : int
            number of CPU threads to use in augmentations
            
        Returns
        -------
        originals_array : numpy.ndarray
            array with original MNIST images
        labels_array : numpy.ndarray
            array with MNIST labels
        aug_arrays : list of numpy.ndarrays
            list wih augmented versions of MNIST images
        
        """

    dataset_alb = AlbMNIST(os.path.join(target_dir, "MNIST"), download=True)
    dataloader = DataLoader(dataset_alb, batch_size=len(dataset_alb), shuffle=False, num_workers=num_workers)

    print("Fetching original dataset...", end =" ")
    originals_array = next(iter(dataloader))['original'].numpy()
    labels_array = next(iter(dataloader))['label'].numpy()
    print("Done!")

    dataset_alb.set_transofrms(alb_transforms)
    aug_arrays = []
    dataloader = DataLoader(
        dataset_alb, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    for aug_idx in range(aug_number):
        print("Making aug #%i"%aug_idx)
        aug_arrays.append(compose_array_from_dataloader(dataloader, key = "augmented"))


    return originals_array, labels_array, aug_arrays