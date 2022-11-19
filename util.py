# Some Helpfull Functions:

import torch
from torch import Tensor
import matplotlib.pyplot as plt


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)



#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#



import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # with tqdm( total=num_val_batches, desc='Validation round', unit='img', leave=False) as pbar:
    for idx_batch, (image, mask_true, labelsrgb) in enumerate( dataloader):
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            
            # pbar.update(image.shape[0])
            # pbar.set_postfix(**{'dice score (batch)': dice_score.item()})

            pred_class = torch.zeros((mask_pred.size()[0], mask_pred.size()[2], mask_pred.size()[3]))
            for idx in range(0, mask_pred.size()[0]):
                pred_class[idx] = torch.argmax( mask_pred[idx], dim=0).cpu().int()

    img = image.cpu().data.numpy()[0].transpose( 1, 2, 0)
    pred = pred_class.float()/mask_pred.size()[1]
    pred = pred.cpu().numpy()[0]


    plt.figure( figsize=(15, 30))
    plt.subplot( 1, 3, 1)
    plt.imshow( img)
    plt.title( "Image" )
    plt.subplot( 1, 3, 2)
    plt.imshow( labelsrgb[0].numpy().transpose( 1, 2, 0))
    plt.title( "Labels" )
    plt.subplot( 1, 3, 3)
    plt.imshow( pred)
    plt.title( "Prediction" )
    plt.show()

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
