# Train and Evaluate the Model on The Cityscapes Dataset:

from tqdm import tqdm
import os
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from util import *


def train( net, img_batch, epochs, n_train, amp, optimizer, grad_scaler, criterion, device):
	global_step = 0

	for epoch in range( epochs):
	    net.train()
	    epoch_loss = 0
	    with tqdm(total=n_train, desc=f'Training Round, Epoch: {epoch}/{epochs}', unit='img') as pbar:
	        for idx_batch, (images, true_masks, labelsrgb) in enumerate(img_batch):
	            assert images.shape[1] == net.n_channels, \
	                f' input channels !=  loaded images channels.'

	            images = images.to(device=device, dtype=torch.float32)
	            true_masks = true_masks.to(device=device, dtype=torch.long)

	            with torch.cuda.amp.autocast(enabled=amp):
	                masks_pred = net(images)
	                loss = criterion(masks_pred, true_masks) \
	                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
	                                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
	                                    multiclass=True)

	            optimizer.zero_grad(set_to_none=True)
	            grad_scaler.scale(loss).backward()
	            grad_scaler.step(optimizer)
	            grad_scaler.update()

	            pbar.update(images.shape[0])
	            global_step += 1
	            epoch_loss += loss.item()

	            pbar.set_postfix(**{'loss (batch)': loss.item()})

	            # Evaluation round
	            if ( idx_batch % 5 == 0 and idx_batch != 0 ):
	                # Training Results:
	                pred_class = torch.zeros((masks_pred.size()[0], masks_pred.size()[2], masks_pred.size()[3]))
	                for idx in range(0, masks_pred.size()[0]):
	                    pred_class[idx] = torch.argmax( masks_pred[idx], dim=0).cpu().int()

	                pred_class = pred_class.unsqueeze(1).float()

	                img = images.cpu().data.numpy()[0].transpose( 1, 2, 0)
	                pred = pred_class.float()/masks_pred.size()[1]
	                pred = pred.cpu().numpy()[0][0]

	                plt.figure( figsize=(15, 30))
	                plt.subplot( 1, 3, 1)
	                plt.imshow( img)
	                plt.subplot( 1, 3, 2)
	                plt.imshow( labelsrgb[0].cpu().numpy().transpose( 1, 2, 0))
	                plt.subplot( 1, 3, 3)
	                plt.imshow( pred)
	                plt.show()

	                # Evaluation Results:
	                val_score = evaluate(net, val_batch, device)
	                print( f"\n For batch number {idx_batch}/{len(img_batch)}, dice validation score is : {val_score*100}% \n" )
