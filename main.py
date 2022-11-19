from unet_model import UNet
from dataset import CityscapesDataset
from train import train

import torch
import torch.nn as nn
import torch.optim as optim
import os


if __name__ == '__main__':
    chkpt_path = "./Unet.pt"

    batch_size = 32
    datadir = "./Cityscapes"

    img_data = CityscapesDataset( datadir, split='train', mode='fine', augment=True)
    img_batch = torch.utils.data.DataLoader( img_data, batch_size=batch_size, shuffle=True, num_workers=2)


    val_data = CityscapesDataset( datadir, split='val', mode='fine', augment=False)
    val_batch = torch.utils.data.DataLoader( img_data, batch_size=8, shuffle=False, num_workers=2)


    n_channels, n_classes = 3, img_data.num_classes
    net = UNet( n_channels, n_classes)

    print( n_classes)

    amp = False
    learning_rate = 0.0005
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print( f"Running on Device: {device}" )

    optimizer = optim.RMSprop( net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler( enabled=amp)
    criterion = nn.CrossEntropyLoss()


    if os.path.isfile( chkpt_path):
        net.load_state_dict( torch.load(chkpt_path, map_location=torch.device('cpu') ))
        
    net = net.to( device)
    n_train = len( img_batch)


    train( net, img_batch, epochs, n_train, amp, optimizer, grad_scaler, criterion, device)

    # torch.save( net.state_dict(), chkpt_path)

    example = torch.rand(1, 3, 128, 256 )
    tsm = torch.jit.script( net.cpu() , example)
    tsm.save( "./Unet-TSM.pt" )


    print( "Done" )

