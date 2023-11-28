import wandb
from models.ResNet_model import *
#from models.{team_name}_model import *
from utils import *
import h5py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#read data
h5f = h5py.File('X_train.h5','r')
X_train = h5f['X_train'][:]
h5f.close()
h5f = h5py.File('X_val.h5','r')
X_val = h5f['X_val'][:]
h5f.close()
h5f = h5py.File('y_train.h5','r')
y_train = h5f['y_train'][:]
h5f.close()
h5f = h5py.File('y_val.h5','r')
y_val = h5f['y_val'][:]
h5f.close()

#creat dataloaders
trainLoader = createDataLoader(X_train, y_train, num_workers=0, batch_size=1024, shuffle=True, drop_last=True, random_seed = 42)
valLoader = createDataLoader(X_val, y_val, num_workers=0, batch_size=1024, shuffle=True, drop_last=True, random_seed = 42)

#define all the possible options for different layers that your model can accept
l1s = [FirstLayers(4, 64, 7, 1, 'same'), ]
l2s = [ResidualLayers(ResidualBlock, [3,4,6,3]), ]
l3s = [FinalLayers(14, 512, 1), ]
lossFunctions = [torch.nn.MSELoss().to(device),]

#choose a specific combination from all possible options
# after you finish defining l1s, l2s, ... ... ..., lossFunctions, lrs for your model you can just run a for loop to create all 
# possible model combinations

layers = [l1s[0], l2s[0], l3s[0]]
model = PrixFixeNet(layers).to(device)

# torchsummary may not work for your model
summary(model, (4, 110))

loss_fn = lossFunctions[0]
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

#each combination will be assinged an ID to keep track of the combinations being tested.
modelID = 0
# train the model using this sample trainer function
train(trainLoader, model, loss_fn, optimizer, scheduler, valLoader, 10, modelID)