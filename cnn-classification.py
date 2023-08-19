import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt


size=224
transform=transforms.Compose([  transforms.Resize((size,size)),  transforms.ToTensor(),  transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) ])

#data
train_ds=dataset.MNIST(root='/root', train=True,transform=transforms.ToTensor(),download=True)
test_ds=dataset.MNIST(root='/root', train=False,transform=transforms.ToTensor(),download=True)
# if data from my computer
#train_ds=dataset.ImageFolder(root='D:/DataSets/classification/cat_vs_dag/PetImages/dev/train',transform=transforms.ToTensor())
#test_ds=dataset.ImageFolder(root='D:/DataSets/classification/cat_vs_dag/PetImages/dev/train',transform=transforms.ToTensor())
#train_ds.classes()  

batch_size=32
train_dl=DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=2)
test_dl=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=True,num_workers=2)


image,target=next(iter(train_dl))
plt.figure(figsize=(5,5))
for i in range(18):
  plt.subplot(3,6,i+1)  
  img=torch.transpose(image[i],0,1)
  img=torch.transpose(img,1,2)
  plt.imshow(img)
  plt.axis('off')
plt.show() 

# define model
#define class
class CNN(nn.Module):  
   def __init__(self,input_channel=3,num_class=2):   
       super(CNN,self).__init__()
       self.conv1=nn.Conv2d(in_channels=input_channel,out_channels=16,kernel_size=(3,3),padding=(1,1))     
       self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=(1,1))                 
       self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=(1,1))                
       self.pool=nn.MaxPool2d(kernel_size=(2,2))  
       self.fc1=nn.Linear(in_features=112*112*64,out_features=100)             
       self.fc2=nn.Linear(in_features=100,out_features=num_class)                                                                   
       self.bc1=nn.BatchNorm2d(16)
       self.bc2=nn.BatchNorm2d(32)
       self.do=nn.Dropout(0.2) 


   def forward(self,x):  #define out layers     
      out=F.relu(self.bc1(self.conv1(x) ))  
      out=F.relu(self.bc2(self.conv2(out)))
      out=F.relu(self.pool(out))
      out=F.relu(self.conv3(out))
      out=out.reshape(out.shape[0],-1)    
      out=F.relu(self.fc1(out) )
      out=self.do(out)
      out=self.fc2(out)
      return out
#end define the model

device='cuda' if torch.cuda.is_available() else 'cpu'
device

#define CNN algorithm
model=CNN().to(device)
model

citeration=nn.CrossEntropyLoss()

#optimizer
optimizer=optim.Adam(params=model.parameters(),lr=0.01)

epoch=5 
#train
for i in range(epoch):
      sumLoss=0
      for idx,(image,target) in enumerate(train_dl,0):   

            image=image.to(device)
            target=target.to(device)

            optimizer.zero_grad()    

            score_SGD=model(image)    
            loss=citeration(score_SGD,target)       

            sumLoss+=loss                                 
            loss.backward()                              
            optimizer.step()                     

      print(f' in epoch number {i+1} is equal to { sumLoss }'  )

#check accuracy,precision,recall,F1-score : evaluation criteria
def check_accuracy(dataloader,model):
      if dataloader.dataset.train:
           print('accuracy on train data is calculating...')
      else:
           print('accuracy on test data is calculating...')

      true_positive=0
      false_positive=0
      total=0
      model.eval()
      with torch.no_grad():
             for x,y in dataloader:   
                   x=x.to(device)
                   y=y.to(device)

                   score=model(x)
                   _,pred=score.max(1)    
                   true_positive+=(pred==y).sum()   
                   false_positive+=(pred!=y).sum()
                   total+=len(y)
                   ######
                   #accuracy=true_positive/total
                   #precision=true_positive/(true_positive+false_positive)
                   #Recall=true_positive/(true_positive+false_negative)
                   #F1_score= ( 2*(precision*Recall)+(precision+Recall) )
                  
      #print(f'acuracy SGD is { accuracy }')
      #print(f'precision SGD is { precision  }')
      #print(f' Recall SGD is {Recall}')
      #print(f' F1-score SGD is {2*(precision*Recall)+(precision+Recall)}')

check_accuracy(test_dl,model)
check_accuracy(train_dl,model)
