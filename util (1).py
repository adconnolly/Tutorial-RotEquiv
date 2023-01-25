import fsspec
import os
import xarray as xr
import wget
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_data(filename='coarse4x40104_Re900.nc'):
    ## This code doesn't work though I think it should with some small changes
    #dataset_url="https://g-fe3828.0da32.08cc.data.globus.org"
    #mapper = fsspec.get_mapper(f"{dataset_url}/filename")
    #return xr.open_dataset(mapper,decode_times=0)
        
    # If you modify this code you can copy and paste the following code:
        # import importlib
        # import sys
        # importlib.reload(sys.modules['util'])
        # from util import *
    # into a new cell in tutorial.ipynb to make sure the changes take place
  
    if os.path.isfile(filename):
        return xr.open_dataset(filename,decode_times=0)
    else:        
        wget.download("https://g-fe3828.0da32.08cc.data.globus.org/"+filename)
        return xr.open_dataset(filename,decode_times=0)

def scale(x,**kwargs):

    if 'mean' in kwargs:
        mean=kwargs.get('mean')
    else:
        mean=np.mean(x)
    
    if 'std' in kwargs:
        std=kwargs.get('std')
    else:
        std=np.std(x)
    
    return (x - mean) / std#, mean, std
    
def train_model(model, x_train, y_train, criterion, optimizer):
    model.train()
    prediction = model(x_train)
    loss = criterion(prediction, y_train).cpu()   # Calculating loss 
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients to update weights
    
def test_model(model,  x_test, y_test, criterion, optimizer, text = 'validation'):
    model.eval() # Evaluation mode (important when having dropout layers)
    test_loss = 0
    with torch.no_grad():
        prediction = model(x_test)
        test_loss = criterion(prediction, y_test).cpu().data.numpy()   # Calculating loss 
    if text!=None:
        print(text + ' loss:',test_loss)
    return test_loss

def plot_losses(validation_loss,train_loss,startEpoch=0):
    plt.figure()
    epochs=range(len(validation_loss))
    plotEpochs=slice(startEpoch,len(validation_loss))
    plt.plot(epochs[plotEpochs],train_loss[plotEpochs], label = 'Train Loss')
    plt.plot(epochs[plotEpochs],validation_loss[plotEpochs], label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    
def plot_contour(y_test,prediction,tplot,kplot,text=["True", "Predicted"],clim_mode='first'):
    prediction=prediction.detach()#.numpy()
    if clim_mode=='share':
        vmin=np.minimum( np.min(prediction[tplot,kplot].cpu().numpy()), np.min(y_test[tplot,kplot].cpu().numpy()))
        vmax=np.maximum( np.max(prediction[tplot,kplot].cpu().numpy()), np.max(y_test[tplot,kplot].cpu().numpy()))
    else:
        vmin=np.min(y_test[tplot,kplot].cpu().numpy())
        vmax=np.max(y_test[tplot,kplot].cpu().numpy())
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    c1=axs[0].contourf(y_test[tplot,kplot].cpu().T,vmin=vmin,vmax=vmax,cmap='bwr')
    c1.set_clim(vmin, vmax)
    axs[0].set_aspect('equal')
    axs[0].set_title(text[0])
    c2=axs[1].contourf(prediction[tplot,kplot].cpu().T,vmin=vmin,vmax=vmax,cmap='bwr')
    c2.set_clim(vmin, vmax)
    axs[1].set_aspect('equal')
    axs[1].set_title(text[1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    fig.colorbar(c1,cax=cbar_ax)

def plot_quiver(yList,predictionList,tplot,kplot,text=["True", "Predicted"]):
    u_test=yList[0]
    v_test=yList[1]
    u_prediction=predictionList[0]
    v_prediction=predictionList[1]
    
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    axs[0].quiver(u_test[tplot,kplot].cpu().T,v_test[tplot,kplot].cpu().T)
    axs[0].set_aspect('equal')
    axs[0].set_title(text[0])
    axs[1].quiver(u_prediction[tplot,kplot].detach().cpu().T, v_prediction[tplot,kplot].detach().cpu().T)
    axs[1].set_aspect('equal')
    axs[1].set_title(text[1])

def myrotate(inputFields_in,outputFields_in,krot,out_type='scalar'):
    inputFields_out=np.empty(inputFields_in.shape)
    outputFields_out=np.empty(outputFields_in.shape)
    for v in range(inputFields_out.shape[0]):
        inputFields_out[v]=np.rot90(inputFields_in[v].copy(),krot,axes=(-2, -1))
    for v in range(outputFields_out.shape[0]):
        outputFields_out[v]=np.rot90(outputFields_in[v].copy(),krot,axes=(-2, -1))
    
    theta=krot*np.pi/2.0
    R=np.rint([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    for s in range(inputFields_out.shape[1]):
            for k in range(inputFields_out.shape[2]):
                for i in range(inputFields_out.shape[3]):
                    for j in range(inputFields_out.shape[4]):
                        #u,v are index 0,1 of first dimension
                        inputFields_out[0:2,s,k,i,j]=R@inputFields_out[0:2,s,k,i,j] 
    inputFields_final=np.concatenate([inputFields_out[i] for i in range(inputFields_out.shape[0])],axis=1)
    
    if out_type=='sign':
        outputFields_out=-np.sign(krot%2-0.5)*outputFields_out

    return inputFields_final,outputFields_out