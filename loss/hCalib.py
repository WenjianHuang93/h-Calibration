"""
Core implementation of H-Calibration loss computation to obatined calibrated probability prediction.

This module defines the H-Calibration loss, a structured calibration-aware 
loss function that explicitly evaluates event-wise miscalibration and applies 
adaptive weighting strategies. It is designed for use in probabilistic classification tasks 
to improve the alignment between model predicted probability and accuracy.

Author: Wenjian Huang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss.mutuinfo_binning import get_mutualinfo_binning

import faiss
from concurrent.futures import ThreadPoolExecutor


def denfunc(prob_evt,veclen):  
    """
    Estimate local density (inverse of interval distance) for each probability value.

    Args:
        prob_evt (Tensor): Sorted 1D tensor of predicted probabilities.
        veclen (int): Size of the convolution window.

    Returns:
        Tensor: Estimated distance vector.
    """    
    prob_evt_pad = torch.nn.functional.pad(prob_evt.detach().view(1,-1),pad=(int(veclen/2),int(veclen/2)),mode='replicate').view(-1)
    itvdis = prob_evt_pad[veclen:] - prob_evt_pad[:(len(prob_evt_pad)-veclen)]
    return itvdis


def avgdisfunc(prob_evt,veclen):
    """
    Estimate average probability distance around each prediction using windows.

    Args:
        prob_evt (Tensor): 1D tensor of probabilities.
        veclen (int): Convolution window size.

    Returns:
        Tensor: Local average distances (density-related weights).
    """    
    prob_evt_pad = torch.nn.functional.pad(prob_evt,pad=(int(veclen/2),int(veclen/2)),mode='constant',value=0)

    convweight = torch.ones((1,1,veclen+1)).to(prob_evt.device)
    convweight_left, convweight_right = convweight.clone(), convweight.clone()
    convweight_right[:,:,:int(veclen/2)+1] = 0
    convweight_left[:,:,int(veclen/2):] = 0

    prob_evt_ma_left = torch.nn.functional.conv1d(prob_evt_pad[None,None,:].float(),weight=convweight_left).flatten()
    prob_evt_ma_right = torch.nn.functional.conv1d(prob_evt_pad[None,None,:].float(),weight=convweight_right).flatten()
    
    # Compute normalization factors
    denoninator = prob_evt.clone()
    denoninator[:] = 1.0
    denoninator = torch.nn.functional.pad(denoninator,pad=(int(veclen/2),int(veclen/2)),mode='constant',value=0)
    denoninator_left = torch.nn.functional.conv1d(denoninator[None,None,:].float(),weight=convweight_left)
    denoninator_right = torch.nn.functional.conv1d(denoninator[None,None,:].float(),weight=convweight_right)

    avgdis = ( (denoninator_left*prob_evt-prob_evt_ma_left) + (prob_evt_ma_right-denoninator_right*prob_evt) ) / (denoninator_left+denoninator_right)
    return avgdis


def hcalib_subfunc(oriprobs, probsmat, targetmat, epsilon, veclen, weightingtype, binbound_mib=None, dropout=False):
    """
    Compute the H-Calibration loss for a given probability matrix and target matrix.

    Args:
        oriprobs (Tensor): Original probabilities (optional, used in certain strategies).
        probsmat (Tensor): Calibrated probability matrix (N x C).
        targetmat (Tensor): One-hot encoded target labels (N x C).
        epsilon (float): Tolerance threshold for error.
        veclen (int): Window length used for smoothing.
        weightingtype (str): Weighting scheme to use (e.g., 'unweighted', 'denweighted').
        binbound_mib (ndarray or Tensor, optional): Custom bin boundaries for mutual info binning.
        dropout (bool): If True, apply stochastic dropout over neighborhood.

    Returns:
        Tensor: Computed H-Calibration loss.
    """
    prob_evt, occurmark = probsmat.flatten(), targetmat.flatten()
    callogit_derive = torch.log(prob_evt + 1e-20)
    logitvec_sorted, ind = torch.sort(callogit_derive)
  
    occurmark = torch.gather(occurmark,dim=0,index = ind)
    prob_evt = torch.gather(prob_evt,dim=0,index = ind)
    
    # Convolution or dropout-based smoothing
    inv_prob_evt =  1.0 - prob_evt
    prob_evt_not_occ = prob_evt * (1.0 - occurmark)
    inv_prob_evt_occ = inv_prob_evt * occurmark
    

    if dropout == True:
        prob_evt_not_occ = torch.nn.functional.pad(prob_evt_not_occ,pad=(int(veclen/2),int(veclen/2)),mode='constant',value=0)
        inv_prob_evt_occ = torch.nn.functional.pad(inv_prob_evt_occ,pad=(int(veclen/2),int(veclen/2)),mode='constant',value=0)
        prob_evt_not_occ_unfold = nn.functional.unfold(input=prob_evt_not_occ[None,None,None,:].float(),kernel_size=(1,veclen+1))
        inv_prob_evt_occ_unfold = nn.functional.unfold(input=inv_prob_evt_occ[None,None,None,:].float(),kernel_size=(1,veclen+1))
        
        catprob = torch.cat((prob_evt_not_occ_unfold[...,None],inv_prob_evt_occ_unfold[...,None]),axis=-1)
        dropprob = 0.5
        kernelwid, problen = catprob.shape[1], catprob.shape[2]
        catprob = catprob.view(1,catprob.shape[1]*catprob.shape[2],2)

        catprob = F.dropout2d(catprob[...,None], dropprob, True, True).squeeze() * (1-dropprob) # please remultiple the scaling factor
        catprob = catprob.view(1, kernelwid, problen, 2)
        catprob = catprob.sum(1)
        prob_evt_not_occ_vec, inv_prob_evt_occ_vec = catprob[...,0].flatten(), catprob[...,1].flatten()

    else:
        convweight = torch.ones((1,1,veclen+1)).to(prob_evt.device)/(veclen+1)
        prob_evt_not_occ = torch.nn.functional.pad(prob_evt_not_occ,pad=(int(veclen/2),int(veclen/2)),mode='constant',value=0)
        prob_evt_not_occ_vec = torch.nn.functional.conv1d(prob_evt_not_occ[None,None,:].float(),weight=convweight)
        
        inv_prob_evt_occ = torch.nn.functional.pad(inv_prob_evt_occ,pad=(int(veclen/2),int(veclen/2)),mode='constant',value=0)
        inv_prob_evt_occ_vec = torch.nn.functional.conv1d(inv_prob_evt_occ[None,None,:].float(),weight=convweight)


    # Compute event-wise calibration error
    T_values = (prob_evt_not_occ_vec.flatten() - inv_prob_evt_occ_vec.flatten())

    T_values = torch.relu(T_values) + torch.relu(-T_values)
    ece_vector = F.relu(  T_values  - epsilon)
    
    # Apply different weighting strategies (default is cluaceweighted, which is used in our study)
    if weightingtype == 'unweighted':
        return ece_vector.mean()
    
    if weightingtype == 'denweighted':
        itvdis = denfunc(prob_evt,veclen)
        return ( ece_vector * itvdis.detach() ).sum() / veclen
     
    if weightingtype == 'dstweighted':
        avgdis = avgdisfunc(prob_evt,veclen)
        return ( ece_vector * avgdis.detach() ).sum()
    
    if weightingtype in ['clueceweighted', 'cluaceweighted']:
        bins = 15
        kmeans = faiss.Kmeans(1,15)
        Vals = prob_evt[:,None].detach().cpu().numpy()
        kmeans.train(Vals)
        D, cluster_ids_x = kmeans.index.search(Vals,1)

        Ind = torch.nn.functional.one_hot(torch.from_numpy(cluster_ids_x.flatten()).to(prob_evt.device),num_classes=bins)
        Ind = Ind.t()
        binloss = (ece_vector[None,:]  * Ind).sum(1) / (Ind.sum(1) + 1e-5)

        if weightingtype=='cluaceweighted':
            return binloss.mean()
        elif weightingtype=='clueceweighted':
            _, maxprob_id = kmeans.index.search(torch.max(probsmat,dim=1)[0].detach().cpu().numpy()[:,None],1)
            element_bin = np.histogram(maxprob_id.flatten(),bins=bins,range=[0-0.5,bins-0.5])[0]
            element_bin = torch.from_numpy(element_bin).to(binloss.device)     
                   
            return (binloss * (element_bin/element_bin.sum())).sum()


    if weightingtype in ['softdsteceweighted','harddsteceweighted']:
        avgdis_all = avgdisfunc(prob_evt,veclen)
        if weightingtype == 'softdsteceweighted':
            argmaxid = torch.argmax(torch.log(probsmat + 1e-20),1,keepdim=True)
            argmaxmat = torch.zeros_like(probsmat).scatter_(1,argmaxid,1)
            probstop1 = probsmat.flatten()[argmaxmat.flatten()>0.5]
            probstop1 = torch.sort(probstop1)[0]
            avgdis_top1 = avgdisfunc(probstop1,veclen)
            insertsortind = torch.searchsorted(probstop1,prob_evt)
            insertsortind[insertsortind>=len(probstop1)] = len(probstop1)-1
            reweight = torch.index_select(avgdis_top1.flatten(),0,insertsortind)
            
            finalweight = avgdis_all / (reweight+1e-10)
            finalweight = finalweight / finalweight.mean()

            return ( ece_vector * finalweight.detach() ).mean()
        else:
            weight = avgdis_all / avgdis_all.mean()
            ece_vector = ece_vector * weight.detach().flatten()
    
    # Histogram-based weighting
    argmaxid = torch.argmax(torch.log(probsmat + 1e-20),1,keepdim=True)
    argmaxmat = torch.zeros_like(probsmat).scatter_(1,argmaxid,1)
    argmaxvec = argmaxmat.flatten()        

    if weightingtype in ['aceweighted', 'eceweighted', 'harddsteceweighted']:
        bins = 15
        binbound = torch.linspace(start=0,end=1,steps=bins+1)
        binbound = torch.cat((binbound[:-1,None],binbound[1:,None]),dim=1).to(probsmat.device)
        element_bin = ((probsmat.flatten()[None,argmaxvec>0.5] >= binbound[:,0][:,None]) * (probsmat.flatten()[None,argmaxvec>0.5] <= binbound[:,1][:,None])).sum(1) # element number in each bin

        Ind = (prob_evt[None,:] >= binbound[:,0][:,None]) * (prob_evt[None,:] <= binbound[:,1][:,None])
        binloss = (ece_vector[None,:]  * Ind).sum(1) / (Ind.sum(1) + 1e-5)

        if weightingtype=='aceweighted':
            return binloss.mean()
        elif weightingtype in ['eceweighted', 'harddsteceweighted']:
            return (binloss * (element_bin/element_bin.sum())).mean()
        
    
    if weightingtype in ['mibaceweighted','mibeceweighted']:     
        binbound_mib = torch.from_numpy(binbound_mib).to(prob_evt.device)
        logodds = torch.log(prob_evt+1e-20) - torch.log(1-prob_evt+1e-20)
        Ind_mib = (logodds[None,:] >= binbound_mib[:-1][:,None]) * (logodds[None,:] <= binbound_mib[1:][:,None])
        binloss_mib = (ece_vector[None,:]  * Ind_mib).sum(1) / (Ind_mib.sum(1) + 1e-5)
        if weightingtype=='mibaceweighted':
            return binloss_mib.mean()
        elif weightingtype=='mibeceweighted':
            probsmat_logodds = torch.log(probsmat+1e-20) - torch.log(1-probsmat+1e-20)
            element_bin = ((probsmat_logodds.flatten()[None,argmaxvec>0.5] >= binbound_mib[:-1][:,None]) * (probsmat_logodds.flatten()[None,argmaxvec>0.5] <= binbound_mib[1:][:,None])).sum(1) # element number in each bin
            return (binloss_mib * (element_bin/element_bin.sum())).mean()
        



def hcalibration(calprobs, targetmat, IsLossPool=False, orilogit=None, weightingtype='cluaceweighted', epsilon = 1e-20, IsIter = True, veclen=200, num_classes=None, lossweight=1e5, randomdrop=False):
    """
    Compute overall H-Calibration loss for a batch of predictions.

    Args:
        calprobs (Tensor): Calibrated probability predictions (N x C).
        targetmat (Tensor): Ground truth class labels (N,).
        IsLossPool (bool): Whether to use multithreaded parallel loss computation.
        orilogit (Tensor, optional): Original (pre-calibrated) logits.
        weightingtype (str): Weighting strategy to apply. (default cluaceweighted, used in our study)
        epsilon (float): Tolerance margin for calibration error.
        IsIter (bool): Whether to compute leave-one-class-out calibration loss.
        veclen (int): Length of convolutional smoothing window.
        num_classes (int): Total number of classes.
        lossweight (float): Scaling factor for the loss.
        randomdrop (bool): Whether to apply stochastic smoothing dropout.

    Returns:
        Tensor: Final scalar H-Calibration loss.
    """

    oriprobs = None if orilogit is None else F.softmax(orilogit, dim=1)
    targetmat = F.one_hot(targetmat, num_classes=num_classes)

    overall_loss = []
    evtnums = calprobs.shape[1]

    if weightingtype in ['mibaceweighted', 'mibeceweighted']:
        binbound_mib = get_mutualinfo_binning(logits=torch.log(calprobs+1e-20).detach().cpu().numpy(), probs=calprobs.detach().cpu().numpy(), y=targetmat.detach().cpu().numpy(), num_bins=15)
    else:
        binbound_mib = None

    if IsIter==True:
        if IsLossPool == False:
            for evtid in range(evtnums):        
                probvec = calprobs[torch.argmax(targetmat,1)!=evtid,:]
                evtvec = targetmat[torch.argmax(targetmat,1)!=evtid,:]
                oripvec = None if orilogit is None else oriprobs[torch.argmax(targetmat,1)!=evtid,:]
                val = hcalib_subfunc(oripvec, probvec, evtvec, epsilon, veclen, weightingtype, binbound_mib, randomdrop)
                overall_loss.append( val )
        else:
            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = [executor.submit(hcalib_subfunc, None, calprobs[torch.argmax(targetmat,1)!=evtid,:], targetmat[torch.argmax(targetmat,1)!=evtid,:], epsilon, veclen, weightingtype, binbound_mib, randomdrop) for evtid in range(evtnums)]
                overall_loss = [future.result() for future in futures]

    else:
        overall_loss.append(  hcalib_subfunc(oriprobs, calprobs, targetmat, epsilon, veclen, weightingtype, binbound_mib, randomdrop)  )
    
    loss = torch.stack(overall_loss)
    return loss.flatten().mean() * lossweight




