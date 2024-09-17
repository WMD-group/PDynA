"""
pdyna.analysis: The collection of analysis functions for outputs and figures.

"""
import os
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

if os.path.exists("style.mplstyle"):
    plt.style.use("style.mplstyle")

def savitzky_golay(y, window_size=51, order=3, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    References
    [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
    [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688

    Args:   
        y (numpy.ndarray): the input signal
        window_size (int): the length of the window (must be an odd integer)
        order (int): the order of the polynomial used in the filtering
        deriv (int): the order of the derivative to compute (default is 0 means only smoothing)
        rate (int): rate of change of the x values (default is 1)

    Returns:
        numpy.ndarray: the smoothed signal
    """
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



def draw_lattice_density(Lat, uniname, saveFigures = False, n_bins = 50, num_crop = 0, screen = None, title = None):
    """
    Draws the lattice density distribution for a given lattice parameter array.

    Args:
        Lat (numpy.ndarray): The lattice parameter array.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): The number of bins for the histogram.
        num_crop (int): The number of initial rows to crop from the array.
        screen (list): The range of values to screen.
        title (str): The title of the plot.

    Returns:
        tuple: Mean and standard deviation of the lattice parameters.
    """
    
    fig_name = f"lattice_density_{uniname}.png"
    
    if np.isnan(Lat).any():
        raise TypeError("The Lat array contains nan values. ")
    
    if num_crop != 0:
        Lat = Lat[num_crop:,:]
        
    if Lat.ndim == 3:
        Lat = Lat.reshape((Lat.shape[0]*Lat.shape[1],Lat.shape[2]))
    
    if screen is None:
        histranges = np.zeros((3,2))
        for i in range(3):
            histranges[i,:] = [np.quantile(Lat[:,i], 0.02),np.quantile(Lat[:,i], 0.98)]
            
        histrange = np.zeros((2,))
        ra = np.amax(histranges[:,1])-np.amin(histranges[:,0])
        histrange[0] = np.amin(histranges[:,0])-ra*0.2
        histrange[1] = np.amax(histranges[:,1])+ra*0.2
        
        figs, axs = plt.subplots(3, 1)
        #labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
        labels = ['a','b','c']
        colors = ["C0","C1","C2"]
        for i in range(3):
            y,binEdges=np.histogram(Lat[:,i],bins=n_bins,range = histrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth=2.4)
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes, style='italic')
            
        Mu = []
        Std = []
        for i in range(3):
            mu, std = norm.fit(Lat[:,i])
            Mu.append(mu)
            Std.append(std)
            axs[i].text(0.88, 0.82, 'Mean: %.2f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
            axs[i].text(0.88, 0.58, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)    
        
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel(r'Lattice Parameter ($\mathrm{\AA}$)', fontsize = 15) # X label
            ax.set_xlim(histrange)
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[1].set_xlabel("")
        axs[2].set_ylabel("")
        
        #if not title is None:
        #    axs[0].set_title(title,fontsize=14,loc='left')
            
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
            
        plt.show()
    
    else:
        Lat1 = Lat[:,0]
        Lat2 = Lat[:,1]
        Lat3 = Lat[:,2]
        
        # screen out the values outside provided range
        Lat1 = Lat1[ (Lat1 >= screen[0]) & (Lat1 <= screen[1]) ]
        Lat2 = Lat2[ (Lat2 >= screen[0]) & (Lat2 <= screen[1]) ]
        Lat3 = Lat3[ (Lat3 >= screen[0]) & (Lat3 <= screen[1]) ]
        
        if Lat.shape[0] == len(Lat1) and Lat.shape[0] == len(Lat2) and Lat.shape[0] == len(Lat3):
            #print("crystal_lat: the screening does not detect any out-of-range value. ")
            pass
        else:
            if len(Lat1)/Lat.shape[0] < 0.95 or len(Lat2)/Lat.shape[0] < 0.95 or len(Lat3)/Lat.shape[0] < 0.95:
                print(f"!pseudo_cubic_lat: screening gets - a:{len(Lat1)}/{Lat.shape[0]}, b:{len(Lat2)}/{Lat.shape[0]}, c:{len(Lat3)}/{Lat.shape[0]}")
        
        histranges = np.zeros((3,2))
        for i in range(3):
            histranges[i,:] = [np.quantile(Lat[:,i], 0.02),np.quantile(Lat[:,i], 0.98)]
            
        histrange = np.zeros((2,))
        ra = np.amax(histranges[:,1])-np.amin(histranges[:,0])
        histrange[0] = np.amin(histranges[:,0])-ra*0.2
        histrange[1] = np.amax(histranges[:,1])+ra*0.2
        
        Lats = (Lat1,Lat2,Lat3)
        
        figs, axs = plt.subplots(3, 1)
        labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
        colors = ["C0","C1","C2"]
        for i in range(3):
            y,binEdges=np.histogram(Lats[i],bins=n_bins,range = histrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth=2.4)
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes, style='italic')
            
        Mu = []
        Std = []
        for i in range(3):
            mu, std = norm.fit(Lats[i])
            Mu.append(mu)
            Std.append(std)
            axs[i].text(0.88, 0.82, 'Mean: %.2f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
            axs[i].text(0.88, 0.58, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)    
        
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel(r'Lattice Parameter ($\mathrm{\AA}$)', fontsize = 15) # X label
            ax.set_xlim(histrange)
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[1].set_xlabel("")
        axs[2].set_ylabel("")
        
        #if not title is None:
        #    axs[0].set_title(title,fontsize=14,loc='left')
            
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
            
        plt.show()
    

    return Mu, Std



def draw_lattice_evolution(dm, steps, Tgrad, uniname, saveFigures = False, xaxis_type = 'N', Ti = None, x_lims = None, y_lims = None, invert_x = False):
    """
    Draws the evolution of lattice parameters over time or temperature or MD steps.

    Args:
        dm (numpy.ndarray): The lattice parameter array.
        steps (numpy.ndarray): The time or MD steps.
        Tgrad (float): The temperature gradient.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        xaxis_type (str): The type of x-axis ('N', 't', or 'T').
        Ti (float): Initial temperature.
        x_lims (list): Limits for the x-axis.
        y_lims (list): Limits for the y-axis.
        invert_x (bool): Whether to invert the x-axis.

    Returns:
        None
    """
    
    fig_name = f"lattice_evo_{uniname}.png"
    lw = 3
    #print(dm.shape[0],len(steps))
    if dm.shape[0] != len(steps):
        raise ValueError("The dimension of the lattice array does not match with the timesteps. ")
    if dm.ndim == 3:
        La, Lb, Lc = np.nanmean(dm[:,:,0],axis=1), np.nanmean(dm[:,:,1],axis=1), np.nanmean(dm[:,:,2],axis=1)
        
    elif dm.ndim == 2:
        La, Lb, Lc = dm[:,0], dm[:,1], dm[:,2]
    
    plt.subplots(1,1)
    if steps[0] == steps[-1] and xaxis_type == 'T':
        steps1 = list(range(0,len(steps)))
        plt.plot(steps1,La,label = r'$\mathit{a}$')
        plt.plot(steps1,Lb,label = r'$\mathit{b}$')
        plt.plot(steps1,Lc,label = r'$\mathit{c}$')
    else: 
        plt.plot(steps,La,label = r'$\mathit{a}$')
        plt.plot(steps,Lb,label = r'$\mathit{b}$')
        plt.plot(steps,Lc,label = r'$\mathit{c}$')
    ax = plt.gca()
    if xaxis_type == 'T':
        if steps[0] > steps[-1]:
            #ax.text(0.22, 0.95, f'Cooling ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Cooling ({round(Tgrad,2)} K/ps)', fontsize=14)
        elif steps[0] < steps[-1]:
            #ax.text(0.22, 0.95, f'Heating ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Heating ({round(Tgrad,2)} K/ps)', fontsize=14)
        else:
            #ax.text(0.22, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Heat bath at {Ti}K', fontsize=14)
    
    plt.legend()
    if xaxis_type == 'N':
        plt.xlabel("MD step")
    elif xaxis_type == 't':
        ax.set_xlim(left=0)
        plt.xlabel("Time (ps)")
    elif xaxis_type == 'T':
        plt.xlabel("Temperature (K)")
    plt.ylabel('Lattice Parameter ($\mathrm{AA}$)')

    if not x_lims is None:
        ax.set_xlim(x_lims)
    
    if not y_lims is None:
        ax.set_ylim(y_lims)
        
    if invert_x:
        ax.set_xlim([ax.get_xlim()[1],ax.get_xlim()[0]])
    
    plt.show()
    
    # get a smoothened line
    sgw = round(La.shape[0]/6)
    if sgw<5: sgw = 5
    if sgw%2==0: sgw+=1
    Las=savitzky_golay(La,window_size=sgw)
    Lbs=savitzky_golay(Lb,window_size=sgw)
    Lcs=savitzky_golay(Lc,window_size=sgw)
    
    plt.subplots(1,1)
    if steps[0] == steps[-1] and xaxis_type == 'T':
        steps1 = list(range(0,len(steps)))
        plt.scatter(steps1,La,s=4,alpha=0.3)
        plt.scatter(steps1,Lb,s=4,alpha=0.3)
        plt.scatter(steps1,Lc,s=4,alpha=0.3)
        plt.plot(steps1,Las,label = r'$\mathit{a}$',linewidth=lw)
        plt.plot(steps1,Lbs,label = r'$\mathit{b}$',linewidth=lw)
        plt.plot(steps1,Lcs,label = r'$\mathit{c}$',linewidth=lw)
    else: 
        plt.scatter(steps,La,s=4,alpha=0.3)
        plt.scatter(steps,Lb,s=4,alpha=0.3)
        plt.scatter(steps,Lc,s=4,alpha=0.3)
        plt.plot(steps,Las,label = r'$\mathit{a}$',linewidth=lw)
        plt.plot(steps,Lbs,label = r'$\mathit{b}$',linewidth=lw)
        plt.plot(steps,Lcs,label = r'$\mathit{c}$',linewidth=lw)
    ax = plt.gca()
    if xaxis_type == 'T':
        if steps[0] > steps[-1]:
            #ax.text(0.22, 0.95, f'Cooling ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Cooling ({round(Tgrad,2)} K/ps)', fontsize=14)
        elif steps[0] < steps[-1]:
            #ax.text(0.22, 0.95, f'Heating ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Heating ({round(Tgrad,2)} K/ps)', fontsize=14)
        else:
            #ax.text(0.22, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Heat bath at {Ti}K', fontsize=14)
    
    plt.legend()
    if xaxis_type == 'N':
        plt.xlabel("MD step")
    elif xaxis_type == 't':
        ax.set_xlim(left=0)
        plt.xlabel("Time (ps)")
    elif xaxis_type == 'T':
        plt.xlabel("Temperature (K)")
    plt.ylabel('Lattice Parameter ($\mathrm{AA}$)')

    if not x_lims is None:
        ax.set_xlim(x_lims)
    
    if not y_lims is None:
        ax.set_ylim(y_lims)
        
    if invert_x:
        ax.set_xlim([ax.get_xlim()[1],ax.get_xlim()[0]])
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    
    plt.show()


def draw_lattice_evolution_time(dm, steps, Ti,uniname, saveFigures, smoother = 0, x_lims = None, y_lims = None):
    """
    Draws the evolution of lattice parameters over time.

    Args:
        dm (numpy.ndarray): The lattice parameter array.
        steps (numpy.ndarray): The time or MD steps.
        Ti (float): Initial temperature.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): The smoothing factor.
        x_lims (list): Limits for the x-axis.
        y_lims (list): Limits for the y-axis.

    Returns:
        tuple: The steps and the smoothed lattice parameters.
    """
    
    fig_name = f"lattice_time_{uniname}.png"
    
    assert dm.shape[0] == len(steps)
    if dm.ndim == 3:
        La, Lb, Lc = np.nanmean(dm[:,:,0],axis=1), np.nanmean(dm[:,:,1],axis=1), np.nanmean(dm[:,:,2],axis=1)   
    elif dm.ndim == 2:
        La, Lb, Lc = dm[:,0], dm[:,1], dm[:,2]
    
    if smoother != 0:
        ts = steps[1] - steps[0]
        time_window = smoother # picosecond
        sgw = round(time_window/ts)
        if sgw<5: sgw = 5
        if sgw%2==0: sgw+=1
        La = savitzky_golay(La,window_size=sgw)
        Lb = savitzky_golay(Lb,window_size=sgw)
        Lc = savitzky_golay(Lc,window_size=sgw)
    
    w, h = figaspect(0.8/1.45)
    plt.subplots(figsize=(w,h))
    ax = plt.gca()
    plt.plot(steps,La,label = r'$\mathit{a}$',linewidth=2)
    plt.plot(steps,Lb,label = r'$\mathit{b}$',linewidth=2)
    plt.plot(steps,Lc,label = r'$\mathit{c}$',linewidth=2)
    #ax.text(0.2, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
    
    ax.set_xlim([0,max(steps)])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("Time (ps)", fontsize=14)
    plt.ylabel(r'Lattice Parameter ($\mathrm{AA}$)', fontsize=14)
    plt.legend(prop={'size': 12})
    

    if not x_lims is None:
        ax.set_xlim(x_lims)
    
    if not y_lims is None:
        ax.set_ylim(y_lims)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return (steps,La,Lb,Lc)


def draw_tilt_evolution(T, steps, Tgrad, uniname, saveFigures = False, xaxis_type = 'N', Ti = None, x_lims = None, y_lims = None, invert_x = False, use_gaussian = False):
    """
    Draws the evolution of tilt angles over time or temperature or MD steps.

    Args:
        T (numpy.ndarray): The tilt angle array.
        steps (numpy.ndarray): The time or MD steps.
        Tgrad (float): The temperature gradient.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        xaxis_type (str): The type of x-axis ('N', 't', or 'T').
        Ti (float): Initial temperature.
        x_lims (list): Limits for the x-axis.
        y_lims (list): Limits for the y-axis.
        invert_x (bool): Whether to invert the x-axis.
        use_gaussian (bool): Whether to use Gaussian fitting (require a large enough data).

    Returns:
        tuple: The steps and the tilt angles.
    """
    
    fig_name = f"tilt_evo_{uniname}.png"
    lw=2
    #print(dm.shape[0],len(steps))
    
    if use_gaussian:
        from tqdm import tqdm
        nstep = 20
        subdiv = np.round(np.linspace(0,T.shape[0]+1,nstep+1)).astype(int)
        Tm = []
        for i in tqdm(range(subdiv.shape[0]-1)):
            Tm.append(compute_tilt_density(T[list(range(subdiv[i],subdiv[i]+1)),:,:],method='gaussian',plot_fitting=True))
        Tm = np.array(Tm)
        subdiv = np.round(np.linspace(0,T.shape[0]-1,nstep+1)).astype(int) # endpoint difference
        newsteps = []
        for i in range(subdiv.shape[0]-1):
            newsteps.append((steps[subdiv[i]]+steps[subdiv[i+1]])/2)
        steps = np.array(newsteps)
        
    else:
        aw = round(T.shape[0]/40)
        
        Tm = []
        for i in range(T.shape[0]):
            f1 = max(0,i-aw)
            f2 = min(i+aw,T.shape[0])
            #print(list(range(f1,f2)))
            Tm.append(compute_tilt_density(T[list(range(f1,f2)),:,:],method='curve'))
        Tm = np.array(Tm)
    
    plt.subplots(1,1)
    
    plt.plot(steps,Tm[:,0],label = r'$\mathit{a}$',linewidth=lw,alpha=0.8)
    plt.plot(steps,Tm[:,1],label = r'$\mathit{b}$',linewidth=lw,alpha=0.8)
    plt.plot(steps,Tm[:,2],label = r'$\mathit{c}$',linewidth=lw,alpha=0.8)
        
    ax = plt.gca()
    if xaxis_type == 'T':
        if steps[0] > steps[-1]:
            #ax.text(0.22, 0.95, f'Cooling ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Cooling ({round(Tgrad,2)} K/ps)', fontsize=14)
        elif steps[0] < steps[-1]:
            #ax.text(0.22, 0.95, f'Heating ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Heating ({round(Tgrad,2)} K/ps)', fontsize=14)
        else:
            #ax.text(0.22, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Heat bath at {Ti}K', fontsize=14)
    
    plt.legend()
    if xaxis_type == 'N':
        plt.xlabel("MD step")
    elif xaxis_type == 't':
        ax.set_xlim(left=0)
        plt.xlabel("Time (ps)")
    elif xaxis_type == 'T':
        plt.xlabel("Temperature (K)")
    plt.ylabel('Tilting (deg)')

    if not x_lims is None:
        ax.set_xlim(x_lims)
    
    if not y_lims is None:
        ax.set_ylim(y_lims)
        
    if invert_x:
        ax.set_xlim([ax.get_xlim()[1],ax.get_xlim()[0]])
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    
    plt.show()
    
    return (steps,Tm)


def draw_tilt_evolution_time(T, steps, uniname, saveFigures, smoother = 0, y_lim = None):
    """
    Draws the evolution of tilt angles over time.

    Args:
        T (numpy.ndarray): The tilt angle array.
        steps (numpy.ndarray): The time or MD steps.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): The smoothing factor.
        y_lim (list): Limits for the y-axis.

    Returns:
        tuple: The steps and the smoothed tilt angles.
    """
    
    fig_name = f"traj_tilt_time_{uniname}.png"
    
    assert T.ndim == 3
    assert T.shape[0] == len(steps)
    
    Tline = np.empty((0,3))

    aw = 15
    
    for i in range(T.shape[0]-aw+1):
        temp = T[list(range(i,i+aw)),:,:]
        #temp1 = temp[np.where(np.logical_and(temp<20,temp>-20))]
        fitted = np.array(compute_tilt_density(temp)).reshape((1,3))
        Tline = np.concatenate((Tline,fitted),axis=0)
    
    ts = steps[1] - steps[0]
    time_window = smoother # picosecond
    sgw = round(time_window/ts)
    if sgw<5: sgw = 5
    if sgw%2==0: sgw+=1
    if smoother != 0:
        Ta = savitzky_golay(Tline[:,0],window_size=sgw)
        Tb = savitzky_golay(Tline[:,1],window_size=sgw)
        Tc = savitzky_golay(Tline[:,2],window_size=sgw)
    else:
        Ta = Tline[:,0]
        Tb = Tline[:,1]
        Tc = Tline[:,2]
    
    w, h = figaspect(0.8/1.45)
    plt.subplots(figsize=(w,h))
    ax = plt.gca()
    
    #for i in range(T.shape[0]):
    #    plt.scatter(steps[i],T[i,])
    
    plt.plot(steps[:(len(steps)-aw+1)],Ta,label = r'$\mathit{a}$',linewidth=2.5)
    plt.plot(steps[:(len(steps)-aw+1)],Tb,label = r'$\mathit{b}$',linewidth=2.5)
    plt.plot(steps[:(len(steps)-aw+1)],Tc,label = r'$\mathit{c}$',linewidth=2.5)    

    #ax.set_ylim([-45,45])
    #ax.set_yticks([-45,-30,-15,0,15,30,45])
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Tilting (deg)', fontsize=14)
    plt.legend(prop={'size': 13})
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return (steps[:(len(steps)-aw+1)],Ta,Tb,Tc)


def draw_dist_evolution(D, steps, Tgrad, uniname, saveFigures = False, xaxis_type = 'N', Ti = None, x_lims = None, y_lims = None, invert_x = False):  
    """
    Draws the evolution of distortion parameters over time or temperature or MD steps.

    Args:
        D (numpy.ndarray): The distortion array.
        steps (numpy.ndarray): The time or MD steps.
        Tgrad (float): The temperature gradient.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        xaxis_type (str): The type of x-axis ('N', 't', or 'T').
        Ti (float): Initial temperature.
        x_lims (list): Limits for the x-axis.
        y_lims (list): Limits for the y-axis.
        invert_x (bool): Whether to invert the x-axis.

    Returns:
        tuple: The steps and the distortion parameters.
    """
    
    Dm = np.nanmean(D,axis=1)
    
    if D.shape[2] == 4:
        fig_name = f"dist_evo_{uniname}.png"
        #print(dm.shape[0],len(steps))

        plt.subplots(1,1)
        labels = ["Eg","T2g","T1u","T2u"]
        
        for i in range(4):
            plt.plot(steps,Dm[:,i],label = labels[i],linewidth=1.2)


        ax = plt.gca()
        if xaxis_type == 'T':
            if steps[0] > steps[-1]:
                #ax.text(0.22, 0.95, f'Cooling ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'Cooling ({round(Tgrad,2)} K/ps)', fontsize=14)
            elif steps[0] < steps[-1]:
                #ax.text(0.22, 0.95, f'Heating ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'Heating ({round(Tgrad,2)} K/ps)', fontsize=14)
            else:
                #ax.text(0.22, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'Heat bath at {Ti}K', fontsize=14)
                
        plt.legend()
        if xaxis_type == 'N':
            plt.xlabel("MD step")
        elif xaxis_type == 't':
            ax.set_xlim(left=0)
            plt.xlabel("Time (ps)")
        elif xaxis_type == 'T':
            plt.xlabel("Temperature (K)")
        plt.ylabel('Distortion')

        if not x_lims is None:
            ax.set_xlim(x_lims)
        
        if not y_lims is None:
            ax.set_ylim(y_lims)
            
        if invert_x:
            ax.set_xlim([ax.get_xlim()[1],ax.get_xlim()[0]])
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
        plt.show()
    
    elif D.shape[2] == 3:
        fig_name = f"distB_evo_{uniname}.png"
        #print(dm.shape[0],len(steps))

        plt.subplots(1,1)
        labels = ["B100","B110","B111"]
        
        for i in range(3):
            plt.plot(steps,Dm[:,i],label = labels[i],linewidth=1.2)


        ax = plt.gca()
        if xaxis_type == 'T':
            if steps[0] > steps[-1]:
                #ax.text(0.22, 0.95, f'Cooling ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'Cooling ({round(Tgrad,2)} K/ps)', fontsize=14)
            elif steps[0] < steps[-1]:
                #ax.text(0.22, 0.95, f'Heating ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'Heating ({round(Tgrad,2)} K/ps)', fontsize=14)
            else:
                #ax.text(0.22, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'Heat bath at {Ti}K', fontsize=14)
                
        plt.legend()
        if xaxis_type == 'N':
            plt.xlabel("MD step")
        elif xaxis_type == 't':
            ax.set_xlim(left=0)
            plt.xlabel("Time (ps)")
        elif xaxis_type == 'T':
            plt.xlabel("Temperature (K)")
        plt.ylabel('Distortion')

        if not x_lims is None:
            ax.set_xlim(x_lims)
        
        if not y_lims is None:
            ax.set_ylim(y_lims)
            
        if invert_x:
            ax.set_xlim([ax.get_xlim()[1],ax.get_xlim()[0]])
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
        plt.show()

    return (steps,Dm)


def draw_dist_evolution_time(D, steps, uniname, saveFigures, smoother = 0, y_lim = None):
    """
    Draws the evolution of distortion over time.

    Args:
        D (numpy.ndarray): The distortion array.
        steps (numpy.ndarray): The time or MD steps.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): The smoothing factor.
        y_lim (list): Limits for the y-axis.

    Returns:
        tuple: The steps and the smoothed distortion parameters.
    """
    
    assert D.ndim == 3
    assert D.shape[0] == len(steps)
    
    if D.shape[2] == 4:
        
        fig_name = f"traj_dist_time_{uniname}.png"
    
        Dline = np.empty((0,4))

        aw = 15
        
        for i in range(D.shape[0]-aw+1):
            temp = D[list(range(i,i+aw)),:,:].reshape(-1,4)
            #temp1 = temp[np.where(np.logical_and(temp<20,temp>-20))]
            fitted = np.mean(temp,axis=0)[np.newaxis,:]
            Dline = np.concatenate((Dline,fitted),axis=0)
        
        ts = steps[1] - steps[0]
        time_window = smoother # picosecond
        sgw = round(time_window/ts)
        if sgw<5: sgw = 5
        if sgw%2==0: sgw+=1
        if smoother != 0:
            Da = savitzky_golay(Dline[:,0],window_size=sgw)
            Db = savitzky_golay(Dline[:,1],window_size=sgw)
            Dc = savitzky_golay(Dline[:,2],window_size=sgw)
            Dd = savitzky_golay(Dline[:,3],window_size=sgw)
        else:
            Da = Dline[:,0]
            Db = Dline[:,1]
            Dc = Dline[:,2]
            Dd = Dline[:,3]
        
        w, h = figaspect(0.8/1.45)
        plt.subplots(figsize=(w,h))
        ax = plt.gca()
        
        labels = ["Eg","T2g","T1u","T2u"]
        
        #for i in range(D.shape[0]):
        #    plt.scatter(steps[i],D[i,])
        
        plt.plot(steps[:(len(steps)-aw+1)],Da,label = labels[0],linewidth=2.5)
        plt.plot(steps[:(len(steps)-aw+1)],Db,label = labels[1],linewidth=2.5)
        plt.plot(steps[:(len(steps)-aw+1)],Dc,label = labels[2],linewidth=2.5)    
        plt.plot(steps[:(len(steps)-aw+1)],Dd,label = labels[3],linewidth=2.5)   

        #ax.set_ylim([-45,45])
        #ax.set_yticks([-45,-30,-15,0,15,30,45])
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('Time (ps)', fontsize=14)
        plt.ylabel('Distortion', fontsize=14)
        plt.legend(prop={'size': 13})
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
        
        return (steps[:(len(steps)-aw+1)],Da,Db,Dc,Dd)
    
    elif D.shape[2] == 3:
        fig_name = f"traj_distB_time_{uniname}.png"
    
        Dline = np.empty((0,3))

        aw = 15
        
        for i in range(D.shape[0]-aw+1):
            temp = D[list(range(i,i+aw)),:,:].reshape(-1,3)
            #temp1 = temp[np.where(np.logical_and(temp<20,temp>-20))]
            fitted = np.mean(temp,axis=0)[np.newaxis,:]
            Dline = np.concatenate((Dline,fitted),axis=0)
        
        ts = steps[1] - steps[0]
        time_window = smoother # picosecond
        sgw = round(time_window/ts)
        if sgw<5: sgw = 5
        if sgw%2==0: sgw+=1
        if smoother != 0:
            Da = savitzky_golay(Dline[:,0],window_size=sgw)
            Db = savitzky_golay(Dline[:,1],window_size=sgw)
            Dc = savitzky_golay(Dline[:,2],window_size=sgw)
        else:
            Da = Dline[:,0]
            Db = Dline[:,1]
            Dc = Dline[:,2]
        
        w, h = figaspect(0.8/1.45)
        plt.subplots(figsize=(w,h))
        ax = plt.gca()
        
        labels = ["B100","B110","B111"]
        
        plt.plot(steps[:(len(steps)-aw+1)],Da,label = labels[0],linewidth=2.5)
        plt.plot(steps[:(len(steps)-aw+1)],Db,label = labels[1],linewidth=2.5)
        plt.plot(steps[:(len(steps)-aw+1)],Dc,label = labels[2],linewidth=2.5)    

        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('Time (ps)', fontsize=14)
        plt.ylabel('Distortion', fontsize=14)
        plt.legend(prop={'size': 13})
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
        
        return (steps[:(len(steps)-aw+1)],Da,Db,Dc)


def compute_tilt_density(T, method = "auto", plot_fitting = False, corr_vals = None): #"curve"
    """
    Computes the tilt density from the given tilt angles.

    Args:   
        T (numpy.ndarray): The tilt angle array.
        method (str): The method for computing tilt density ('auto', 'curve', 'kmean', or 'gaussian').
        plot_fitting (bool): Whether to plot the fitted peaks, only for the Gaussian mode.
        corr_vals (list): NN1 Correlation values for the tilt angles.

    Returns:
        numpy.ndarray: The fitted tilt angles for each axis.
    """
    
    def fold_abs(arr):
        """Helper function to fold and reverse the array."""
        return arr[:(len(arr)//2)][::-1] + arr[(len(arr)//2):], (len(arr)//2)

    tup_T = (T[:,:,0].reshape((-1,)),T[:,:,1].reshape((-1,)),T[:,:,2].reshape((-1,)))
    zero_threshold = 1.5
    
    if method == "auto":
        if T.shape[1] > 200:
            method = "gaussian"
        else:
            method = "curve"
     
    if method == "curve":
        n_bins = 300
        
        Y = []
        for i in range(3):
            y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                Y.append(bincenters)
            Y.append(y)
        Y = np.transpose(np.array(Y))
        
        maxs = []
        window_size = round(n_bins/15)
        if window_size%2==0:
            window_size+=1
        for i in range(3):
            temp = savitzky_golay(Y[:,i+1], window_size=window_size, order=3, deriv=0, rate=1)
            temp, foldind = fold_abs(temp)
            maxs.append(round(abs(Y[foldind:,0][np.argmax(temp)]),4))
    
    elif method == "kmean":
        from scipy.cluster.vq import kmeans
        init_arr = np.array([-0.1,0.1])
        maxs = []
        for i in range(3):
            centers = kmeans(tup_T[i], k_or_guess=init_arr, iter=20, thresh=1e-05)[0]
            if abs(np.abs(centers[1])-np.abs(centers[0])) > 0.3:
                print("!Tilting-Kmeans: Fit error above threshold, turned to curve fitting, see difference below. ")
            maxs.append((np.abs(centers[1])+np.abs(centers[0]))/2)
    
    elif method == "gaussian":
        
        bins = 900
        cplot = []
        c = []
        for i in range(3):
            ctemp, y = np.histogram(tup_T[i],bins=bins,range=[-45,45],density=True)
            cplot.append(ctemp)
            cdiff = np.log10(np.mean(np.abs(np.diff(ctemp))))
            if cdiff > -3: # the curve is too kinky and will be smoothened
                ctemp = savitzky_golay(ctemp,window_size=((bins/7)-(bins/7)%2)+1,order=2)
            yc = (y[:-1]+y[1:])/2
            c.append(ctemp)
        c = np.array(c).T    
        cplot = np.array(cplot).T
        
        xres = 601
        scanx = np.linspace(0,30,xres)[np.newaxis,np.newaxis,:]
        std_res = 100
        std_dev = np.linspace(0.1,9,std_res)[:,np.newaxis,np.newaxis]
        ymat = yc[np.newaxis,:,np.newaxis]
        dmat = []

        d1 = (1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(ymat-scanx)**2/(2*std_dev**2))
        d2 = (1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(ymat+scanx)**2/(2*std_dev**2))
        dmat = d1+d2
        
        normfac = np.sum(dmat)/xres/std_res
        c = c*(normfac/(np.sum(c,axis=0)))
        cplot = cplot*(normfac/(np.sum(cplot,axis=0)))
        
        maxs = []
        pred = []
        preds = []
        for i in range(3):
            #dtemp = d.copy()
            #dtemp = dtemp*(np.amax(c[:,[i]])/np.amax(d,axis=0)[np.newaxis,:])
            diffmat = np.sum(np.power(np.abs(dmat-c[:,[i]][np.newaxis,:,:]),2),axis=1)
            m = round(float(scanx[0,0,:][np.argmin(np.amin(diffmat,axis=0))]),3)
            pred.append(dmat[np.where(diffmat==np.amin(diffmat))[0].astype(int),:,np.where(diffmat==np.amin(diffmat))[1].astype(int)][0,:])
            temp1 = d1[np.where(diffmat==np.amin(diffmat))[0].astype(int),:,np.where(diffmat==np.amin(diffmat))[1].astype(int)][0,:][:,np.newaxis]
            temp2 = d2[np.where(diffmat==np.amin(diffmat))[0].astype(int),:,np.where(diffmat==np.amin(diffmat))[1].astype(int)][0,:][:,np.newaxis]
            preds.append(np.concatenate((temp1,temp2),axis=1))
            
            if m<zero_threshold: m = 0
            maxs.append(m)
        #print(maxs)
        
        override=[False,False,False]
        if corr_vals:
            for i in range(3):
                if abs(corr_vals[i]) < 0.42 and maxs[i] < 5:
                    maxs[i] = 0
                    override[i] = True
        
        if plot_fitting:
            draw_tilt_prediction(cplot,pred,preds,yc,maxs,override=override)
            
    elif method == 'both':
        
        n_bins = 200
        
        Y = []
        for i in range(3):
            y,binEdges=np.histogram(np.abs(tup_T[i]),bins=n_bins,range=[0,45])
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                Y.append(bincenters)
            Y.append(y)
        Y = np.transpose(np.array(Y))
        
        maxs1 = []
        window_size = n_bins/4
        if window_size%2==0:
            window_size+=1
        for i in range(3):
            temp = savitzky_golay(Y[:,i+1], window_size=window_size, order=3, deriv=0, rate=1)
            maxs1.append(abs(Y[:,0][np.argmax(temp)]))
            
        bins = 900
        c = []
        cplot = []
        for i in range(3):
            ctemp, y = np.histogram(tup_T[i],bins=bins,range=[-45,45],density=True)
            cplot.append(ctemp)
            cdiff = np.log10(np.mean(np.abs(np.diff(ctemp))))
            if cdiff > -3: # the curve is too kinky and will be smoothened
                ctemp = savitzky_golay(ctemp,window_size=((bins/7)-(bins/7)%2)+1,order=2)
            yc = (y[:-1]+y[1:])/2
            c.append(ctemp)
        c = np.array(c).T
        cplot = np.array(cplot).T
        
        xres = 601
        scanx = np.linspace(0,30,xres)[np.newaxis,np.newaxis,:]
        std_res = 100
        std_dev = np.linspace(0.1,9,std_res)[:,np.newaxis,np.newaxis]
        ymat = yc[np.newaxis,:,np.newaxis]
        dmat = []

        d1 = (1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(ymat-scanx)**2/(2*std_dev**2))
        d2 = (1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(ymat+scanx)**2/(2*std_dev**2))
        dmat = d1+d2
        
        normfac = np.sum(dmat)/xres/std_res
        c = c*(normfac/(np.sum(c,axis=0)))
        cplot = cplot*(normfac/(np.sum(cplot,axis=0)))
        
        maxs2 = []
        pred = []
        preds = []
        for i in range(3):
            #dtemp = d.copy()
            #dtemp = dtemp*(np.amax(c[:,[i]])/np.amax(d,axis=0)[np.newaxis,:])
            diffmat = np.sum(np.power(np.abs(dmat-c[:,[i]][np.newaxis,:,:]),2),axis=1)
            m = round(float(scanx[0,0,:][np.argmin(np.amin(diffmat,axis=0))]),3)
            pred.append(dmat[np.where(diffmat==np.amin(diffmat))[0].astype(int),:,np.where(diffmat==np.amin(diffmat))[1].astype(int)][0,:])
            temp1 = d1[np.where(diffmat==np.amin(diffmat))[0].astype(int),:,np.where(diffmat==np.amin(diffmat))[1].astype(int)][0,:][:,np.newaxis]
            temp2 = d2[np.where(diffmat==np.amin(diffmat))[0].astype(int),:,np.where(diffmat==np.amin(diffmat))[1].astype(int)][0,:][:,np.newaxis]
            preds.append(np.concatenate((temp1,temp2),axis=1))
            
            if m<zero_threshold: m = 0
            maxs2.append(m)
        #print(maxs2)
        
        override=[False,False,False]
        if corr_vals:
            for i in range(3):
                if abs(corr_vals[i]) < 0.42 and maxs1[i] < 6:
                    maxs1[i] = 0
                    override[i] = True
        
        if plot_fitting:
            draw_tilt_prediction(cplot,pred,preds,yc,maxs2,override=override)
            
        maxs = [maxs1,maxs2]
                
    return maxs


def draw_tilt_prediction(c,pred,preds,yc,maxs,override=[False,False,False]):
    """
    Draws the tilt prediction based on the computed density.

    Args:
        c (numpy.ndarray): The computed density.
        pred (list): The total density.
        preds (list): The individual peaks.
        yc (numpy.ndarray): The y-coordinates for the density plot.
        maxs (list): The maximum values for each tilt angle.
        override (list): Flags to override the predictions.

    Returns:
        None
    """
    
    figs, axs = plt.subplots(3, 1)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    tlabel = [-45,-30,-15,0,15,30,45]
    for i in range(3):
        axs[i].plot(yc,c[:,i],label='raw',linewidth=5,alpha=0.9,color='grey')
        axs[i].plot(yc,pred[i],label='total',linewidth=3,alpha=0.7,color='C2')
        if not override[i]:
            axs[i].plot(yc,preds[i][:,0],linewidth=1,alpha=0.7,color='C5')
            axs[i].plot(yc,preds[i][:,1],linewidth=1,alpha=0.7,color='C5')
        axs[i].text(0.02, 0.82, labels[i], horizontalalignment='left', fontsize=15, verticalalignment='center', transform=axs[i].transAxes)
        axs[i].text(0.02, 0.52, str(maxs[i])+r'$\degree$', horizontalalignment='left', fontsize=13.6, verticalalignment='center', transform=axs[i].transAxes)
        
    for ax in axs.flat:
        #ax.set(xlabel=r'Tilt Angle ($\degree$)', ylabel='Counts (a.u.)')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticks(tlabel)
        ax.set_xlim([-45,45])
        ax.set_yticks([])
    
    axs[2].set_xlabel(r'Tilt Angle ($\degree$)', fontsize=15)
    axs[1].set_ylabel('Counts (a.u.)', fontsize=15)
    if sum(override) != 3:
        axs[2].plot([0],[0],label='peaks',linewidth=1,alpha=0.7,color='C5')
    axs[2].legend(loc=4,prop={'size': 10.5})
    
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[1].set_xlabel("")
    axs[2].set_ylabel("")


def draw_distortion_evolution_sca(D, steps, uniname, saveFigures, xaxis_type = 'N', scasize = 2.5, y_lim = 0.4):
    """
    Draws the evolution of distortion parameters over time or temperature or MD steps.

    Args:
        D (numpy.ndarray): The distortion array.
        steps (numpy.ndarray): The time or MD steps.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        xaxis_type (str): The type of x-axis ('N', 't', or 'T').
        scasize (float): Size of the scatter points.
        y_lim (float): Limit for the y-axis.

    Returns:
        None
    """

    fig_name = f"traj_dist_sca_{uniname}.png"
    
    assert D.shape[0] == len(steps)
    
    if D.ndim == 3:
        steps = np.repeat(steps,D.shape[1])
        D = D.reshape(D.shape[0]*D.shape[1],4)
        
    Eg, T2g, T1u, T2u = D[:,0], D[:,1], D[:,2], D[:,3]
        
    plt.scatter(steps,Eg,label = 'Eg', s = scasize)
    plt.scatter(steps,T2g,label = 'T2g', s = scasize)
    plt.scatter(steps,T1u,label = 'T1u', s = scasize)
    plt.scatter(steps,T2u,label = 'T2u', s = scasize)
    plt.legend()
    if y_lim != None:
        ax = plt.gca()
        ax.set_ylim([0, y_lim])
    
    if xaxis_type == 'N':
        plt.xlabel("MD step")
    elif xaxis_type == 'T':
        plt.xlabel("Temperature (K)")
    elif xaxis_type == 't':
        plt.xlabel("Time (ps)")
    plt.ylabel("Distortion")
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()


def draw_tilt_evolution_sca(T, steps, uniname, saveFigures, xaxis_type = 't', scasize = 2.5, y_lim = None):
    """
    Draws the evolution of tilt angles over time or temperature or MD steps.

    Args:
        T (numpy.ndarray): The tilt angle array.
        steps (numpy.ndarray): The time or MD steps.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        xaxis_type (str): The type of x-axis ('N', 't', or 'T').
        scasize (float): Size of the scatter points.
        y_lim (list): Limits for the y-axis.

    Returns:
        None
    """
    
    fig_name = f"traj_tilt_sca_{uniname}.png"
    
    assert T.shape[0] == len(steps)
    
    for i in range(T.shape[1]):
        if i == 0:
            plt.scatter(steps,T[:,i,0],label = r'$\mathit{a}$', s = scasize, c = 'green')
            plt.scatter(steps,T[:,i,1],label = r'$\mathit{b}$', s = scasize, c = 'blue')
            plt.scatter(steps,T[:,i,2],label = r'$\mathit{c}$', s = scasize, c = 'red')
        else:
            plt.scatter(steps,T[:,i,0], s = scasize, c = 'green')
            plt.scatter(steps,T[:,i,1], s = scasize, c = 'blue')
            plt.scatter(steps,T[:,i,2], s = scasize, c = 'red')
        
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([-45,45])
    ax.set_yticks([-45,-30,-15,0,15,30,45])
    
    if xaxis_type == 'N':
        plt.xlabel("MD step")
    elif xaxis_type == 'T':
        plt.xlabel("Temperature (K)")
    elif xaxis_type == 't':
        plt.xlabel("Time (ps)")
    plt.ylabel("Tilt angle (deg)")
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()


def draw_dist_density(D, uniname, saveFigures, n_bins = 100, xrange = [0,0.5], gaus_fit = True, title = None):
    """
    Draws the density of distortion parameters.

    Args:
        D (numpy.ndarray): The distortion array.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        xrange (list): Range for the x-axis.
        gaus_fit (bool): Whether to fit a Gaussian to the plot.
        title (str): Title for the plot.

    Returns:
        tuple: The mean and standard deviation of the fitted Gaussian if gaus_fit = True.
    """
    
    if D.shape[-1] == 4:
        fig_name = f"traj_dist_density_{uniname}.png"
        
        if D.ndim == 3:
            D = D.reshape(D.shape[0]*D.shape[1],4)
        
        figs, axs = plt.subplots(4, 1)
        labels = ["Eg","T2g","T1u","T2u"]
        colors = ["C3","C4","C5","C6"]
        for i in range(4):
            y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
            axs[i].text(0.05, 0.78, labels[i], horizontalalignment='center', fontsize=13, verticalalignment='center', transform=axs[i].transAxes)
        
        if gaus_fit: # fit a gaussian to the plot
            Mu = []
            Std = []
            for i in range(4):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)
                axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
                axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
        
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel('Distortion', fontsize = 15) # X label
            ax.set_xlim(xrange)
            ax.set_xticks(np.linspace(xrange[0], xrange[1], round((xrange[1]-xrange[0])/0.1+1)))
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[2].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[3].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[1].set_xlabel("")
        axs[2].set_xlabel("")
        axs[0].set_ylabel("")
        axs[2].set_ylabel("")
        axs[3].set_ylabel("")
        axs[1].yaxis.set_label_coords(-0.02,-0.40)
        
        if not title is None:
            axs[0].set_title(title,fontsize=14,loc='left')
            
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
            
        plt.show()
        
    elif D.shape[-1] == 3:
        fig_name = f"traj_distB_density_{uniname}.png"
        
        if D.ndim == 3:
            D = D.reshape(D.shape[0]*D.shape[1],3)
        
        figs, axs = plt.subplots(3, 1)
        labels = ["B100","B110","B111"]
        colors = ["C3","C4","C5","C6"]
        for i in range(3):
            y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
            axs[i].text(0.07, 0.78, labels[i], horizontalalignment='center', fontsize=13, verticalalignment='center', transform=axs[i].transAxes)
        
        if gaus_fit: # fit a gaussian to the plot
            Mu = []
            Std = []
            for i in range(3):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)
                axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
                axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
        
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel('Distortion', fontsize = 15) # X label
            ax.set_xlim(xrange)
            ax.set_xticks(np.linspace(xrange[0], xrange[1], round((xrange[1]-xrange[0])/0.1+1)))
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[1].set_xlabel("")
        axs[2].set_xlabel("")
        axs[0].set_ylabel("")
        axs[2].set_ylabel("")
        #axs[1].yaxis.set_label_coords(-0.02,-0.40)
        
        if not title is None:
            axs[0].set_title(title,fontsize=14,loc='left')
            
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
            
        plt.show()
        
    elif D.shape[-1] == 7:
        if D.ndim == 3:
            Dx = D[:,:,:4]
            Db = D[:,:,4:]
        elif D.ndim == 2:
            Dx = D[:,:4]
            Db = D[:,4:]
        
        dmu, dstd = draw_dist_density(Dx, uniname, saveFigures, n_bins, xrange, gaus_fit, title)
        dbmu, dbstd = draw_dist_density(Db, uniname, saveFigures, n_bins, xrange, gaus_fit, title)
        
        Mu = dmu+dbmu
        Std = dstd+dbstd

    if gaus_fit:
        return Mu, Std


def draw_dist_density_frame(D, uniname, saveFigures, n_bins = 100, xrange = [0,0.5]):
    """
    Draws the density of distortion parameters for frames.

    Args:
        D (numpy.ndarray): The distortion array.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        xrange (list): Range for the x-axis.

    Returns:
        list of float: The mean of the fitted Gaussian for each distortion parameter.
    """
    
    fig_name1 = f"frame_dist_{uniname}.png"
    fig_name2 = f"frame_distB_{uniname}.png"
    
    assert D.ndim == 2
    
    if D.shape[1] == 4:
    
        figs, axs = plt.subplots(4, 1)
        labels = ["Eg","T2g","T1u","T2u"]
        colors = ["C3","C4","C5","C6"]
        for i in range(4):
            y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
            axs[i].text(0.05, 0.78, labels[i], horizontalalignment='center', fontsize=13, verticalalignment='center', transform=axs[i].transAxes)
        
        Mu = []
        Std = []
        for i in range(4):
            dfil = D[:,i].copy()
            dfil = dfil[~np.isnan(dfil)]
            mu, std = norm.fit(dfil)
            Mu.append(mu)
            Std.append(std)
            axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
        
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel('Distortion', fontsize = 15) # X label
            ax.set_xlim(xrange)
            ax.set_xticks(np.linspace(xrange[0], xrange[1], round((xrange[1]-xrange[0])/0.1+1)))
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[2].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[3].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[1].set_xlabel("")
        axs[2].set_xlabel("")
        axs[0].set_ylabel("")
        axs[2].set_ylabel("")
        axs[3].set_ylabel("")
        axs[1].yaxis.set_label_coords(-0.02,-0.40)

        if saveFigures:
            plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
        plt.show()
        
    elif D.shape[1] == 7:
        Db = D[:,4:]
        D = D[:,:4]
        
        figs, axs = plt.subplots(4, 1)
        labels = ["Eg","T2g","T1u","T2u"]
        colors = ["C3","C4","C5","C6"]
        for i in range(4):
            y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
            axs[i].text(0.05, 0.78, labels[i], horizontalalignment='center', fontsize=13, verticalalignment='center', transform=axs[i].transAxes)
        
        Mu = []
        Std = []
        for i in range(4):
            dfil = D[:,i].copy()
            dfil = dfil[~np.isnan(dfil)]
            mu, std = norm.fit(dfil)
            Mu.append(mu)
            Std.append(std)
            axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
        
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel('Distortion', fontsize = 15) # X label
            ax.set_xlim(xrange)
            ax.set_xticks(np.linspace(xrange[0], xrange[1], round((xrange[1]-xrange[0])/0.1+1)))
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[2].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[3].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[1].set_xlabel("")
        axs[2].set_xlabel("")
        axs[0].set_ylabel("")
        axs[2].set_ylabel("")
        axs[3].set_ylabel("")
        axs[1].yaxis.set_label_coords(-0.02,-0.40)

        if saveFigures:
            plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
        plt.show()
        
        figs, axs = plt.subplots(3, 1)
        labels = ["B100","B110","B111"]
        colors = ["C3","C4","C5","C6"]
        for i in range(3):
            y,binEdges=np.histogram(Db[:,i],bins=n_bins,range=xrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
            axs[i].text(0.07, 0.78, labels[i], horizontalalignment='center', fontsize=13, verticalalignment='center', transform=axs[i].transAxes)
        
        Mu = []
        Std = []
        for i in range(3):
            dfil = Db[:,i].copy()
            dfil = dfil[~np.isnan(dfil)]
            mu, std = norm.fit(dfil)
            Mu.append(mu)
            Std.append(std)
            axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
        
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel('Distortion', fontsize = 15) # X label
            ax.set_xlim(xrange)
            ax.set_xticks(np.linspace(xrange[0], xrange[1], round((xrange[1]-xrange[0])/0.1+1)))
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[1].set_xlabel("")
        axs[0].set_ylabel("")
        axs[2].set_ylabel("")
        #axs[1].yaxis.set_label_coords(-0.02,-0.40)

        if saveFigures:
            plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
        plt.show()
        
    return Mu


def if_arrays_are_different(arr,tol=0.2):
    """
    Checks if the maximum and minimum values of an array are significantly different.

    Args:
        arr (numpy.ndarray): The input array.
        tol (float): The tolerance value for comparison.

    Returns:
        tuple: A tuple indicating the category of difference.
    """

    arrmaxs = np.amax(arr,axis=0)
    arrmins = np.amin(arr,axis=0)
    
    sdmax = sorted([(abs(arrmaxs[0]-arrmaxs[1]),0,1), (abs(arrmaxs[0]-arrmaxs[2]),0,2), (abs(arrmaxs[1]-arrmaxs[2]),1,2)])
    sdmin = sorted([(abs(arrmins[0]-arrmins[1]),0,1), (abs(arrmins[0]-arrmins[2]),0,2), (abs(arrmins[1]-arrmins[2]),1,2)])
    
    # Case 1: All three numbers are different
    if (sdmax[0][0] > tol or sdmin[0][0] > tol):
        return (3,)
    
    # Case 2: Two numbers are close together, one is different
    elif (sdmax[0][0] <= tol and sdmin[0][0] <= tol) and (sdmax[1][0] > tol or sdmin[1][0] > tol):
        return (2,sdmax[0][1:])
    
    # Case 3: All three numbers are close together
    elif (sdmax[2][0] <= tol and sdmin[2][0] <= tol):
        return (1,)
    
    else: # the values are more or less close
        print("The TCP values do not fall into the preset categories, treated as the same range. ")
        return (1,)
        #raise TypeError("Fatal: unexpected category")


def draw_tilt_density(T, uniname, saveFigures, n_bins = 100, symm_n_fold = 4, title = None):
    """
    Draws the density of tilt angles.

    Args:   
        T (numpy.ndarray): The tilt angle array.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        symm_n_fold (int): Symmetry fold for periodicity.
        title (str): Title for the plot.

    Returns:
        None
    """
    
    fig_name=f"traj_tilt_density_{uniname}.png"
    
    from pdyna.structural import periodicity_fold
    T = periodicity_fold(T,n_fold=symm_n_fold)
    
    if symm_n_fold == 2:
        hrange = [-90,90]
        tlabel = [-90,-60,-30,0,30,60,90]
    elif symm_n_fold == 4:
        hrange = [-45,45]
        tlabel = [-45,-30,-15,0,15,30,45]
    elif symm_n_fold == 8:
        hrange = [0,45]
        tlabel = [0,15,30,45]
    
    if type(T) is list:
        T=np.vstack(T)
        T_a = T[:,0]
        T_b = T[:,1]
        T_c = T[:,2]
    else:
        if T.ndim == 3:
            T_a = T[:,:,0].reshape(-1,)
            T_b = T[:,:,1].reshape(-1,)
            T_c = T[:,:,2].reshape(-1,)
        elif T.ndim == 2:
            T_a = T[:,0]
            T_b = T[:,1]
            T_c = T[:,2]
    tup_T = (T_a,T_b,T_c)
    
    figs, axs = plt.subplots(3, 1)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2"]
    for i in range(3):
        y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=hrange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth=2)
        axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes)
        
    for ax in axs.flat:
        #ax.set(xlabel=r'Tilt Angle ($\degree$)', ylabel='Counts (a.u.)')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(hrange)
        ax.set_xticks(tlabel)
        ax.set_yticks([])
    
    axs[2].set_xlabel(r'Tilt Angle ($\degree$)', fontsize=15)
    axs[1].set_ylabel('Counts (a.u.)', fontsize=15)
    
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[1].set_xlabel("")
    axs[2].set_ylabel("")
    
    if not title is None:
        axs[0].set_title(title,fontsize=16)
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()


def draw_conntype_tilt_density(T, oc, uniname, saveFigures, n_bins = 100, symm_n_fold = 4, title = None):
    """ 
    Isolate tilting pattern wrt. the connectivity type.  

    Args:
        T (numpy.ndarray): The tilt angle array.
        oc (list): The connectivity type list.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        symm_n_fold (int): Symmetry fold for periodicity.
        title (str): Title for the plot.

    Returns:
        None
    """
    
    fig_name=f"traj_tilt_density_{uniname}.png"
    
    from pdyna.structural import periodicity_fold
    T = periodicity_fold(T,n_fold=symm_n_fold)
    
    if symm_n_fold == 2:
        hrange = [-90,90]
        tlabel = [-90,-60,-30,0,30,60,90]
    elif symm_n_fold == 4:
        hrange = [-45,45]
        tlabel = [-45,-30,-15,0,15,30,45]
    elif symm_n_fold == 8:
        hrange = [0,45]
        tlabel = [0,15,30,45]
    
    types = []
    Ts = []
    for ent in oc:
        types.append(ent)
        T_a = T[:,oc[ent],0].reshape(-1,)
        T_b = T[:,oc[ent],1].reshape(-1,)
        T_c = T[:,oc[ent],2].reshape(-1,)
        Ts.append((T_a,T_b,T_c))
    
    figs, axs = plt.subplots(3, 1)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2"]
    lstyle = ["solid", "dashed", "dotted", "dashdot"]
    if len(types) > len(lstyle): 
        raise TypeError("The connectivity types are more than available line styles. ")
    for j,t in enumerate(types):
        for i in range(3):
            y,binEdges=np.histogram(Ts[j][i],bins=n_bins,range=hrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                axs[i].plot(bincenters,y,label = t,color = colors[i],linewidth=2,linestyle=lstyle[j])
            else:
                axs[i].plot(bincenters,y,color = colors[i],linewidth=2,linestyle=lstyle[j])
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes)
            
    for ax in axs.flat:
        #ax.set(xlabel=r'Tilt Angle ($\degree$)', ylabel='Counts (a.u.)')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(hrange)
        ax.set_xticks(tlabel)
        ax.set_yticks([])
    
    axs[2].set_xlabel(r'Tilt Angle ($\degree$)', fontsize=15)
    axs[1].set_ylabel('Counts (a.u.)', fontsize=15)
    
    axs[0].legend(prop={'size': 12},frameon=True)
    
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[1].set_xlabel("")
    axs[2].set_ylabel("")
    
    if not title is None:
        axs[0].set_title(title,fontsize=16)
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()


def draw_octatype_tilt_density_transient(Ttype, steps, typelib, config_types, uniname, saveFigures, smoother = 0):
    """ 
    Isolate tilting pattern wrt. the local halide configuration in transient mode.  

    Args:
        Ttype (list): The tilt angle array for each type.
        steps (numpy.ndarray): The time or MD steps.
        typelib (list): The types.
        config_types (list): The configuration types.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): Smoothing window size.

    Returns:
        tuple: The configuration types, steps, and tilt angles.
    """
    
    from pdyna.structural import periodicity_fold
    fig_name=f"tilt_octatype_density_evo_{uniname}.png"
    fig_name1=f"tilt_octatype_density_tcp_evo_{uniname}.png"
    
    typesname = ["I6 Br0","I5 Br1","I4 Br2: cis","I4 Br2: trans","I3 Br3: fac",
                 "I3 Br3: mer","I2 Br4: cis","I2 Br4: trans","I1 Br5","I0 Br6"]
    typexval = [0,1,1.83,2.17,2.83,3.17,3.83,4.17,5,6]
    typextick = ['0','1','2c','2t','3f','3m','4c','4t','5','6']
    
    config_types = list(config_types)
    config_involved = []
    
    awside = 14
    
    Tarr = []
    for ti, T in enumerate(Ttype):
        if T.shape[1] < 35: continue # ignore population that is too small
        Tmax = []
        for i in range(T.shape[0]):
            fi = max(0,i-awside)
            ff = min(T.shape[0],i+awside)
            temp = T[list(range(fi,ff)),:,:]
            #temp1 = temp[np.where(np.logical_and(temp<20,temp>-20))]
            fitted = np.array(compute_tilt_density(temp,method='curve'))
            Tmax.append(fitted)
        Tmax = np.array(Tmax)
        Tarr.append(Tmax)
        config_involved.append(config_types[ti])  
    Tarr = np.array(Tarr)
    
    if smoother != 0:
        sgw = smoother
        if sgw<5: sgw = 5
        if sgw%2==0: sgw+=1
        for i in range(Tarr.shape[0]):
            for j in range(3):
                Tarr[i,:,j] = savitzky_golay(Tarr[i,:,j],window_size=sgw)

    
    lw = 1.4
    fig, axs = plt.subplots(3,1,figsize=(5.1,3.5),sharey=True,sharex=True)
    colors = plt.cm.plasma(np.linspace(0, 1, 10))
    for ti in range(Tarr.shape[0]):
        axs[0].plot(steps,Tarr[ti,:,0],label = typesname[config_types[ti]],alpha=0.7,linewidth=lw, color=colors[config_involved[ti]])
        axs[1].plot(steps,Tarr[ti,:,1],alpha=0.7,linewidth=lw, color=colors[config_involved[ti]])
        axs[2].plot(steps,Tarr[ti,:,2],alpha=0.7,linewidth=lw, color=colors[config_involved[ti]])  

    axs[2].set_xlabel('Temperature (K)', fontsize=14)
    axs[1].set_ylabel('Tilting (deg)', fontsize=14)
    axs[0].legend(prop={'size': 8},ncol=4,frameon=True)
    axs[0].set_title("Tilting with Types", fontsize=15)
    if steps[0]>steps[1]:
        axs[2].set_xlim([max(steps),min(steps)])
    else:
        axs[2].set_xlim([min(steps),max(steps)])
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.05)
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return config_involved, steps, Tarr


def draw_octatype_tilt_density(Ttype, typelib, config_types, uniname, saveFigures, corr_vals = None, n_bins = 100, symm_n_fold = 4):
    """ 
    Isolate tilting pattern wrt. the local halide configuration.  

    Args:
        Ttype (list): The tilt angle array for each type.
        typelib (list): The types.
        config_types (list): The configuration types.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        corr_vals (list): The correlation values.
        n_bins (int): Number of bins for the histogram.
        symm_n_fold (int): Symmetry fold for periodicity.

    Returns:    
        list: The fitted tilt angles.
    """
    
    from pdyna.structural import periodicity_fold
    fig_name=f"tilt_octatype_density_{uniname}.png"
    fig_name1=f"tilt_octatype_density_tcp_{uniname}.png"
    
    if symm_n_fold == 2:
        hrange = [-90,90]
        tlabel = [-90,-60,-30,0,30,60,90]
    elif symm_n_fold == 4:
        hrange = [-45,45]
        tlabel = [-45,-30,-15,0,15,30,45]
    elif symm_n_fold == 8:
        hrange = [0,45]
        tlabel = [0,15,30,45]
    
    typesname = ["I6 Br0","I5 Br1","I4 Br2: cis","I4 Br2: trans","I3 Br3: fac",
                 "I3 Br3: mer","I2 Br4: cis","I2 Br4: trans","I1 Br5","I0 Br6"]
    typexval = [0,1,1.83,2.17,2.83,3.17,3.83,4.17,5,6]
    typextick = ['0','1','2c','2t','3f','3m','4c','4t','5','6']
    
    config_types = list(config_types)
    config_involved = []
    
    maxs = np.empty((0,3))
    for ti, T in enumerate(Ttype):
        
        T = periodicity_fold(T,n_fold=symm_n_fold)
        
        T_a = T[:,:,0].reshape((T.shape[0]*T.shape[1]))
        T_b = T[:,:,1].reshape((T.shape[0]*T.shape[1]))
        T_c = T[:,:,2].reshape((T.shape[0]*T.shape[1]))
        tup_T = (T_a,T_b,T_c)
        assert len(tup_T) == 3
        
        figs, axs = plt.subplots(3, 1)
        labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
        colors = ["C0","C1","C2"]
        for i in range(3):
            y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=hrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i])
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes)
            
        for ax in axs.flat:
            #ax.set(xlabel=r'Tilt Angle ($\degree$)', ylabel='Counts (a.u.)')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlim(hrange)
            ax.set_xticks(tlabel)
            ax.set_yticks([])
        
        axs[2].set_xlabel(r'Tilt Angle ($\degree$)', fontsize=15)
        axs[1].set_ylabel('Counts (a.u.)', fontsize=15)
        
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[1].set_xlabel("")
        axs[2].set_ylabel("")
        
        axs[0].set_title(typesname[config_types[ti]],fontsize=16)
        config_involved.append(typesname[config_types[ti]])   

        plt.show()
        
        m1 = np.array(compute_tilt_density(T,method='curve')).reshape(1,-1)
        maxs = np.concatenate((maxs,m1),axis=0)
    
    # plot type dependence   
    plotx = np.array([typexval[i] for i in config_types])
    plotxlab = [typextick[i] for i in config_types]
    histcolortypes = ['grey','darkred','darkblue']
    histfull = [histcolortypes[i] for i in [0,0,1,2,1,2,1,2,0,0]]
    histcolors = [histfull[i] for i in config_types]
    
    scaalpha = 0.9
    scasize = 50
    
# =============================================================================
#     plt.subplots(1,1)
#     ax = plt.gca()
#     ax.scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
#     ax.scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
#     ax.scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
#     plt.legend(prop={'size': 12})
#     
#     ax.tick_params(axis='both', which='major', labelsize=14)
#     ax.set_ylabel('Tilting (deg)', fontsize = 15) # Y label
#     ax.set_xlabel('Br content', fontsize = 15) # X label
#     ax.set_xticks(plotx)
#     ax.set_xticklabels(plotxlab)
#     ax.set_ylim(bottom=0)
# =============================================================================
    
    
    config_types = list(config_types)
    types = np.zeros((10,)).astype(int)
    for i,t in enumerate(typelib):
        types[config_types[i]] = len(t)
    
    y1, y2, y3 = np.zeros((7,)),np.zeros((7,)),np.zeros((7,))
    y1[:2] = types[:2]
    y2[2:5] = types[[2,4,6]]
    y3[2:5] = types[[3,5,7]]
    y1[5:] = types[8:]
    xts = ['0', '1', '2', '3', '4', '5', '6']
    
    # plot bars in stack manner
    
    fig, axs = plt.subplots(2,1,figsize=(5.1,4.2),gridspec_kw={'height_ratios':[3, 1]},sharex=False)

    axs[0].scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    axs[0].scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    axs[0].scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    axs[0].legend(prop={'size': 13.2},frameon = True)
    
    axs[1].bar(xts, y1, width=0.5, color=histcolortypes[0])
    axs[1].bar(xts, y2, bottom=y1, width=0.5, color=histcolortypes[1])
    axs[1].bar(xts, y3, bottom=y1+y2, width=0.5, color=histcolortypes[2])
    axs[1].legend(['pure', 'cis/fac', 'trans/mer'],prop={'size': 11},frameon = True)
    
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
    axs[1].set_ylabel('Count', fontsize = 15) 
    axs[1].set_xlabel('Br content', fontsize = 15) # X label
    axs[0].set_xticks(plotx)
    axs[0].set_xticklabels(plotxlab)
    #axs[0].set_xticks(typexval)
    #axs[0].set_xticklabels(typextick)
    for tick_label, color in zip(axs[0].get_xticklabels(), histcolors):
        tick_label.set_color(color)
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[0].set_ylim(bottom=0)
    axs[1].set_xlim(axs[0].get_xlim())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    if not (corr_vals is None):
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable
        scasize = 80
        cate = if_arrays_are_different(corr_vals)
        cmnames = ['summer','winter','cool_r']
            
        if cate[0] == 3: # all TCP values are in different ranges and thus plot with three colorbars
            
            norm1 = Normalize(vmin=np.amin(corr_vals[:,0]), vmax=np.amax(corr_vals[:,0]))  
            norm2 = Normalize(vmin=np.amin(corr_vals[:,1]), vmax=np.amax(corr_vals[:,1]))  
            norm3 = Normalize(vmin=np.amin(corr_vals[:,2]), vmax=np.amax(corr_vals[:,2]))

            fig, axs = plt.subplots(2,1,figsize=(8.6,6.0),gridspec_kw={'height_ratios':[2.6, 0.8]},sharex=False)
            
            axs[0].scatter(plotx,maxs[:,0],alpha=scaalpha,s=scasize,c=corr_vals[:,0],cmap=cmnames[0],norm=norm1)
            axs[0].scatter(plotx,maxs[:,1],alpha=scaalpha,s=scasize,c=corr_vals[:,1],cmap=cmnames[1],norm=norm2)
            axs[0].scatter(plotx,maxs[:,2],alpha=scaalpha,s=scasize,c=corr_vals[:,2],cmap=cmnames[2],norm=norm3)
            
            # Add colorbars
            clb1 = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmnames[0]),ax=axs[0], pad=-0.05)
            clb2 = plt.colorbar(ScalarMappable(norm=norm2, cmap=cmnames[1]),ax=axs[0], pad=-0.03)
            clb3 = plt.colorbar(ScalarMappable(norm=norm3, cmap=cmnames[2]),ax=axs[0], pad=0.025)
            
            clb1.set_label(label='TCP (a.u.)', size=13)
            
            axs[1].bar(xts, y1, width=0.5, color=histcolortypes[0])
            axs[1].bar(xts, y2, bottom=y1, width=0.5, color=histcolortypes[1])
            axs[1].bar(xts, y3, bottom=y1+y2, width=0.5, color=histcolortypes[2])
            axs[1].legend(['pure', 'cis/fac', 'trans/mer'],prop={'size': 12},loc='upper left', bbox_to_anchor=(1, 1))
            
            axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
            axs[1].set_ylabel('Count', fontsize = 15) 
            axs[1].set_xlabel('Br content', fontsize = 15) # X label
            axs[0].set_xticks(plotx)
            axs[0].set_xticklabels(plotxlab)
            #axs[0].set_xticks(typexval)
            #axs[0].set_xticklabels(typextick)
            for tick_label, color in zip(axs[0].get_xticklabels(), histcolors):
                tick_label.set_color(color)
            axs[1].set_yticks([])
            axs[1].set_yticklabels([])
            axs[0].set_ylim(bottom=0)
            axs[1].set_xlim(axs[0].get_xlim())
            
            newwid = axs[0].get_position().width
            pax1 = axs[1].get_position()
            new_position = [pax1.x0, pax1.y0, newwid, pax1.height+0.03]
            axs[1].set_position(new_position)

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0.15, hspace=0.15)
            
            if saveFigures:
                plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
            plt.show()

        elif cate[0] == 2: # two axes have similar TCP and thus plot with two colorbars
            
            combaxes = list(cate[1])
            soleaxis = [i for i in [0,1,2] if i not in combaxes]
            norm1 = Normalize(vmin=np.amin(corr_vals[:,combaxes]), vmax=np.amax(corr_vals[:,combaxes]))  
            norm2 = Normalize(vmin=np.amin(corr_vals[:,soleaxis]), vmax=np.amax(corr_vals[:,soleaxis]))  

            fig, axs = plt.subplots(2,1,figsize=(8.0,6.0),gridspec_kw={'height_ratios':[2.6, 0.8]},sharex=False)
            
            axs[0].scatter(plotx,maxs[:,soleaxis[0]],alpha=scaalpha,s=scasize,c=corr_vals[:,soleaxis[0]],cmap=cmnames[0],norm=norm2)
            axs[0].scatter(plotx,maxs[:,combaxes[0]],alpha=scaalpha,s=scasize,c=corr_vals[:,combaxes[0]],cmap=cmnames[1],norm=norm1)
            axs[0].scatter(plotx,maxs[:,combaxes[1]],alpha=scaalpha,s=scasize,c=corr_vals[:,combaxes[1]],cmap=cmnames[1],norm=norm1)
            
            # Add colorbars
            clb1 = plt.colorbar(ScalarMappable(norm=norm2, cmap=cmnames[0]),ax=axs[0], pad=-0.03)
            clb2 = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmnames[1]),ax=axs[0], pad=0.025)
            
            clb1.set_label(label='TCP (a.u.)', size=13)
            
            axs[1].bar(xts, y1, width=0.5, color=histcolortypes[0])
            axs[1].bar(xts, y2, bottom=y1, width=0.5, color=histcolortypes[1])
            axs[1].bar(xts, y3, bottom=y1+y2, width=0.5, color=histcolortypes[2])
            axs[1].legend(['pure', 'cis/fac', 'trans/mer'],prop={'size': 12},loc='upper left', bbox_to_anchor=(1, 1))
            
            axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
            axs[1].set_ylabel('Count', fontsize = 15) 
            axs[1].set_xlabel('Br content', fontsize = 15) # X label
            axs[0].set_xticks(plotx)
            axs[0].set_xticklabels(plotxlab)
            #axs[0].set_xticks(typexval)
            #axs[0].set_xticklabels(typextick)
            for tick_label, color in zip(axs[0].get_xticklabels(), histcolors):
                tick_label.set_color(color)
            axs[1].set_yticks([])
            axs[1].set_yticklabels([])
            axs[0].set_ylim(bottom=0)
            axs[1].set_xlim(axs[0].get_xlim())
            
            newwid = axs[0].get_position().width
            pax1 = axs[1].get_position()
            new_position = [pax1.x0, pax1.y0, newwid, pax1.height+0.03]
            axs[1].set_position(new_position)

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0.15, hspace=0.15)
            
            if saveFigures:
                plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
            plt.show()
            
        elif cate[0] == 1: # all TCP values are in the same range
        
            norm1 = Normalize(vmin=np.amin(corr_vals), vmax=np.amax(corr_vals))  

            fig, axs = plt.subplots(2,1,figsize=(7.4,6.0),gridspec_kw={'height_ratios':[2.6, 0.8]},sharex=False)
            
            axs[0].scatter(plotx,maxs[:,0],alpha=scaalpha,s=scasize,c=corr_vals[:,0],cmap=cmnames[0],norm=norm1)
            axs[0].scatter(plotx,maxs[:,1],alpha=scaalpha,s=scasize,c=corr_vals[:,1],cmap=cmnames[0],norm=norm1)
            axs[0].scatter(plotx,maxs[:,2],alpha=scaalpha,s=scasize,c=corr_vals[:,2],cmap=cmnames[0],norm=norm1)
            
            # Add colorbars
            clb1 = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmnames[0]),ax=axs[0], pad=0.025)
            
            clb1.set_label(label='TCP (a.u.)', size=13)
            
            axs[1].bar(xts, y1, width=0.5, color=histcolortypes[0])
            axs[1].bar(xts, y2, bottom=y1, width=0.5, color=histcolortypes[1])
            axs[1].bar(xts, y3, bottom=y1+y2, width=0.5, color=histcolortypes[2])
            axs[1].legend(['pure', 'cis/fac', 'trans/mer'],prop={'size': 12},loc='upper left', bbox_to_anchor=(1, 1))
            
            axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
            axs[1].set_ylabel('Count', fontsize = 15) 
            axs[1].set_xlabel('Br content', fontsize = 15) # X label
            axs[0].set_xticks(plotx)
            axs[0].set_xticklabels(plotxlab)
            #axs[0].set_xticks(typexval)
            #axs[0].set_xticklabels(typextick)
            for tick_label, color in zip(axs[0].get_xticklabels(), histcolors):
                tick_label.set_color(color)
            axs[1].set_yticks([])
            axs[1].set_yticklabels([])
            axs[0].set_ylim(bottom=0)
            axs[1].set_xlim(axs[0].get_xlim())
            
            newwid = axs[0].get_position().width
            pax1 = axs[1].get_position()
            new_position = [pax1.x0, pax1.y0, newwid, pax1.height+0.03]
            axs[1].set_position(new_position)

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0.15, hspace=0.15)
            
            if saveFigures:
                plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
            plt.show()
        
    return maxs


def draw_octatype_dist_density(Dtype, config_types, uniname, saveFigures, n_bins = 100, xrange = [0,0.5]):
    """ 
    Isolate distortion mode wrt. the local halide configuration.  

    Args:
        Dtype (list): The distortion array for each type.
        config_types (list): The configuration types.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        xrange (list): Range for the histogram.

    Returns:
        tuple: The distortion values, and distortion standard deviations.
    """
    
    typesname = ["I6 Br0","I5 Br1","I4 Br2: cis","I4 Br2: trans","I3 Br3: fac",
                 "I3 Br3: mer","I2 Br4: cis","I2 Br4: trans","I1 Br5","I0 Br6"]
    typexval = [0,1,1.83,2.17,2.83,3.17,3.83,4.17,5,6]
    typextick = ['0','1','2c','2t','3f','3m','4c','4t','5','6']
    
    config_types = list(config_types)
    config_involved = []
    
    if Dtype[0].shape[2] == 4:
    
        fig_name=f"dist_octatype_density_{uniname}.png"

        Dgauss = np.empty((0,4))
        Dgaussstd = np.empty((0,4))
        
        for di, D in enumerate(Dtype):
            if D.ndim == 3:
                D = D.reshape(D.shape[0]*D.shape[1],4)
            
            #figs, axs = plt.subplots(4, 1)
            labels = ["Eg","T2g","T1u","T2u"]
            colors = ["C3","C4","C5","C6"]
            for i in range(4):
                y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                #axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
                #axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
            
            Mu = []
            Std = []
            for i in range(4):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)
                #axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
                #axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
        
            config_involved.append(typesname[config_types[di]])    
                
            #plt.show()
            
            Dgauss = np.concatenate((Dgauss,np.array(Mu).reshape(1,-1)),axis=0)
            Dgaussstd = np.concatenate((Dgaussstd,np.array(Std).reshape(1,-1)),axis=0)
         
        # plot type dependence   
        plotx = np.array([typexval[i] for i in config_types])
        plotxlab = [typextick[i] for i in config_types]
        
        scaalpha = 0.8
        scasize = 50
        plt.subplots(1,1)
        ax = plt.gca()
        ax.scatter(plotx-0.075,Dgauss[:,0],label='Eg',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx-0.075,Dgauss[:,0],yerr=Dgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx-0.025,Dgauss[:,1],label='T2g',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx-0.025,Dgauss[:,1],yerr=Dgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx+0.025,Dgauss[:,2],label='T1u',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx+0.025,Dgauss[:,2],yerr=Dgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx+0.075,Dgauss[:,3],label='T2u',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx+0.075,Dgauss[:,3],yerr=Dgaussstd[:,3],fmt='o',solid_capstyle='projecting', capsize=5)
        plt.legend(prop={'size': 12})
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(r'Distortion', fontsize = 15) # Y label
        ax.set_xlabel('Br content', fontsize = 15) # X label
        ax.set_xticks(plotx)
        ax.set_xticklabels(plotxlab)
        ax.set_ylim(bottom=0)
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
    
    elif Dtype[0].shape[2] == 3:
        fig_name=f"distB_octatype_density_{uniname}.png"

        Dgauss = np.empty((0,3))
        Dgaussstd = np.empty((0,3))
        
        for di, D in enumerate(Dtype):
            if D.ndim == 3:
                D = D.reshape(D.shape[0]*D.shape[1],3)
            
            #figs, axs = plt.subplots(4, 1)
            labels = ["B100","B110","B111"]
            colors = ["C3","C4","C5","C6"]
            for i in range(3):
                y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                #axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
                #axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
            
            Mu = []
            Std = []
            for i in range(3):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)
                #axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
                #axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
        
            config_involved.append(typesname[config_types[di]])    
                
            #plt.show()
            
            Dgauss = np.concatenate((Dgauss,np.array(Mu).reshape(1,-1)),axis=0)
            Dgaussstd = np.concatenate((Dgaussstd,np.array(Std).reshape(1,-1)),axis=0)
         
        # plot type dependence   
        plotx = np.array([typexval[i] for i in config_types])
        plotxlab = [typextick[i] for i in config_types]
        
        scaalpha = 0.8
        scasize = 50
        plt.subplots(1,1)
        ax = plt.gca()
        ax.scatter(plotx-0.05,Dgauss[:,0],label=labels[0],alpha=scaalpha,s=scasize)
        ax.errorbar(plotx-0.05,Dgauss[:,0],yerr=Dgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx,Dgauss[:,1],label=labels[1],alpha=scaalpha,s=scasize)
        ax.errorbar(plotx,Dgauss[:,1],yerr=Dgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx+0.05,Dgauss[:,2],label=labels[2],alpha=scaalpha,s=scasize)
        ax.errorbar(plotx+0.05,Dgauss[:,2],yerr=Dgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
        plt.legend(prop={'size': 12})
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(r'Distortion', fontsize = 15) # Y label
        ax.set_xlabel('Br content', fontsize = 15) # X label
        ax.set_xticks(plotx)
        ax.set_xticklabels(plotxlab)
        ax.set_ylim(bottom=0)
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
    
    return Dgauss, Dgaussstd


def draw_octatype_lat_density(Ltype, config_types, uniname, saveFigures, n_bins = 100):
    """ 
    Isolate distortion mode wrt. the local halide configuration.  

    Args:
        Ltype (list): The lattice parameter array for each type.
        config_types (list): The configuration types.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        tuple: The lattice values, and lattice standard deviations.
    """
    
    fig_name=f"lat_octatype_density_{uniname}.png"
    
    typesname = ["I6 Br0","I5 Br1","I4 Br2: cis","I4 Br2: trans","I3 Br3: fac",
                 "I3 Br3: mer","I2 Br4: cis","I2 Br4: trans","I1 Br5","I0 Br6"]
    typexval = [0,1,1.83,2.17,2.83,3.17,3.83,4.17,5,6]
    typextick = ['0','1','2c','2t','3f','3m','4c','4t','5','6']
    
    config_types = list(config_types)
    config_involved = []
    
    Lgauss = np.empty((0,3))
    Lgaussstd = np.empty((0,3))
    
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2"]
    
    for di, L in enumerate(Ltype):
        if L.ndim == 3:
            L = L.reshape(L.shape[0]*L.shape[1],3)
        
# =============================================================================
#         figs, axs = plt.subplots(3, 1)
#         
#         histranges = np.zeros((3,2))
#         for i in range(3):
#             histranges[i,:] = [np.quantile(L[:,i], 0.02),np.quantile(L[:,i], 0.98)]
#             
#         histrange = np.zeros((2,))
#         ra = np.amax(histranges[:,1])-np.amin(histranges[:,0])
#         histrange[0] = np.amin(histranges[:,0])-ra*0.2
#         histrange[1] = np.amax(histranges[:,1])+ra*0.2
# 
#         for i in range(3):
#             y,binEdges=np.histogram(L[:,i],bins=n_bins,range=histrange)
#             bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#             axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
#             axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
# =============================================================================
        
        Mu = []
        Std = []
        for i in range(3):
            dfil = L[:,i].copy()
            dfil = dfil[~np.isnan(dfil)]
            mu, std = norm.fit(dfil)
            Mu.append(mu)
            Std.append(std)
            #axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
            #axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
    
            
# =============================================================================
#         for ax in axs.flat:
#             ax.tick_params(axis='both', which='major', labelsize=14)
#             ax.set_ylabel('counts (a.u.)', fontsize = 15) # Y label
#             ax.set_xlabel(r'Lattice Parameter ($\mathrm{\AA}$)', fontsize = 15) # X label
#             ax.set_yticks([])
#             ax.set_xlim(histrange)
#             
#         axs[0].xaxis.set_ticklabels([])
#         axs[1].xaxis.set_ticklabels([])
#         axs[0].yaxis.set_ticklabels([])
#         axs[1].yaxis.set_ticklabels([])
#         axs[2].yaxis.set_ticklabels([])
#         axs[0].set_xlabel("")
#         axs[1].set_xlabel("")
#         axs[0].set_ylabel("")
#         axs[2].set_ylabel("")
#         axs[0].yaxis.set_label_coords(-0.02,-0.40)
# 
#         axs[0].set_title(typesname[config_types[di]],fontsize=16)
# =============================================================================
        config_involved.append(typesname[config_types[di]])    
            
        #plt.show()
        
        Lgauss = np.concatenate((Lgauss,np.array(Mu).reshape(1,-1)),axis=0)
        Lgaussstd = np.concatenate((Lgaussstd,np.array(Std).reshape(1,-1)),axis=0)
     
    # plot type dependence   
    plotx = np.array([typexval[i] for i in config_types])
    plotxlab = [typextick[i] for i in config_types]
    
    scaalpha = 0.8
    scasize = 50
    plt.subplots(1,1)
    ax = plt.gca()
    ax.scatter(plotx-0.05,Lgauss[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx-0.05,Lgauss[:,0],yerr=Lgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
    ax.scatter(plotx,Lgauss[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx,Lgauss[:,1],yerr=Lgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
    ax.scatter(plotx+0.05,Lgauss[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx+0.05,Lgauss[:,2],yerr=Lgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
    plt.legend(prop={'size': 12})
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'Lattice Parameter ($\mathrm{\AA}$)', fontsize = 15) # Y label
    ax.set_xlabel('Br content', fontsize = 15) # X label
    ax.set_xticks(plotx)
    ax.set_xticklabels(plotxlab)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return Lgauss, Lgaussstd


def print_partition(typelib,config_types,brconc,halcounts):
    """
    Print the partition of the halide configurations.

    Args:
        typelib (list): The list of halide configurations.
        config_types (list): The configuration types.
        brconc (list): The bromine concentration.
        halcounts (list): The counts of halides.

    Returns:    
        None
    """

    #typesname = ["I6 Br0","I5 Br1","I4 Br2: cis","I4 Br2: trans","I3 Br3: fac",
    #             "I3 Br3: mer","I2 Br4: cis","I2 Br4: trans","I1 Br5","I0 Br6"]
    #typextick = ['0','1','2c','2t','3f','3m','4c','4t','5','6']
    config_types = list(config_types)
    #bincent = (Bins[1:]+Bins[:-1])/2
    
    title = f"Total: {halcounts[0]} I, {halcounts[1]} Br"
    
    types = np.zeros((10,)).astype(int)
    for i,t in enumerate(typelib):
        types[config_types[i]] = len(t)
    
    #concs = np.zeros((len(bincent),)).astype(int)
    #for i,t in enumerate(brbins):
    #    concs[i] = len(t)
    
    # plotting
    y1, y2, y3 = np.zeros((7,)),np.zeros((7,)),np.zeros((7,))
    y1[:2] = types[:2]
    y2[2:5] = types[[2,4,6]]
    y3[2:5] = types[[3,5,7]]
    y1[5:] = types[8:]
    
    x = ['0', '1', '2', '3', '4', '5', '6']
    # plot bars in stack manner
    plt.bar(x, y1, )
    plt.bar(x, y2, bottom=y1)
    plt.bar(x, y3, bottom=y1+y2)
    plt.title(title,fontsize=13)
    plt.xlabel("Br content",fontsize=12)
    plt.ylabel("Counts",fontsize=12)
    plt.legend(['pure', 'cis/fac', 'trans/mer'])
    plt.show()
    
    #plt.hist(bincent, bins=len(concs)*5,weights=concs, range=(min(Bins), max(Bins)))
    plt.hist(brconc, bins=50, range=[0,1])
    ax = plt.gca()
    plt.title(title,fontsize=13)
    ax.set_xlim([-0.1,1.1])
    plt.xlabel("Br content",fontsize=12)
    plt.ylabel("Counts",fontsize=12)
    plt.show()


def draw_halideconc_tilt_density_transient(Tconc, steps, concent, uniname, saveFigures, smoother = 0):
    """ 
    Isolate tilting pattern wrt. the local halide concentration.  

    Args:
        Tconc (list): The tilting data of each concentraion level.
        steps (list): The x-axis steps.
        concent (list): The halide concentrations.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): The smoothing window size.

    Returns:
        tuple: The halide concentrations, steps, and tilting data.
    """
    
    from pdyna.structural import periodicity_fold
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    fig_name=f"tilt_halideconc_density_evo_{uniname}.png"
    fig_name1=f"tilt_halideconc_density_tcp_ev0_{uniname}.png"
    
    awside = 5
    
    Tarr = []
    for ti, T in enumerate(Tconc):
        Tmax = []
        for i in range(T.shape[0]):
            fi = max(0,i-awside)
            ff = min(T.shape[0],i+awside)
            temp = T[list(range(fi,ff)),:,:]
            #temp1 = temp[np.where(np.logical_and(temp<20,temp>-20))]
            fitted = np.array(compute_tilt_density(temp,method='curve'))
            Tmax.append(fitted)
        Tmax = np.array(Tmax)
        Tarr.append(Tmax)
    Tarr = np.array(Tarr)
    
    if smoother != 0:
        sgw = smoother
        if sgw<5: sgw = 5
        if sgw%2==0: sgw+=1
        for i in range(Tarr.shape[0]):
            for j in range(3):
                Tarr[i,:,j] = savitzky_golay(Tarr[i,:,j],window_size=sgw)
    
    lw = 1.4
    fig, axs = plt.subplots(3,1,figsize=(5.1,3.5),sharey=True,sharex=True)
    scalecolor = (np.array(concent)-np.amin(concent))/(np.amax(concent)-np.amin(concent))
    colors = plt.cm.viridis(scalecolor)
    for ti in range(Tarr.shape[0]):
        axs[0].plot(steps,Tarr[ti,:,0],alpha=0.7,linewidth=lw, color=colors[ti])
        axs[1].plot(steps,Tarr[ti,:,1],alpha=0.7,linewidth=lw, color=colors[ti])
        axs[2].plot(steps,Tarr[ti,:,2],alpha=0.7,linewidth=lw, color=colors[ti])  

    axs[2].set_xlabel('Temperature (K)', fontsize=14)
    axs[1].set_ylabel('Tilting (deg)', fontsize=14)
    axs[0].set_title("Tilting with Concentrations", fontsize=15)
    if steps[0]>steps[1]:
        axs[2].set_xlim([max(steps),min(steps)])
    else:
        axs[2].set_xlim([min(steps),max(steps)])
    
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.05)
    
    norm = Normalize(vmin=np.amin(concent), vmax=np.amax(concent))
    #clb = plt.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'),ax=axs[1], pad=0.025)
    clb = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=axs, shrink=0.8, pad=0.025)
    clb.ax.set_ylabel('Br Conc.',rotation=270,labelpad=10)
    #clb.ax.set_title('Halide Concentration')
 
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return concent, steps, Tarr
 

def draw_halideconc_tilt_density(Tconc, brconc, concent, uniname, saveFigures, corr_vals = None, n_bins = 100, symm_n_fold = 4):
    """ 
    Isolate tilting pattern wrt. the local halide concentration.  

    Args:
        Tconc (list): The tilting data of each concentraion level.
        brconc (list): The bromine concentration.
        concent (list): The halide concentrations.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        corr_vals (list): The correlation values.
        n_bins (int): Number of bins for the histogram.
        symm_n_fold (int): The symmetry fold.

    Returns:
        tuple: The fitted tilt angles
    """
    
    from pdyna.structural import periodicity_fold
    fig_name=f"tilt_halideconc_density_{uniname}.png"
    fig_name1=f"tilt_halideconc_density_tcp_{uniname}.png"
    
    if symm_n_fold == 2:
        hrange = [-90,90]
        tlabel = [-90,-60,-30,0,30,60,90]
    elif symm_n_fold == 4:
        hrange = [-45,45]
        tlabel = [-45,-30,-15,0,15,30,45]
    elif symm_n_fold == 8:
        hrange = [0,45]
        tlabel = [0,15,30,45]
    
    maxs = np.empty((0,3))
    for ti, T in enumerate(Tconc):
        
        T = periodicity_fold(T,n_fold=symm_n_fold)
        
        T_a = T[:,:,0].reshape((T.shape[0]*T.shape[1]))
        T_b = T[:,:,1].reshape((T.shape[0]*T.shape[1]))
        T_c = T[:,:,2].reshape((T.shape[0]*T.shape[1]))
        tup_T = (T_a,T_b,T_c)
        
        figs, axs = plt.subplots(3, 1)
        labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
        colors = ["C0","C1","C2"]
        for i in range(3):
            y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=hrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i])
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes)
            
        for ax in axs.flat:
            #ax.set(xlabel=r'Tilt Angle ($\degree$)', ylabel='Counts (a.u.)')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlim(hrange)
            ax.set_xticks(tlabel)
            ax.set_yticks([])
        
        axs[2].set_xlabel(r'Tilt Angle ($\degree$)', fontsize=15)
        axs[1].set_ylabel('Counts (a.u.)', fontsize=15)
        
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[1].set_xlabel("")
        axs[2].set_ylabel("")
        
        axs[0].set_title('Br concentration: '+str(round(concent[ti],4)),fontsize=16)  
            
        plt.show()
        
        m1 = np.array(compute_tilt_density(T,method='gaussian')).reshape(1,-1)
        maxs = np.concatenate((maxs,m1),axis=0)
    
    # plot type dependence   
    plotx = concent
    
    scaalpha = 0.9
    scasize = 50
    
# =============================================================================
#     plt.subplots(1,1)
#     ax = plt.gca()
#     ax.scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
#     ax.scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
#     ax.scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
#     plt.legend(prop={'size': 12})
#     
#     ax.tick_params(axis='both', which='major', labelsize=14)
#     ax.set_ylabel('Tilting (deg)', fontsize = 15) # Y label
#     ax.set_xlabel('Br content', fontsize = 15) # X label
#     ax.set_ylim(bottom=0)
# =============================================================================
    
    fig, axs = plt.subplots(2,1,figsize=(5.1,4.2),gridspec_kw={'height_ratios':[3, 1]},sharex=False)

    axs[0].scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    axs[0].scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    axs[0].scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    axs[0].legend(prop={'size': 13.2},frameon = True)
    
    axs[1].hist(brconc, bins=100, range=[0,1], color='grey')
    
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
    axs[1].set_ylabel('Count', fontsize = 15) 
    axs[1].set_xlabel('Br content', fontsize = 15) # X label
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[0].set_ylim(bottom=0)
    axs[1].set_xlim(axs[0].get_xlim())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    if not (corr_vals is None):
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable
        scasize = 80
        cate = if_arrays_are_different(corr_vals)
        cmnames = ['summer','cool_r','winter']
            
        if cate[0] == 3: # all TCP values are in different ranges and thus plot with three colorbars
            
            norm1 = Normalize(vmin=np.amin(corr_vals[:,0]), vmax=np.amax(corr_vals[:,0]))  
            norm2 = Normalize(vmin=np.amin(corr_vals[:,1]), vmax=np.amax(corr_vals[:,1]))  
            norm3 = Normalize(vmin=np.amin(corr_vals[:,2]), vmax=np.amax(corr_vals[:,2]))

            fig, axs = plt.subplots(2,1,figsize=(8.6,6.0),gridspec_kw={'height_ratios':[2.6, 0.8]},sharex=False)
            
            axs[0].scatter(plotx,maxs[:,0],alpha=scaalpha,s=scasize,c=corr_vals[:,0],cmap=cmnames[0],norm=norm1)
            axs[0].scatter(plotx,maxs[:,1],alpha=scaalpha,s=scasize,c=corr_vals[:,1],cmap=cmnames[1],norm=norm2)
            axs[0].scatter(plotx,maxs[:,2],alpha=scaalpha,s=scasize,c=corr_vals[:,2],cmap=cmnames[2],norm=norm3)
            
            # Add colorbars
            clb1 = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmnames[0]),ax=axs[0], pad=-0.05)
            clb2 = plt.colorbar(ScalarMappable(norm=norm2, cmap=cmnames[1]),ax=axs[0], pad=-0.03)
            clb3 = plt.colorbar(ScalarMappable(norm=norm3, cmap=cmnames[2]),ax=axs[0], pad=0.025)
            
            clb1.set_label(label='TCP (a.u.)', size=13)
            
            axs[1].hist(brconc, bins=100, range=[0,1], color='grey')
            
            axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
            axs[1].set_ylabel('Count', fontsize = 15) 
            axs[1].set_xlabel('Br content', fontsize = 15) # X label

            #axs[0].set_xticks(typexval)
            #axs[0].set_xticklabels(typextick)

            axs[1].set_yticks([])
            axs[1].set_yticklabels([])
            axs[0].set_xticklabels([])
            axs[0].set_ylim(bottom=0)
            axs[1].set_xlim(axs[0].get_xlim())
            
            newwid = axs[0].get_position().width
            pax1 = axs[1].get_position()
            new_position = [pax1.x0, pax1.y0, newwid, pax1.height+0.03]
            axs[1].set_position(new_position)

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0.15, hspace=0.15)
            
            if saveFigures:
                plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
            plt.show()

        elif cate[0] == 2: # two axes have similar TCP and thus plot with two colorbars
            
            combaxes = list(cate[1])
            soleaxis = [i for i in [0,1,2] if i not in combaxes]
            norm1 = Normalize(vmin=np.amin(corr_vals[:,combaxes]), vmax=np.amax(corr_vals[:,combaxes]))  
            norm2 = Normalize(vmin=np.amin(corr_vals[:,soleaxis]), vmax=np.amax(corr_vals[:,soleaxis]))  

            fig, axs = plt.subplots(2,1,figsize=(8.0,6.0),gridspec_kw={'height_ratios':[2.6, 0.8]},sharex=False)
            
            axs[0].scatter(plotx,maxs[:,soleaxis[0]],alpha=scaalpha,s=scasize,c=corr_vals[:,soleaxis[0]],cmap=cmnames[0],norm=norm2)
            axs[0].scatter(plotx,maxs[:,combaxes[0]],alpha=scaalpha,s=scasize,c=corr_vals[:,combaxes[0]],cmap=cmnames[1],norm=norm1)
            axs[0].scatter(plotx,maxs[:,combaxes[1]],alpha=scaalpha,s=scasize,c=corr_vals[:,combaxes[1]],cmap=cmnames[1],norm=norm1)
            
            # Add colorbars
            clb1 = plt.colorbar(ScalarMappable(norm=norm2, cmap=cmnames[0]),ax=axs[0], pad=-0.03)
            clb2 = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmnames[1]),ax=axs[0], pad=0.025)
            
            clb1.set_label(label='TCP (a.u.)', size=13)
            
            axs[1].hist(brconc, bins=100, range=[0,1], color='grey')
            
            axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
            axs[1].set_ylabel('Count', fontsize = 15) 
            axs[1].set_xlabel('Br content', fontsize = 15) # X label

            #axs[0].set_xticks(typexval)
            #axs[0].set_xticklabels(typextick)

            axs[1].set_yticks([])
            axs[1].set_yticklabels([])
            axs[0].set_xticklabels([])
            axs[0].set_ylim(bottom=0)
            axs[1].set_xlim(axs[0].get_xlim())
            
            newwid = axs[0].get_position().width
            pax1 = axs[1].get_position()
            new_position = [pax1.x0, pax1.y0, newwid, pax1.height+0.05]
            axs[1].set_position(new_position)

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0.15, hspace=0.15)
            
            if saveFigures:
                plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
            plt.show()
            
        elif cate[0] == 1: # all TCP values are in the same range
        
            norm1 = Normalize(vmin=np.amin(corr_vals), vmax=np.amax(corr_vals))  

            fig, axs = plt.subplots(2,1,figsize=(7.4,6.0),gridspec_kw={'height_ratios':[2.6, 0.8]},sharex=False)
            
            axs[0].scatter(plotx,maxs[:,0],alpha=scaalpha,s=scasize,c=corr_vals[:,0],cmap=cmnames[0],norm=norm1)
            axs[0].scatter(plotx,maxs[:,1],alpha=scaalpha,s=scasize,c=corr_vals[:,1],cmap=cmnames[0],norm=norm1)
            axs[0].scatter(plotx,maxs[:,2],alpha=scaalpha,s=scasize,c=corr_vals[:,2],cmap=cmnames[0],norm=norm1)
            
            # Add colorbars
            clb1 = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmnames[0]),ax=axs[0], pad=0.025)
            
            clb1.set_label(label='TCP (a.u.)', size=13)
            
            axs[1].hist(brconc, bins=100, range=[0,1], color='grey')
            
            axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
            axs[1].set_ylabel('Count', fontsize = 15) 
            axs[1].set_xlabel('Br content', fontsize = 15) # X label

            #axs[0].set_xticks(typexval)
            #axs[0].set_xticklabels(typextick)

            axs[1].set_yticks([])
            axs[1].set_yticklabels([])
            axs[0].set_xticklabels([])
            axs[0].set_ylim(bottom=0)
            axs[1].set_xlim(axs[0].get_xlim())
            
            newwid = axs[0].get_position().width
            pax1 = axs[1].get_position()
            new_position = [pax1.x0, pax1.y0, newwid, pax1.height+0.03]
            axs[1].set_position(new_position)

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0.15, hspace=0.15)
            
            if saveFigures:
                plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
            plt.show()
    
    return maxs


def draw_halideconc_dist_density(Dconc, concent, uniname, saveFigures, n_bins = 100, xrange = [0,0.5]):
    """ 
    Isolate distortion mode wrt. the local halide concentration.  

    Args:   
        Dconc (list): The distortion data of each concentraion level.
        concent (list): The halide concentrations.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        xrange (list): The range for the histogram.

    Returns:
        tuple: The fitted distortion, distortion standard deviation.
    """
    
    if Dconc[0].shape[2] == 4:
        fig_name=f"dist_halideconc_density_{uniname}.png"
        
        Dgauss = np.empty((0,4))
        Dgaussstd = np.empty((0,4))
        
        for di, D in enumerate(Dconc):
            if D.ndim == 3:
                D = D.reshape(D.shape[0]*D.shape[1],4)

            Mu = []
            Std = []
            for i in range(4):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)
                #axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
                #axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)

            Dgauss = np.concatenate((Dgauss,np.array(Mu).reshape(1,-1)),axis=0)
            Dgaussstd = np.concatenate((Dgaussstd,np.array(Std).reshape(1,-1)),axis=0)
         
        # plot type dependence   
        plotx = np.array(concent)
        
        dist_gap = 0.0005
        
        scaalpha = 0.8
        scasize = 50
        plt.subplots(1,1)
        ax = plt.gca()
        ax.scatter(plotx-3*dist_gap,Dgauss[:,0],label='Eg',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx-3*dist_gap,Dgauss[:,0],yerr=Dgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx-1*dist_gap,Dgauss[:,1],label='T2g',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx-1*dist_gap,Dgauss[:,1],yerr=Dgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx+1*dist_gap,Dgauss[:,2],label='T1u',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx+1*dist_gap,Dgauss[:,2],yerr=Dgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx+3*dist_gap,Dgauss[:,3],label='T2u',alpha=scaalpha,s=scasize)
        ax.errorbar(plotx+3*dist_gap,Dgauss[:,3],yerr=Dgaussstd[:,3],fmt='o',solid_capstyle='projecting', capsize=5)
        plt.legend(prop={'size': 12})
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(r'Distortion', fontsize = 15) # Y label
        ax.set_xlabel('Br content', fontsize = 15) # X label

        ax.set_ylim(bottom=0)
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
        
    elif Dconc[0].shape[2] == 3:
        fig_name=f"distB_halideconc_density_{uniname}.png"
        
        Dgauss = np.empty((0,3))
        Dgaussstd = np.empty((0,3))
        
        labels = ["B100","B110","B111"]
        
        for di, D in enumerate(Dconc):
            if D.ndim == 3:
                D = D.reshape(D.shape[0]*D.shape[1],3)

            Mu = []
            Std = []
            for i in range(3):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)
                #axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
                #axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)

            Dgauss = np.concatenate((Dgauss,np.array(Mu).reshape(1,-1)),axis=0)
            Dgaussstd = np.concatenate((Dgaussstd,np.array(Std).reshape(1,-1)),axis=0)
         
        # plot type dependence   
        plotx = np.array(concent)
        
        dist_gap = 0.0005
        
        scaalpha = 0.8
        scasize = 50
        plt.subplots(1,1)
        ax = plt.gca()
        ax.scatter(plotx-2*dist_gap,Dgauss[:,0],label=labels[0],alpha=scaalpha,s=scasize)
        ax.errorbar(plotx-2*dist_gap,Dgauss[:,0],yerr=Dgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx,Dgauss[:,1],label=labels[1],alpha=scaalpha,s=scasize)
        ax.errorbar(plotx,Dgauss[:,1],yerr=Dgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx+2*dist_gap,Dgauss[:,2],label=labels[2],alpha=scaalpha,s=scasize)
        ax.errorbar(plotx+2*dist_gap,Dgauss[:,2],yerr=Dgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
        plt.legend(prop={'size': 12})
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(r'Distortion', fontsize = 15) # Y label
        ax.set_xlabel('Br content', fontsize = 15) # X label

        ax.set_ylim(bottom=0)
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
    
    return Dgauss, Dgaussstd


def draw_halideconc_lat_density(Lconc, concent, uniname, saveFigures, n_bins = 100):
    """ 
    Isolate lattice parameter wrt. the local halide concentration.  

    Args:
        Lconc (list): The lattice data of each concentraion level.
        concent (list): The halide concentrations.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        tuple: The fitted lattice parameters, lattice parameter standard deviation.
    """
    
    fig_name=f"lat_halideconc_density_{uniname}.png"
    
    Lgauss = np.empty((0,3))
    Lgaussstd = np.empty((0,3))
    
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2"]
    
    for di, L in enumerate(Lconc):
        if L.ndim == 3:
            L = L.reshape(L.shape[0]*L.shape[1],3)
        
# =============================================================================
#         figs, axs = plt.subplots(3, 1)
#         
#         histranges = np.zeros((3,2))
#         for i in range(3):
#             histranges[i,:] = [np.quantile(L[:,i], 0.02),np.quantile(L[:,i], 0.98)]
#             
#         histrange = np.zeros((2,))
#         ra = np.amax(histranges[:,1])-np.amin(histranges[:,0])
#         histrange[0] = np.amin(histranges[:,0])-ra*0.2
#         histrange[1] = np.amax(histranges[:,1])+ra*0.2
# 
#         for i in range(3):
#             
#             y,binEdges=np.histogram(L[:,i],bins=n_bins,range=histrange)
#             bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#             axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
#             axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
# =============================================================================
        
        Mu = []
        Std = []
        for i in range(3):
            dfil = L[:,i].copy()
            dfil = dfil[~np.isnan(dfil)]
            mu, std = norm.fit(dfil)
            Mu.append(mu)
            Std.append(std)
            #axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
            #axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
    
            
# =============================================================================
#         for ax in axs.flat:
#             ax.tick_params(axis='both', which='major', labelsize=14)
#             ax.set_ylabel('counts (a.u.)', fontsize = 15) # Y label
#             ax.set_xlabel(r'Lattice Parameter ($\mathrm{\AA}$)', fontsize = 15) # X label
#             ax.set_yticks([])
#             ax.set_xlim(histrange)
#             
#         axs[0].xaxis.set_ticklabels([])
#         axs[1].xaxis.set_ticklabels([])
#         axs[0].yaxis.set_ticklabels([])
#         axs[1].yaxis.set_ticklabels([])
#         axs[2].yaxis.set_ticklabels([])
#         axs[0].set_xlabel("")
#         axs[1].set_xlabel("")
#         axs[0].set_ylabel("")
#         axs[2].set_ylabel("")
#         #axs[1].yaxis.set_label_coords(-0.02,-0.40)
# 
#         axs[0].set_title('Br concentration: '+str(round(concent[di],4)),fontsize=16)  
#             
#         plt.show()
# =============================================================================
        
        Lgauss = np.concatenate((Lgauss,np.array(Mu).reshape(1,-1)),axis=0)
        Lgaussstd = np.concatenate((Lgaussstd,np.array(Std).reshape(1,-1)),axis=0)
     
    # plot type dependence   
    plotx = np.array(concent)
    
    dist_gap = 0.0005
    
    scaalpha = 0.8
    scasize = 50
    plt.subplots(1,1)
    ax = plt.gca()
    ax.scatter(plotx-2*dist_gap,Lgauss[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx-2*dist_gap,Lgauss[:,0],yerr=Lgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
    ax.scatter(plotx,Lgauss[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx,Lgauss[:,1],yerr=Lgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
    ax.scatter(plotx+2*dist_gap,Lgauss[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx+2*dist_gap,Lgauss[:,2],yerr=Lgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
    plt.legend(prop={'size': 12})
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'Lattice Parameter ($\mathrm{\AA}$)', fontsize = 15) # Y label
    ax.set_xlabel('Br content', fontsize = 15) # X label
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return Lgauss, Lgaussstd


def draw_hetero_tilt_density(Tcls, TCNcls, typelib, uniname, saveFigures, corr_vals = None, n_bins = 100, symm_n_fold = 4):
    """ 
    Isolate tilting pattern wrt. the local halide configuration.  

    Args:
        Tcls (list): The tilt data of each configuration.
        TCNcls (list): The tilt data of each configuration.
        typelib (str): The type of tilt.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        corr_vals (list): The correlation values.
        n_bins (int): Number of bins for the histogram.
        symm_n_fold (int): The symmetry fold.

    Returns:
        list of floats: The maximum tilt values.
    """
    
    from pdyna.structural import periodicity_fold
    fig_name=f"tilt_hetero_density_{uniname}.png"
    fig_name1=f"tilt_hetero_density_tcp_{uniname}.png"
    
    if symm_n_fold == 2:
        hrange = [-90,90]
        tlabel = [-90,-60,-30,0,30,60,90]
    elif symm_n_fold == 4:
        hrange = [-45,45]
        tlabel = [-45,-30,-15,0,15,30,45]
    elif symm_n_fold == 8:
        hrange = [0,45]
        tlabel = [0,15,30,45]
    
    typesname = ["bulk","grain boundary","grain"]
    typexval = [0,1,2]
    typextick = ["bulk","grain boundary","grain"]
    fill_alpha = 0.5
    
    if not TCNcls is None:
        C = []
        if TCNcls[0].shape[2] == 3:
            for temp in TCNcls:
                C.append((temp[:,:,0].reshape((-1,)),
                          temp[:,:,1].reshape((-1,)),
                          temp[:,:,2].reshape((-1,))))
        elif TCNcls[0].shape[2] == 6:
            for temp in TCNcls:
                C.append((np.concatenate((temp[:,:,0],temp[:,:,1]),axis=0).reshape((-1,)),
                    np.concatenate((temp[:,:,2],temp[:,:,3]),axis=0).reshape((-1,)),
                    np.concatenate((temp[:,:,4],temp[:,:,5]),axis=0).reshape((-1,))))
    
    maxs = np.empty((0,3))
    for ti, T in enumerate(Tcls):
        
        T = periodicity_fold(T,n_fold=symm_n_fold)
        
        T_a = T[:,:,0].reshape((T.shape[0]*T.shape[1]))
        T_b = T[:,:,1].reshape((T.shape[0]*T.shape[1]))
        T_c = T[:,:,2].reshape((T.shape[0]*T.shape[1]))
        tup_T = (T_a,T_b,T_c)
        assert len(tup_T) == 3
        
        figs, axs = plt.subplots(3, 1)
        labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
        colors = ["C0","C1","C2"]
        for i in range(3):
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes)
            
            y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=hrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yt=y/max(y)
            axs[i].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 2.4)
            #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)

            y,binEdges=np.histogram(C[ti][i],bins=n_bins,range=hrange) 
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yc=y/max(y)
            yy=yt*yc

            axs[i].fill_between(bincenters, yy, 0, facecolor = colors[i], alpha=fill_alpha, interpolate=True)
            
        for ax in axs.flat:
            #ax.set(xlabel=r'Tilt Angle ($\degree$)', ylabel='Counts (a.u.)')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlim(hrange)
            ax.set_xticks(tlabel)
            ax.set_yticks([])
        
        axs[2].set_xlabel(r'Tilt Angle ($\degree$)', fontsize=15)
        axs[1].set_ylabel('Counts (a.u.)', fontsize=15)
        
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[1].set_xlabel("")
        axs[2].set_ylabel("")
        
        axs[0].set_title(typesname[ti],fontsize=16)

        plt.show()
        
        m1 = np.array(compute_tilt_density(T,method='gaussian',plot_fitting=True,corr_vals=list(corr_vals[:,0]))).reshape(1,-1)
        maxs = np.concatenate((maxs,m1),axis=0)
    
    # plot type dependence   
    plotx = np.array([0,1,2])
    plotxlab = typextick
    histcolors = ['grey','darkred','darkblue']
    
    scaalpha = 0.9
    scasize = 80
    
# =============================================================================
#     plt.subplots(1,1)
#     ax = plt.gca()
#     ax.scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
#     ax.scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
#     ax.scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
#     plt.legend(prop={'size': 12})
#     
#     ax.tick_params(axis='both', which='major', labelsize=14)
#     ax.set_ylabel('Tilting (deg)', fontsize = 15) # Y label
#     ax.set_xlabel('Br content', fontsize = 15) # X label
#     ax.set_xticks(plotx)
#     ax.set_xticklabels(plotxlab)
#     ax.set_ylim(bottom=0)
# =============================================================================
    
    
    config_types = [0,1,2]
    types = np.zeros((3,)).astype(int)
    for i,t in enumerate(typelib):
        types[config_types[i]] = len(t)
    
    xts = typextick
    
    # plot bars in stack manner
    
    fig, axs = plt.subplots(2,1,figsize=(5.1,4.2),gridspec_kw={'height_ratios':[3, 1]},sharex=False)

    axs[0].scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    axs[0].scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    axs[0].scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    axs[0].legend(loc=2,prop={'size': 13.2},frameon = True)
    axs[0].set_xlim([-0.5,2.5])
    
    axs[1].bar(xts, types, width=0.5, color=histcolors[0])
    axs[1].text(0, types[0], str(types[0]), fontsize=11,horizontalalignment='center',verticalalignment='bottom')
    axs[1].text(1, types[1], str(types[1]), fontsize=11,horizontalalignment='center',verticalalignment='bottom')
    axs[1].text(2, types[2], str(types[2]), fontsize=11,horizontalalignment='center',verticalalignment='bottom')
    if max(types)/min(types) > 15:
        axs[1].set_yscale("log")
        axs[1].set_ylabel('Log count', fontsize = 15) 
        axs[1].set_ylim([axs[1].get_ylim()[0],axs[1].get_ylim()[1]*3])
    else:
        axs[1].set_ylabel('Count', fontsize = 15) 
        axs[1].set_ylim([axs[1].get_ylim()[0],axs[1].get_ylim()[1]*1.2])
    
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
    #axs[1].set_xlabel('Br content', fontsize = 15) # X label
    axs[0].set_xticks(plotx)
    axs[0].set_xticklabels([])
    #axs[0].set_xticks(typexval)
    #axs[0].set_xticklabels(typextick)
    #for tick_label, color in zip(axs[0].get_xticklabels(), histcolors):
    #    tick_label.set_color(color)
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[0].set_ylim(bottom=0)
    axs[1].set_xlim(axs[0].get_xlim())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.05)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    if not (corr_vals is None):
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable
        scasize = 150
        cmname = "coolwarm"
            
        norm1 = Normalize(vmin=-np.amax(np.abs(corr_vals)), vmax=np.amax(np.abs(corr_vals)))  

        fig, axs = plt.subplots(2,1,figsize=(7.4,6.0),gridspec_kw={'height_ratios':[2.6, 0.8]},sharex=False)
        
        axs[0].scatter(plotx,maxs[:,0],alpha=scaalpha,s=scasize,c=corr_vals[:,0],cmap=cmname,norm=norm1)
        axs[0].scatter(plotx,maxs[:,1],alpha=scaalpha,s=scasize,c=corr_vals[:,1],cmap=cmname,norm=norm1)
        axs[0].scatter(plotx,maxs[:,2],alpha=scaalpha,s=scasize,c=corr_vals[:,2],cmap=cmname,norm=norm1)
        axs[0].set_xlim([-0.5,2.5])
        
        # Add colorbars
        clb1 = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmname),ax=axs[0], pad=0.025)
        
        clb1.set_label(label='TCP (a.u.)', size=13)
        
        axs[1].bar(xts, types, width=0.5, color=histcolors[0])
        axs[1].text(0, types[0], str(types[0]), fontsize=11,horizontalalignment='center',verticalalignment='bottom')
        axs[1].text(1, types[1], str(types[1]), fontsize=11,horizontalalignment='center',verticalalignment='bottom')
        axs[1].text(2, types[2], str(types[2]), fontsize=11,horizontalalignment='center',verticalalignment='bottom')
        if max(types)/min(types) > 15:
            axs[1].set_yscale("log")
            axs[1].set_ylabel('Log count', fontsize = 15) 
            axs[1].set_ylim([axs[1].get_ylim()[0],axs[1].get_ylim()[1]*3])
        else:
            axs[1].set_ylabel('Count', fontsize = 15) 
            axs[1].set_ylim([axs[1].get_ylim()[0],axs[1].get_ylim()[1]*1.2])
               
        axs[0].tick_params(axis='both', which='major', labelsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        axs[0].set_ylabel('Tilting (deg)', fontsize = 15) 
        
        #axs[1].set_xlabel('Br content', fontsize = 15) # X label
        axs[0].set_xticks(plotx)
        axs[0].set_xticklabels([])
        #axs[0].set_xticks(typexval)
        #axs[0].set_xticklabels(typextick)
        #for tick_label, color in zip(axs[0].get_xticklabels(), histcolors):
        #    tick_label.set_color(color)
        axs[1].set_yticks([])
        axs[1].set_yticklabels([])
        axs[0].set_ylim(bottom=0)
        axs[1].set_xlim(axs[0].get_xlim())
        
        newwid = axs[0].get_position().width
        pax1 = axs[1].get_position()
        new_position = [pax1.x0, pax1.y0, newwid, pax1.height+0.03]
        axs[1].set_position(new_position)

        #plt.tight_layout()
        #plt.subplots_adjust(wspace=0.15, hspace=0.15)
        
        if saveFigures:
            plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
        plt.show()
        
    return maxs


def draw_hetero_dist_density(Dcls, uniname, saveFigures, n_bins = 100, xrange = [0,0.5]):
    """ 
    Isolate distortion mode wrt. the local halide configuration.  

    Args:
        Dcls (list): The distortion data of each configuration.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        xrange (list): The range for the histogram.

    Returns:
        tuple: The fitted distortion, distortion standard deviation.
    """
    from sklearn.decomposition import PCA
    
    typesname = ["bulk","grain boundary","grain"]
    typexval = [0,1,2]
    typextick = ["bulk","grain boundary","grain"]
    
    if Dcls[0].shape[-1] == 4:
        fig_name=f"dist_hetero_density_{uniname}.png"
        
        Dgauss = np.empty((0,4))
        Dgaussstd = np.empty((0,4))
        
        Dlist = []
        for di, D in enumerate(Dcls):
            if D.ndim == 3:
                D = D.reshape(-1,4)
            Dlist.append(D)
            #figs, axs = plt.subplots(4, 1)
            labels = ["Eg","T2g","T1u","T2u"]
            colors = ["C3","C4","C5","C6"]
            for i in range(4):
                y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                #axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
                #axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
            
            Mu = []
            Std = []
            for i in range(4):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)

            Dgauss = np.concatenate((Dgauss,np.array(Mu).reshape(1,-1)),axis=0)
            Dgaussstd = np.concatenate((Dgaussstd,np.array(Std).reshape(1,-1)),axis=0)
         
        # plot type dependence   
        plotx = np.array([0,1,2])
        plotxlab = typesname
        
        Ddelta = Dgauss-Dgauss[0,:]
        
        scaalpha = 0.8
        scasize = 50
        plt.subplots(1,1)
        ax = plt.gca()
        ax.scatter(plotx,Ddelta[:,0],label='Eg',alpha=scaalpha,s=scasize)
        #ax.errorbar(plotx-0.075,Dgauss[:,0],yerr=Dgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx,Ddelta[:,1],label='T2g',alpha=scaalpha,s=scasize)
        #ax.errorbar(plotx-0.025,Dgauss[:,1],yerr=Dgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx,Ddelta[:,2],label='T1u',alpha=scaalpha,s=scasize)
        #ax.errorbar(plotx+0.025,Dgauss[:,2],yerr=Dgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx,Ddelta[:,3],label='T2u',alpha=scaalpha,s=scasize)
        #ax.errorbar(plotx+0.075,Dgauss[:,3],yerr=Dgaussstd[:,3],fmt='o',solid_capstyle='projecting', capsize=5)
        plt.legend(prop={'size': 12},frameon=True)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(r'$\Delta$ Distortion', fontsize = 15) # Y label
        #ax.set_xlabel('Br content', fontsize = 15) # X label
        ax.set_xticks(plotx)
        ax.set_xticklabels(plotxlab)
        plt.axhline(0,linestyle='--',linewidth=2,color='k')
        #ax.set_ylim(bottom=0)
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
        
        # PCA on Distortions
        plt.subplots(1,1)
        ax = plt.gca()
        for di, dtemp in enumerate(Dlist):
            pca=PCA(n_components=2) 
            Y=pca.fit_transform(dtemp)
            
            ax.scatter(Y[:,0],Y[:,1],s=10,alpha=0.4,label=typesname[di])
            
        plt.legend(prop={'size': 12},frameon=True)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(r'PC1', fontsize = 15) # Y label
        ax.set_ylabel(r'PC2', fontsize = 15) # Y label
        
    elif Dcls[0].shape[-1] == 3:
        fig_name=f"distB_hetero_density_{uniname}.png"
        
        Dgauss = np.empty((0,3))
        Dgaussstd = np.empty((0,3))
        
        Dlist = []
        for di, D in enumerate(Dcls):
            if D.ndim == 3:
                D = D.reshape(-1,3)
            Dlist.append(D)
            #figs, axs = plt.subplots(4, 1)
            labels = ["B100","B110","B111"]
            colors = ["C3","C4","C5","C6"]
            for i in range(3):
                y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                #axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
                #axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
            
            Mu = []
            Std = []
            for i in range(3):
                dfil = D[:,i].copy()
                dfil = dfil[~np.isnan(dfil)]
                mu, std = norm.fit(dfil)
                Mu.append(mu)
                Std.append(std)

            Dgauss = np.concatenate((Dgauss,np.array(Mu).reshape(1,-1)),axis=0)
            Dgaussstd = np.concatenate((Dgaussstd,np.array(Std).reshape(1,-1)),axis=0)
         
        # plot type dependence   
        plotx = np.array([0,1,2])
        plotxlab = typesname
        
        Ddelta = Dgauss-Dgauss[0,:]
        
        scaalpha = 0.8
        scasize = 50
        plt.subplots(1,1)
        ax = plt.gca()
        ax.scatter(plotx,Ddelta[:,0],label=labels[0],alpha=scaalpha,s=scasize)
        #ax.errorbar(plotx-0.075,Dgauss[:,0],yerr=Dgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx,Ddelta[:,1],label=labels[1],alpha=scaalpha,s=scasize)
        #ax.errorbar(plotx-0.025,Dgauss[:,1],yerr=Dgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
        ax.scatter(plotx,Ddelta[:,2],label=labels[2],alpha=scaalpha,s=scasize)
        #ax.errorbar(plotx+0.025,Dgauss[:,2],yerr=Dgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
        plt.legend(prop={'size': 12},frameon=True)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(r'$\Delta$ Distortion', fontsize = 15) # Y label
        #ax.set_xlabel('Br content', fontsize = 15) # X label
        ax.set_xticks(plotx)
        ax.set_xticklabels(plotxlab)
        plt.axhline(0,linestyle='--',linewidth=2,color='k')
        #ax.set_ylim(bottom=0)
        
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        plt.show()
        
        # PCA on Distortions
        from sklearn.decomposition import PCA
        
        plt.subplots(1,1)
        ax = plt.gca()
        for di, dtemp in enumerate(Dlist):
            pca=PCA(n_components=2) 
            Y=pca.fit_transform(dtemp)
            
            ax.scatter(Y[:,0],Y[:,1],s=10,alpha=0.4,label=typesname[di])
            
        plt.legend(prop={'size': 12},frameon=True)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(r'PC1', fontsize = 15) # Y label
        ax.set_ylabel(r'PC2', fontsize = 15) # Y label

    return Dgauss, Dgaussstd


def draw_hetero_lat_density(Lcls, uniname, saveFigures, n_bins = 100):
    """ 
    Isolate distortion mode wrt. the local halide configuration.  

    Args:
        Lcls (list): The lattice data of each configuration.
        uniname (str): The user-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        tuple: The fitted lattice parameters, lattice parameter standard deviation.
    """
    
    fig_name=f"lat_octatype_density_{uniname}.png"
    
    typesname = ["bulk","grain boundary","grain"]
    typexval = [0,1,2]
    typextick = ["bulk","grain boundary","grain"]
    
    Lgauss = np.empty((0,3))
    Lgaussstd = np.empty((0,3))
    
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2"]
    
    for di, L in enumerate(Lcls):
        if L.ndim == 3:
            L = L.reshape(L.shape[0]*L.shape[1],3)
        
        Mu = []
        Std = []
        for i in range(3):
            dfil = L[:,i].copy()
            dfil = dfil[~np.isnan(dfil)]
            mu, std = norm.fit(dfil)
            Mu.append(mu)
            Std.append(std)
            #axs[i].text(0.852, 0.77, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
            #axs[i].text(0.878, 0.45, 'SD: %.4f' % std, horizontalalignment='center', fontsize=12, verticalalignment='center', transform=axs[i].transAxes)
    
        
        Lgauss = np.concatenate((Lgauss,np.array(Mu).reshape(1,-1)),axis=0)
        Lgaussstd = np.concatenate((Lgaussstd,np.array(Std).reshape(1,-1)),axis=0)
     
    # plot type dependence   
    plotx = np.array([0,1,2])
    plotxlab = typextick
    
    scaalpha = 0.8
    scasize = 50
    plt.subplots(1,1)
    ax = plt.gca()
    ax.scatter(plotx-0.05,Lgauss[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx-0.05,Lgauss[:,0],yerr=Lgaussstd[:,0],fmt='o',solid_capstyle='projecting', capsize=5)
    ax.scatter(plotx,Lgauss[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx,Lgauss[:,1],yerr=Lgaussstd[:,1],fmt='o',solid_capstyle='projecting', capsize=5)
    ax.scatter(plotx+0.05,Lgauss[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    ax.errorbar(plotx+0.05,Lgauss[:,2],yerr=Lgaussstd[:,2],fmt='o',solid_capstyle='projecting', capsize=5)
    plt.legend(prop={'size': 12})
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'Lattice Parameter ($\mathrm{\AA}$)', fontsize = 15) # Y label
    #ax.set_xlabel('Br content', fontsize = 15) # X label
    ax.set_xticks(plotx)
    ax.set_xticklabels(plotxlab)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return Lgauss, Lgaussstd


def abs_sqrt(m):
    """Calculate the sign-conversing square root of a number."""

    return np.sqrt(np.abs(m))*np.sign(m)


def draw_tilt_corr_evolution_sca(T, steps, uniname, saveFigures, xaxis_type = 't', scasize = 1.5, y_lim = [-1,1]):
    """
    Draw the tilt correlation evolution.

    Args:
        T (numpy.ndarray): Tilt data.
        steps (numpy.ndarray): Steps data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        xaxis_type (str): Type of x-axis ('N', 'T', or 't').
        scasize (float): Size of scatter points.
        y_lim (list): Y-axis limits.

    Returns:
        None
    """
    
    fig_name=f"traj_tilt_corr_{uniname}.png"
    
    assert T.shape[0] == len(steps)
    
    if T.shape[2] == 3:
        for i in range(T.shape[1]):
            if i == 0:
                plt.scatter(steps,T[:,i,0],label = r'$\mathit{a}$', s = scasize, c = 'green')
                plt.scatter(steps,T[:,i,1],label = r'$\mathit{b}$', s = scasize, c = 'blue')
                plt.scatter(steps,T[:,i,2],label = r'$\mathit{c}$', s = scasize, c = 'red')
            else:
                plt.scatter(steps,T[:,i,0], s = scasize, c = 'green')
                plt.scatter(steps,T[:,i,1], s = scasize, c = 'blue')
                plt.scatter(steps,T[:,i,2], s = scasize, c = 'red')
    elif T.shape[2] == 6:
        for i in range(T.shape[1]):
            if i == 0:
                plt.scatter(steps,T[:,i,0],label = r'$\mathit{a}$', s = scasize, c = 'green')
                plt.scatter(steps,T[:,i,1], s = scasize, c = 'green')
                plt.scatter(steps,T[:,i,2],label = r'$\mathit{b}$', s = scasize, c = 'blue')
                plt.scatter(steps,T[:,i,3], s = scasize, c = 'blue')
                plt.scatter(steps,T[:,i,4],label = r'$\mathit{c}$', s = scasize, c = 'red')
                plt.scatter(steps,T[:,i,5], s = scasize, c = 'red')
            else:
                plt.scatter(steps,T[:,i,0], s = scasize, c = 'green')
                plt.scatter(steps,T[:,i,1], s = scasize, c = 'green')
                plt.scatter(steps,T[:,i,2], s = scasize, c = 'blue')
                plt.scatter(steps,T[:,i,3], s = scasize, c = 'blue')
                plt.scatter(steps,T[:,i,4], s = scasize, c = 'red')
                plt.scatter(steps,T[:,i,5], s = scasize, c = 'red')
    
        
    plt.legend()
    ax = plt.gca()
    if not y_lim is None:
        ax.set_ylim(y_lim)
    ax.set_yticks([])
    
    if xaxis_type == 'N':
        plt.xlabel("MD step")
    elif xaxis_type == 'T':
        plt.xlabel("Temperature (K)")
    elif xaxis_type == 't':
        plt.xlabel("Time (ps)")
    plt.ylabel("Spatial correlation (a.u.)")
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()


def draw_tilt_and_corr_density_shade(T, Corr, uniname, saveFigures, n_bins = 100, title = None):
    """ 
    Generate the Glazer plot. 

    Args:
        T (numpy.ndarray): Tilt data.
        Corr (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        title (str): Title of the plot.

    Returns:
        list of floats: tilt correlation polarity (TCP) values of each axis.
    """
    
    fig_name=f"traj_tilt_corr_density_{uniname}.png"
    
    corr_power = 2.5
    fill_alpha = 0.5
    
    T_a = T[:,:,0].reshape((T.shape[0]*T.shape[1]))
    T_b = T[:,:,1].reshape((T.shape[0]*T.shape[1]))
    T_c = T[:,:,2].reshape((T.shape[0]*T.shape[1]))
    tup_T = (T_a,T_b,T_c)
    assert len(tup_T) == 3
    assert Corr.shape[2] == 3 or Corr.shape[2] == 6

    if Corr.shape[2] == 3:
        C = (Corr[:,:,0].reshape((Corr.shape[0]*Corr.shape[1])),
             Corr[:,:,1].reshape((Corr.shape[0]*Corr.shape[1])),
             Corr[:,:,2].reshape((Corr.shape[0]*Corr.shape[1])))
    elif Corr.shape[2] == 6:
        C = (np.concatenate((Corr[:,:,0],Corr[:,:,1]),axis=0).reshape((Corr.shape[0]*2*Corr.shape[1])),
             np.concatenate((Corr[:,:,2],Corr[:,:,3]),axis=0).reshape((Corr.shape[0]*2*Corr.shape[1])),
             np.concatenate((Corr[:,:,4],Corr[:,:,5]),axis=0).reshape((Corr.shape[0]*2*Corr.shape[1])))

    figs, axs = plt.subplots(3, 1)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0", "C1", "C2"]
    #rgbcode = np.array([[0,1,0,fill_alpha],[0,0,1,fill_alpha],[1,0,0,fill_alpha]])
    por = [0,0,0]
    for i in range(3):
        
        y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yt=y/max(y)
        axs[i].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 2.4)
        #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)

        y,binEdges=np.histogram(C[i],bins=n_bins,range=[-45,45]) 
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yc=y/max(y)
        yy=yt*yc

        axs[i].fill_between(bincenters, yy, 0, facecolor = colors[i], alpha=fill_alpha, interpolate=True)
        axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes, style='italic')
        
        axs[i].set_ylim(bottom=0)
        
        parneg = np.sum(np.power(yc,corr_power)[bincenters<0])
        parpos = np.sum(np.power(yc,corr_power)[bincenters>0])
        por[i] = (-parneg+parpos)/(parneg+parpos)
        
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
        ax.set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 15) # X label
        ax.set_xlim([-45,45])
        ax.set_xticks([-45,-30,-15,0,15,30,45])
        ax.set_yticks([])
        
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[1].set_xlabel("")
    axs[2].set_ylabel("")

    if not title is None:
        axs[0].set_title(title,fontsize=17)
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    
    div = 0.35
    scal = 4/3
    for ci, cval in enumerate(por):
        if abs(cval) > div:
            cval = np.sign(cval)*(np.power(((np.abs(cval)-div)/(1-div)),1/scal)*(1-div)+div)
            por[ci] = np.round(cval,4)
        else:
            cval = np.sign(cval)*(np.power(np.abs(cval)/div,scal)*div)
            por[ci] = np.round(cval,4)
    
    return por


def draw_tilt_and_corr_density_shade_non3D(T, Corr, uniname, saveFigures, n_bins = 100, title = None):
    """ 
    Generate the Glazer plot for non-3D structures.

    Args:
        T (numpy.ndarray): Tilt data.
        Corr (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        title (str): Title of the plot.

    Returns:
        list of floats: tilt correlation polarity (TCP) values of each axis.
    """
    
    fig_name=f"traj_tilt_corr_density_{uniname}.png"
    
    corr_power = 2.5
    fill_alpha = 0.5
    
    if Corr.ndim == 4: # isotropic treatment
        T_a = T[:,:,0].reshape(-1,)
        T_b = T[:,:,1].reshape(-1,)
        T_c = T[:,:,2].reshape(-1,)
        tup_T = (T_a,T_b,T_c)
        
        C = (Corr[:,:,:,0].reshape(-1,),
             Corr[:,:,:,1].reshape(-1,),
             Corr[:,:,:,2].reshape(-1,))
        
        figs, axs = plt.subplots(3, 1)
        labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
        colors = ["C0", "C1", "C2"]
        #rgbcode = np.array([[0,1,0,fill_alpha],[0,0,1,fill_alpha],[1,0,0,fill_alpha]])
        por = [0,0,0]
        for i in range(3):
            
            y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yt=y/max(y)
            axs[i].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 2.4)
            #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)

            y,binEdges=np.histogram(C[i],bins=n_bins,range=[-45,45]) 
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yc=y/max(y)
            yy=yt*yc

            axs[i].fill_between(bincenters, yy, 0, facecolor = colors[i], alpha=fill_alpha, interpolate=True)
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes, style='italic')
            
            axs[i].set_ylim(bottom=0)
            
            parneg = np.sum(np.power(yc,corr_power)[bincenters<0])
            parpos = np.sum(np.power(yc,corr_power)[bincenters>0])
            por[i] = (-parneg+parpos)/(parneg+parpos)
            
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 15) # X label
            ax.set_xlim([-45,45])
            ax.set_xticks([-45,-30,-15,0,15,30,45])
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[1].set_xlabel("")
        axs[2].set_ylabel("")

        if not title is None:
            axs[0].set_title(title,fontsize=17)
            
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
            
        plt.show()
        
        div = 0.35
        scal = 4/3
        for ci, cval in enumerate(por):
            if abs(cval) > div:
                cval = np.sign(cval)*(np.power(((np.abs(cval)-div)/(1-div)),1/scal)*(1-div)+div)
                por[ci] = np.round(cval,4)
            else:
                cval = np.sign(cval)*(np.power(np.abs(cval)/div,scal)*div)
                por[ci] = np.round(cval,4)
    
    elif Corr.ndim == 5:
        T_a = T[:,:,0].reshape(-1,)
        T_b = T[:,:,1].reshape(-1,)
        T_c = T[:,:,2].reshape(-1,)
        tup_T = (T_a,T_b,T_c)
        
        C = (Corr[:,:,:,0].reshape(-1,),
             Corr[:,:,:,1].reshape(-1,),
             Corr[:,:,:,2].reshape(-1,))
        
        figs, axs = plt.subplots(3, 1)
        labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
        colors = ["C0", "C1", "C2"]
        #rgbcode = np.array([[0,1,0,fill_alpha],[0,0,1,fill_alpha],[1,0,0,fill_alpha]])
        por = [0,0,0]
        for i in range(3):
            
            y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yt=y/max(y)
            axs[i].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 2.4)
            #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)

            y,binEdges=np.histogram(C[i],bins=n_bins,range=[-45,45]) 
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yc=y/max(y)
            yy=yt*yc

            axs[i].fill_between(bincenters, yy, 0, facecolor = colors[i], alpha=fill_alpha, interpolate=True)
            axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes, style='italic')
            
            axs[i].set_ylim(bottom=0)
            
            parneg = np.sum(np.power(yc,corr_power)[bincenters<0])
            parpos = np.sum(np.power(yc,corr_power)[bincenters>0])
            por[i] = (-parneg+parpos)/(parneg+parpos)
            
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
            ax.set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 15) # X label
            ax.set_xlim([-45,45])
            ax.set_xticks([-45,-30,-15,0,15,30,45])
            ax.set_yticks([])
            
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[0].yaxis.set_ticklabels([])
        axs[1].yaxis.set_ticklabels([])
        axs[2].yaxis.set_ticklabels([])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[1].set_xlabel("")
        axs[2].set_ylabel("")

        if not title is None:
            axs[0].set_title(title,fontsize=17)
            
        if saveFigures:
            plt.savefig(fig_name, dpi=350,bbox_inches='tight')
            
        plt.show()
        
        div = 0.35
        scal = 4/3
        for ci, cval in enumerate(por):
            if abs(cval) > div:
                cval = np.sign(cval)*(np.power(((np.abs(cval)-div)/(1-div)),1/scal)*(1-div)+div)
                por[ci] = np.round(cval,4)
            else:
                cval = np.sign(cval)*(np.power(np.abs(cval)/div,scal)*div)
                por[ci] = np.round(cval,4)
    
    return por


def draw_tilt_and_corr_density_shade_longarray(T, Corr, uniname, saveFigures, n_bins = 100, title = None):
    """ 
    Generate the Glazer plot for a reshaped array. 

    Args:
        T (numpy.ndarray): Tilt data.
        Corr (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        title (str): Title of the plot.

    Returns:
        list of floats: tilt correlation polarity (TCP) values of each axis.
    """
    
    fig_name=f"traj_tilt_corr_density_{uniname}.png"
    
    corr_power = 2.5
    fill_alpha = 0.5
    
    T_a = T[:,0]
    T_b = T[:,1]
    T_c = T[:,2]
    tup_T = (T_a,T_b,T_c)
    assert len(tup_T) == 3
    assert Corr.shape[1] == 3 or Corr.shape[1] == 6

    if Corr.shape[1] == 3:
        C = (Corr[:,0],Corr[:,1],Corr[:,2])
    elif Corr.shape[1] == 6:
        C = (np.concatenate((Corr[:,0],Corr[:,1]),axis=0),
             np.concatenate((Corr[:,2],Corr[:,3]),axis=0),
             np.concatenate((Corr[:,4],Corr[:,5]),axis=0))

    figs, axs = plt.subplots(3, 1)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0", "C1", "C2"]
    #rgbcode = np.array([[0,1,0,fill_alpha],[0,0,1,fill_alpha],[1,0,0,fill_alpha]])
    por = [0,0,0]
    for i in range(3):
        
        y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yt=y/max(y)
        axs[i].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 2.4)
        #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)

        y,binEdges=np.histogram(C[i],bins=n_bins,range=[-45,45]) 
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yc=y/max(y)
        yy=yt*yc

        axs[i].fill_between(bincenters, yy, 0, facecolor = colors[i], alpha=fill_alpha, interpolate=True)
        axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes, style='italic')
        
        axs[i].set_ylim(bottom=0)
        
        parneg = np.sum(np.power(yc,corr_power)[bincenters<0])
        parpos = np.sum(np.power(yc,corr_power)[bincenters>0])
        por[i] = (-parneg+parpos)/(parneg+parpos)
        
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
        ax.set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 15) # X label
        ax.set_xlim([-45,45])
        ax.set_xticks([-45,-30,-15,0,15,30,45])
        ax.set_yticks([])
        
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[1].set_xlabel("")
    axs[2].set_ylabel("")

    if not title is None:
        axs[0].set_title(title,fontsize=17)
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    
    div = 0.35
    scal = 4/3
    for ci, cval in enumerate(por):
        if abs(cval) > div:
            cval = np.sign(cval)*(np.power(((np.abs(cval)-div)/(1-div)),1/scal)*(1-div)+div)
            por[ci] = np.round(cval,4)
        else:
            cval = np.sign(cval)*(np.power(np.abs(cval)/div,scal)*div)
            por[ci] = np.round(cval,4)
    
    return por


def draw_tilt_and_corr_density_shade_frame(T, Corr, uniname, saveFigures, n_bins = 100):
    """ 
    Generate the Glazer plot for a single frame. 

    Args:
        T (numpy.ndarray): Tilt data.
        Corr (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        list of floats: tilt correlation polarity (TCP) values of each axis.
    """
    
    fig_name=f"frame_tilts_{uniname}.png"
    
    corr_power = 2.5
    fill_alpha = 0.5
    
    T_a = T[:,0].reshape((-1,))
    T_b = T[:,1].reshape((-1,))
    T_c = T[:,2].reshape((-1,))
    tup_T = (T_a,T_b,T_c)
    assert len(tup_T) == 3
    assert Corr.shape[1] == 6

    C = (np.concatenate((Corr[:,0],Corr[:,1]),axis=0).reshape((-1,)),
         np.concatenate((Corr[:,2],Corr[:,3]),axis=0).reshape((-1,)),
         np.concatenate((Corr[:,4],Corr[:,5]),axis=0).reshape((-1,)))

    figs, axs = plt.subplots(3, 1)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0", "C1", "C2"]
    #rgbcode = np.array([[0,1,0,fill_alpha],[0,0,1,fill_alpha],[1,0,0,fill_alpha]])
    por = [0,0,0]
    tquant = []
    for i in range(3):
        y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yt=y/max(y)
        axs[i].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 2.4)
        #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
        
        t1 = list(np.round(bincenters[np.logical_and(y>0,bincenters>-0.0001)],3))
        
        y,binEdges=np.histogram(C[i],bins=n_bins,range=[-45,45]) 
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yc=y/max(y)
        yy=yt*yc

        axs[i].fill_between(bincenters, yy, 0, facecolor = colors[i], alpha=fill_alpha, interpolate=True)
        axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes, style='italic')
        
        axs[i].set_ylim(bottom=0)
        
        parneg = np.sum(np.power(yc,corr_power)[bincenters<0])
        parpos = np.sum(np.power(yc,corr_power)[bincenters>0])
        por[i] = (-parneg+parpos)/(parneg+parpos)
        
        tquant.append(t1)
        
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
        ax.set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 15) # X label
        ax.set_xlim([-45,45])
        ax.set_xticks([-45,-30,-15,0,15,30,45])
        ax.set_yticks([])
        
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[1].set_xlabel("")
    axs[2].set_ylabel("")
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    
    div = 0.35
    scal = 4/3
    for ci, cval in enumerate(por):
        if abs(cval) > div:
            cval = np.sign(cval)*(np.power(((np.abs(cval)-div)/(1-div)),1/scal)*(1-div)+div)
            por[ci] = np.round(cval,4)
        else:
            cval = np.sign(cval)*(np.power(np.abs(cval)/div,scal)*div)
            por[ci] = np.round(cval,4)
    
    return tquant, por


def draw_tilt_coaxial(T, uniname, saveFigures, n_bins = 171, title = None):
    """
    Draw the coaxial tilting correlation plots.

    Args:
        T (numpy.ndarray): Tilt data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        title (str): Title of the plot.

    Returns:
        None
    """
    
    fig_name1=f"traj_tilt_coaxial_xy_{uniname}.png"
    fig_name2=f"traj_tilt_coaxial_xz_{uniname}.png"
    fig_name3=f"traj_tilt_coaxial_yz_{uniname}.png"
    
    cxy = T[:,:,[0,1]].reshape(-1,2)
    cxz = T[:,:,[0,2]].reshape(-1,2)
    cyz = T[:,:,[1,2]].reshape(-1,2)
    a_bins = np.linspace(-20, 20, n_bins)
    b_bins = np.linspace(-20, 20, n_bins)
    
    fig, ax = plt.subplots()
    plt.hist2d(cxy[:,0], cxy[:,1], bins =[a_bins, b_bins])
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('X-tilt (deg)', fontsize=14)
    plt.ylabel('Y-tilt (deg)', fontsize=14)
    if not title is None:
        ax.set_title(title,fontsize=16)
    if saveFigures:
        plt.savefig(fig_name1, dpi=350,bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots()
    plt.hist2d(cxz[:,0], cxz[:,1], bins =[a_bins, b_bins])
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('X-tilt (deg)', fontsize=14)
    plt.ylabel('Z-tilt (deg)', fontsize=14)
    if not title is None:
        ax.set_title(title,fontsize=16)
    if saveFigures:
        plt.savefig(fig_name2, dpi=350,bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots()
    plt.hist2d(cyz[:,0], cyz[:,1], bins =[a_bins, b_bins])
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('Y-tilt (deg)', fontsize=14)
    plt.ylabel('Z-tilt (deg)', fontsize=14)
    if not title is None:
        ax.set_title(title,fontsize=16)
    if saveFigures:
        plt.savefig(fig_name3, dpi=350,bbox_inches='tight')
    plt.show()


def draw_tilt_and_corr_density_full(T, Cf, uniname, saveFigures, n_bins = 100, title = None):
    """
    Generate the full 3-by-3 array of Glazer plots including the off-diagonal correlations.

    Args:
        T (numpy.ndarray): Tilt data.
        Cf (numpy.ndarray): Full correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        title (str): Title of the plot.

    Returns:
        numpy.ndarray: tilt correlation polarity (TCP) values of each axis and direction.
    """
    
    fig_name=f"traj_tilt_corr_density_full_{uniname}.png"
    
    corr_power = 2.5
    fill_alpha = 0.5
    
    div = 0.35
    scal = 4/3
    
    T_a = T[:,:,0].reshape((T.shape[0]*T.shape[1]))
    T_b = T[:,:,1].reshape((T.shape[0]*T.shape[1]))
    T_c = T[:,:,2].reshape((T.shape[0]*T.shape[1]))
    tup_T = (T_a,T_b,T_c)
    
    C = np.empty((3,3,Cf.shape[0]*Cf.shape[1]*2))
    for i in range(3):
        for j in range(3):
            C[i,j,:] = Cf[:,:,i,j,:].reshape(-1,)
    
    figs, axs = plt.subplots(3, 3)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0", "C1", "C2"]
    #rgbcode = np.array([[0,1,0,fill_alpha],[0,0,1,fill_alpha],[1,0,0,fill_alpha]])
    por = np.empty((3,3))
    for i in range(3):
        y,binEdges=np.histogram(tup_T[i],bins=n_bins,range=[-45,45])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        yt=y/max(y)
        for j in range(3):
            
            axs[i,j].plot(bincenters,yt,label = labels[i], color = colors[i],linewidth = 1.5)
            #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)

            y,binEdges=np.histogram(C[i,j,:],bins=n_bins,range=[-45,45]) 
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yc=y/max(y)
            yy=yt*yc

            axs[i,j].fill_between(bincenters, yy, 0, facecolor = colors[j], alpha=fill_alpha, interpolate=True)
            #axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15, verticalalignment='center', transform=axs[i].transAxes)
            
            axs[i,j].set_ylim(bottom=0)
            
            parneg = np.sum(np.power(yc,corr_power)[bincenters<0])
            parpos = np.sum(np.power(yc,corr_power)[bincenters>0])
            cval = (-parneg+parpos)/(parneg+parpos)
            
            if abs(cval) > div:
                cval = np.sign(cval)*(np.power(((np.abs(cval)-div)/(1-div)),1/scal)*(1-div)+div)
                por[i,j] = np.round(cval,4)
            else:
                cval = np.sign(cval)*(np.power(np.abs(cval)/div,scal)*div)
                por[i,j] = np.round(cval,4)
            
            axs[i,j].text(0.80, 0.82, round(cval,3), horizontalalignment='center', fontsize=10, verticalalignment='center', transform=axs[i,j].transAxes)
            
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=10)
        #ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
        #ax.set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 15) # X label
        ax.set_xlim([-20,20])
        ax.set_xticks([-20,-10,0,10,20])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_yticks([])
    
    axs[2,0].xaxis.set_ticklabels([-20,-10,0,10,20])  
    axs[2,1].xaxis.set_ticklabels([-20,-10,0,10,20])    
    axs[2,2].xaxis.set_ticklabels([-20,-10,0,10,20])  
    
    axs[1,0].set_ylabel('Counts (a.u.)', fontsize = 12) # Y label
    axs[2,1].set_xlabel(r'Tilt Angle ($\degree$)', fontsize = 12) # X label

    if not title is None:
        axs[0,1].set_title(title,fontsize=13)
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    
    return por


def draw_tilt_corr_density_time(T, Corr, steps, uniname, saveFigures, smoother = 0, n_bins = 50):
    """
    Generate the time evolution of correlation density plots for tilting.

    Args:
        T (numpy.ndarray): Tilt data.
        Corr (numpy.ndarray): Correlation data.
        steps (numpy.ndarray): Time steps.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): Smoothing window size.
        n_bins (int): Number of bins for the histogram.

    Returns:
        tuple: Time steps and correlation values for each axis.
    """
    
    fig_name = f"traj_tilt_corr_time_{uniname}.png"
    
    Cline = np.empty((0,3))
    aw = 5
    corr_power = 2.5
    
    for i in range(T.shape[0]-aw+1):
        t1 = T[list(range(i,i+aw)),:,:]
        c1 = Corr[list(range(i,i+aw)),:,:]
        
        por = [0,0,0]
        for i in range(3):
            
            y,binEdges=np.histogram(t1[:,:,i].reshape(-1,),bins=n_bins,range=[-45,45])
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yt=y/max(y)

            y,binEdges=np.histogram(c1[:,:,list(range(i*2,i*2+2))],bins=n_bins,range=[-45,45]) 
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            yc=y/max(y)
            
            yy=yt*yc
            
            parneg = np.sum(np.power(yy,corr_power)[bincenters<0])
            parpos = np.sum(np.power(yy,corr_power)[bincenters>0])
            por[i] = (-parneg+parpos)/(parneg+parpos)
        
        Cline = np.concatenate((Cline,np.array(por).reshape(1,3)),axis=0)
    
    ts = steps[1] - steps[0]
    time_window = smoother # picosecond
    sgw = round(time_window/ts)
    if sgw<5: sgw = 5
    if sgw%2==0: sgw+=1
    if smoother != 0:
        Ca = savitzky_golay(Cline[:,0],window_size=sgw)
        Cb = savitzky_golay(Cline[:,1],window_size=sgw)
        Cc = savitzky_golay(Cline[:,2],window_size=sgw)
    else:
        Ca = Cline[:,0]
        Cb = Cline[:,1]
        Cc = Cline[:,2]
    
    w, h = figaspect(0.8/1.45)
    plt.subplots(figsize=(w,h))
    ax = plt.gca()
    
    plt.plot(steps[:(len(steps)-aw+1)],Ca,label = r'$\mathit{a}$',linewidth=2.5)
    plt.plot(steps[:(len(steps)-aw+1)],Cb,label = r'$\mathit{b}$',linewidth=2.5)
    plt.plot(steps[:(len(steps)-aw+1)],Cc,label = r'$\mathit{c}$',linewidth=2.5)    

    ax.set_ylim([-1.05,1.05])
    plt.axhline(y=0,linestyle='dashed',color='k',linewidth=1)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel('Tilting correlation polarity (a.u.)', fontsize=14)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.legend(prop={'size': 13})
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return (steps[:(len(steps)-aw+1)],Ca,Cb,Cc)


def draw_tilt_spatial_corr(C, uniname, saveFigures, n_bins = 100):
    """
    Generate the spatial correlation plots for tilting.

    Args:   
        C (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        None
    """
    
    fig_name = f"traj_tilt_spatial_corr_{uniname}.png"
    
    num_lens = C.shape[0]
    
    fig, axs = plt.subplots(nrows=3, ncols=num_lens, sharex=False, sharey=True)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    for i in range(3):
        for j in range(num_lens):
            axs[i,j].hist(C[j][i],bins=n_bins,range=[-45,45],orientation='horizontal')
            if j == 0:
                axs[i,j].text(0.1, 0.86, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i,j].transAxes)
            if i == 0:
                axs[i,j].text(0.16, 1.1, "NN"+str(j+1), horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i,j].transAxes)
        
    for ax in axs.flat:

        ax.set_ylim([-45,45])
        ax.set_yticks([-45,-22.5,0,22.5,45])
        ax.set_xticks([])

    fig.text(0.5, 0.07, 'Counts (a.u.)', ha='center', fontsize=12)
    fig.text(0.01, 0.5, r'Tilt Angle ($\degree$)', va='center', rotation='vertical', fontsize=12)

    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
                
    plt.show()


def Tilt_correlation(T,MDTimestep,smoother=0):
    """ 
    Compute time-correlation of tilting.  

    Args:
        T (numpy.ndarray): Tilt data.
        MDTimestep (float): Time step in picoseconds.
        smoother (int): Smoothing window size.

    Returns:
        tuple: delta-t and self-correlation values of tilting.
    """
    
    if T.shape[0] > 1500:
        sh=1000
    else:
        sh=round(T.shape[0]/2) 
    correlation=np.zeros((sh,T.shape[1],3))
    
    if smoother != 0:
        time_window = smoother # picosecond
        sgw = round(time_window/MDTimestep)
        if sgw<5: sgw = 5
        if sgw%2==0: sgw+=1
        
        Ts = T.copy()
        for i in range(T.shape[1]):
            for j in range(3):
                Ts[:,i,j] = savitzky_golay(Ts[:,i,j],window_size=sgw)
        
        T = Ts.copy()
    
# =============================================================================
#     Tmean = np.mean(np.abs(T),axis=0)
#     #Tmean=0
#     
#     for i in range(T.shape[0]-sh): 
#         for dt in range(sh): 
#             v1 = T[i,:,:] 
#             v2 = T[i+dt,:,:]
#             temp = np.sign(v1)*np.sign(v2)*np.sqrt(np.abs(np.multiply(np.abs(v1)-Tmean,np.abs(v2)-Tmean)))
#             temp[np.isnan(temp)] = 0
#             correlation[dt,:]=correlation[dt,:]+temp
# =============================================================================
    
    Tmean = np.mean(T,axis=0)
    Tmean=0
    
    for i in range(T.shape[0]-sh): 
        for dt in range(sh): 
            v1 = T[i,:,:] 
            v2 = T[i+dt,:,:]
            prod = np.multiply(v1-Tmean,v2-Tmean)
            #temp = prod
            temp = np.sign(prod)*np.sqrt(np.abs(prod))
            temp[np.isnan(temp)] = 0
            correlation[dt,:]=correlation[dt,:]+temp

    correlation = np.mean(correlation/(T.shape[0]-sh),axis=1)
    
    x = np.array(range(sh))*MDTimestep
    x = x.reshape(1,x.shape[0])
    
    return x, correlation


def quantify_tilt_domain(sc,scnorm,plot_label='tilt'):
    """ 
    Compute spatial coorelation of tilting.  

    Args:
        sc (numpy.ndarray): Spatial correlation data.
        scnorm (numpy.ndarray): Normalized spatial correlation data.
        plot_label (str): Label for plotting.

    Returns:
        numpy.ndarray: Decay lengths for each tilt axis and spatial direction.
    """
    
    if_crosszero = np.sum(np.abs(np.diff(np.sign(sc+0.03),axis=1)),axis=1)>1
    
    nns = sc.shape[1]
    
    #thr = 0.015
    #sc[np.abs(sc)>thr] = (np.sqrt(np.abs(sc))*np.sign(sc))[np.abs(sc)>thr]
    
    from scipy.optimize import curve_fit
    def model_func(x, k):
        #return 0.9 * np.exp(-k1*x) + 0.1 * np.exp(-k2*x)
        return np.exp(-k*x)
    
    pop_warning = []
    scdecay = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            tc = np.abs(scnorm[i,:,j])
            p0 = (5) # starting search coeffs
            opt, pcov = curve_fit(model_func, np.array(list(range(scnorm.shape[1]))), tc, p0)
            k= opt
            
            #p0 = (5,0.1) # starting search coeffs
            #opt, pcov = curve_fit(model_func, np.array(list(range(sc.shape[1]))), tc, p0)
            #k1 ,k2= opt
            
            #print(k1 ,k2)
            #k=k1
            #y2 = model_func(np.array(list(range(sc.shape[1]))), k1 ,k2)
                
            scdecay[i,j] = (1/k)
            
            #fig,ax = plt.subplots()
            #plt.plot(list(range(nns)),sc[i,:,j],linewidth=1.5)
            #plt.plot(list(range(nns)),y2,linewidth=1.5)
    
    if pop_warning:
        print(f"!Property Spatial Corr: fitting of decay length may be wrong, a value(s): {pop_warning}")
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),scnorm[0,:,0],linewidth=1.5,label=f'{plot_label} 0')
    plt.plot(list(range(nns)),scnorm[1,:,0],linewidth=1.5,label=f'{plot_label} 1')
    plt.plot(list(range(nns)),scnorm[2,:,0],linewidth=1.5,label=f'{plot_label} 2')
    plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    plt.title("Correlation along axis 0",fontsize=15)
    ax.set_ylim([-1,1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),scnorm[0,:,1],linewidth=1.5,label=f'{plot_label} 0')
    plt.plot(list(range(nns)),scnorm[1,:,1],linewidth=1.5,label=f'{plot_label} 1')
    plt.plot(list(range(nns)),scnorm[2,:,1],linewidth=1.5,label=f'{plot_label} 2')
    plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    plt.title("Correlation along axis 1",fontsize=15)
    ax.set_ylim([-1,1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),scnorm[0,:,2],linewidth=1.5,label=f'{plot_label} 0')
    plt.plot(list(range(nns)),scnorm[1,:,2],linewidth=1.5,label=f'{plot_label} 1')
    plt.plot(list(range(nns)),scnorm[2,:,2],linewidth=1.5,label=f'{plot_label} 2')
    plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    plt.title("Correlation along axis 2",fontsize=15)
    ax.set_ylim([-1,1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    return scdecay


def quantify_halideconc_tilt_domain(TCconc, concent, uniname, saveFigures, n_bins = 100):
    """ 
    Isolate tilting pattern wrt. the local halide concentration.  

    Args:
        TCconc (numpy.ndarray): Tilt correlation data.
        concent (list): List of halide concentrations.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        numpy.ndarray: Correlation lengths (diagonal and off-diagonal) for each concentration
    """
    
    fig_name=f"tilt_domain_halideconc_{uniname}.png"
    nns = TCconc[0].shape[2]
    
    from scipy.optimize import curve_fit
    def model_func(x, k):
        return np.exp(-k*x)
    
    scdecay = np.empty((len(concent),3,3))
    for t in range(len(concent)):
        for i in range(3):
            for j in range(3):
                tc = np.abs(np.mean(TCconc[t],axis=0)[i,:,j])
                p0 = (5) # starting search coeffs
                opt, pcov = curve_fit(model_func, np.array(list(range(nns))), tc, p0)
                k= opt

                scdecay[t,i,j] = (1/k)
        
    if np.sum(scdecay<0) > 0:
        print("!Halideconc_tilt_domain: found fitted infinite correlation length. ")
        scdecay[scdecay<0] = np.nan
    
    # assume isotropic
    diag = (scdecay[:,0,0]+scdecay[:,1,1]+scdecay[:,2,2])/3
    off_diag = (scdecay[:,0,1]+scdecay[:,1,2]+scdecay[:,0,2]+scdecay[:,1,0]+scdecay[:,2,1]+scdecay[:,2,0])/6
    
    data = np.array([np.array(concent),diag,off_diag])
    
    fig,ax = plt.subplots()
    plt.plot(data[0,:],data[1,:],marker='s',markersize=6,linewidth=2.2,label='Normal')
    plt.plot(data[0,:],data[2,:],marker='s',markersize=6,linewidth=2.2,label='Parallel')
    plt.title("Tilting Correlation Length",fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Br Content', fontsize=15)
    plt.ylabel('Correlation Length (unit cell)', fontsize=15)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return data


def quantify_octatype_tilt_domain(TCtype, config_types, uniname, saveFigures, n_bins = 100):
    """ 
    Isolate tilting pattern wrt. the local halide configuration.  

    Args:
        TCtype (numpy.ndarray): Tilt correlation data for each type.
        config_types (list): List of halide configurations.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        numpy.ndarray: Correlation lengths (diagonal and off-diagonal) for each configuration type.
    """
    
    fig_name=f"tilt_domain_octatype_{uniname}.png"
    nns = TCtype[0].shape[2]
    
    typesname = ["I6 Br0","I5 Br1","I4 Br2: cis","I4 Br2: trans","I3 Br3: fac",
                 "I3 Br3: mer","I2 Br4: cis","I2 Br4: trans","I1 Br5","I0 Br6"]
    typexval = [0,1,1.83,2.17,2.83,3.17,3.83,4.17,5,6]
    typextick = ['0','1','2c','2t','3f','3m','4c','4t','5','6']
    
    config_types = list(config_types)
    config_involved = []
    
    from scipy.optimize import curve_fit
    def model_func(x, k):
        return np.exp(-k*x)
    
    scdecay = np.empty((len(TCtype),3,3))
    for t in range(len(TCtype)):
        meantc = np.mean(TCtype[t],axis=0)
        for i in range(3):
            for j in range(3):
                tc = np.abs(meantc[i,:,j])
                p0 = (5) # starting search coeffs
                opt, pcov = curve_fit(model_func, np.array(list(range(nns))), tc, p0)
                k= opt

                scdecay[t,i,j] = (1/k)
        
        fig,ax = plt.subplots()
        plt.plot(list(range(nns)),meantc[0,:,2],linewidth=1.5,label='axis 0')
        plt.plot(list(range(nns)),meantc[1,:,2],linewidth=1.5,label='axis 1')
        plt.plot(list(range(nns)),meantc[2,:,2],linewidth=1.5,label='axis 2')
        plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
        plt.title(f"{typesname[config_types[t]]} - Along axis 2",fontsize=15)
        ax.set_ylim([-1,1])
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('Distance (unit cell)', fontsize=14)
        plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
        legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
        legend.get_frame().set_alpha(0.7)
        
    if np.sum(scdecay<0) > 0:
        print("!Halideconc_tilt_domain: found fitted infinite correlation length. ")
        scdecay[scdecay<0] = np.nan
    
    # assume isotropic
    diag = (scdecay[:,0,0]+scdecay[:,1,1]+scdecay[:,2,2])/3
    off_diag = (scdecay[:,0,1]+scdecay[:,1,2]+scdecay[:,0,2]+scdecay[:,1,0]+scdecay[:,2,1]+scdecay[:,2,0])/6
    
    # plot type dependence   
    plotx = np.array([typexval[i] for i in config_types])
    plotxlab = [typextick[i] for i in config_types]
    data = [plotxlab,diag,off_diag]
    
    fig,ax = plt.subplots()
    plt.plot(plotx,data[1],marker='s',markersize=6,linewidth=2.2,label='Normal')
    plt.plot(plotx,data[2],marker='s',markersize=6,linewidth=2.2,label='Parallel')
    ax.set_xticks(plotx)
    ax.set_xticklabels(plotxlab)
    plt.title("Tilting Correlation Length",fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Br Content', fontsize=15)
    plt.ylabel('Correlation Length (unit cell)', fontsize=15)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return data


def quantify_hetero_tilt_domain(TCcls, uniname, saveFigures, n_bins = 100):
    """ 
    Isolate tilting pattern wrt. the local halide configuration in hetero-mode.  

    Args:
        TCcls (numpy.ndarray): Tilt correlation data for each type.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.

    Returns:
        numpy.ndarray: Correlation lengths (diagonal and off-diagonal) for each configuration type.
    """
    
    fig_name=f"tilt_domain_hetero_{uniname}.png"
    nns = TCcls[0].shape[2]
    
    typesname = ["bulk","grain boundary","grain"]
    typexval = [0,1,2]
    typextick = ["bulk","grain boundary","grain"]
    
    from scipy.optimize import curve_fit
    def model_func(x, k):
        return np.exp(-k*x)
    
    scdecay = np.empty((len(TCcls),3,3))
    for t in range(len(TCcls)):
        meantc = np.mean(TCcls[t],axis=0)
        for i in range(3):
            for j in range(3):
                tc = np.abs(meantc[i,:,j])
                p0 = (5) # starting search coeffs
                opt, pcov = curve_fit(model_func, np.array(list(range(nns))), tc, p0)
                k= opt

                scdecay[t,i,j] = (1/k)
        
        fig,ax = plt.subplots()
        plt.plot(list(range(nns)),meantc[0,:,2],linewidth=1.5,label='axis 0')
        plt.plot(list(range(nns)),meantc[1,:,2],linewidth=1.5,label='axis 1')
        plt.plot(list(range(nns)),meantc[2,:,2],linewidth=1.5,label='axis 2')
        plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
        plt.title(f"{typesname[t]} - Along axis 2",fontsize=15)
        ax.set_ylim([-1,1])
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('Distance (unit cell)', fontsize=14)
        plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
        legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
        legend.get_frame().set_alpha(0.7)
        
    if np.sum(scdecay<0) > 0:
        print("!Halideconc_tilt_domain: found fitted infinite correlation length. ")
        scdecay[scdecay<0] = np.nan
    
    # assume isotropic
    diag = (scdecay[:,0,0]+scdecay[:,1,1]+scdecay[:,2,2])/3
    off_diag = (scdecay[:,0,1]+scdecay[:,1,2]+scdecay[:,0,2]+scdecay[:,1,0]+scdecay[:,2,1]+scdecay[:,2,0])/6
    
    # plot type dependence   
    plotx = np.array([0,1,2])
    plotxlab = typextick
    data = [plotxlab,diag,off_diag]
    
    fig,ax = plt.subplots()
    plt.plot(plotx,data[1],marker='s',markersize=6,linewidth=2.2,label='Normal')
    plt.plot(plotx,data[2],marker='s',markersize=6,linewidth=2.2,label='Parallel')
    ax.set_xticks(plotx)
    ax.set_xticklabels(plotxlab)
    plt.title("Tilting Correlation Length",fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #plt.xlabel('Br Content', fontsize=15)
    plt.ylabel('Correlation Length (unit cell)', fontsize=15)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return data


def vis3D_domain_anime(cfeat,frs,tstep,ss,bin_indices,figname):
    """ 
    Visualise tilting in 3D animation. Better visual effects can be done with professional software.

    Args:
        cfeat (numpy.ndarray): Color feature data.
        frs (list): Frame indices.
        tstep (float): Time step.
        ss (int): Supercell size of the grid.
        bin_indices (numpy.ndarray): Binned indices of B-sites.
        figname (str): Name of the figure file.

    Returns:
        None
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LightSource
    
    timeline1 = np.array(list(range(frs[-1]+frs[0])))*tstep-frs[0]*tstep
    
    x1, x2, x3 = np.indices((ss+1,ss+1,ss+1))
    grids = np.zeros((ss,ss,ss),dtype="bool")
    selind = list(set(list(np.where(bin_indices[2,:]==ss)[0])+list(np.where(bin_indices[0,:]==ss)[0])+list(np.where(bin_indices[1,:]==1)[0])))
    sel = bin_indices[:,selind]
    for i in range(len(selind)):
        grids[sel[:,i][0]-1,sel[:,i][1]-1,sel[:,i][2]-1] = True
    
    cmval = np.empty((cfeat.shape[0],ss,ss,ss,4))
    for j in range(cfeat.shape[1]):
        cmval[:,bin_indices[0,j]-1,bin_indices[1,j]-1,bin_indices[2,j]-1,:] = cfeat[:,j,:]
    
    def init():
        ls = LightSource(azdeg=80)
        return fig,

    def animate(i):
        ax.voxels(x1,x2,x3,grids,facecolors=cmval[i,:])
        timelabel.set_text(f"{round(timeline1[i],1)} ps")
        return fig,

    fig=plt.figure()
    ax = plt.axes(projection='3d')
    timelabel = ax.text2D(0.85, 0.10, f"{round(timeline1[0],1)} ps", ha='center', va='center', fontsize=14, color="k", transform=ax.transAxes)
    #ax.set_facecolor("grey")
    ax.axis('off')
    ax.set_aspect('auto')
    #ax.set_axis_off()

    anim = FuncAnimation(fig, animate, init_func=init, frames=frs, interval=200)
    writer = PillowWriter(fps=15)
    anim.save(f"{figname}.gif", writer=writer)
    #anim.save("link-anim.gif", writer=writer)
    plt.show()


def vis3D_domain_frame(cfeat,ss,bin_indices,cmap,clbedge,figname,saveFigures):
    """ 
    Visualise tilting in 3D for a single frame. 

    Args: 
        cfeat (numpy.ndarray): Color feature data.
        ss (int): Supercell size of the grid.
        bin_indices (numpy.ndarray): Binned indices of B-sites.
        cmap (str): Color map.
        clbedge (float): Color bar edge.
        figname (str): Name of the figure file.
        saveFigures (bool): Whether to save the figure.

    Returns:
        None
    """
    
    x1, x2, x3 = np.indices((ss+1,ss+1,ss+1))
    grids = np.zeros((ss,ss,ss),dtype="bool")
    selind = list(set(list(np.where(bin_indices[2,:]==ss)[0])+list(np.where(bin_indices[0,:]==ss)[0])+list(np.where(bin_indices[1,:]==1)[0])))
    sel = bin_indices[:,selind]
    for i in range(len(selind)):
        grids[sel[:,i][0]-1,sel[:,i][1]-1,sel[:,i][2]-1] = True
    
    cmval = np.empty((ss,ss,ss,4))
    for j in range(cfeat.shape[0]):
        cmval[bin_indices[0,j]-1,bin_indices[1,j]-1,bin_indices[2,j]-1,:] = cfeat[j,:]

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(x1,x2,x3,grids,facecolors=cmval)
    ax.axis('off')
    ax.set_aspect('auto')
    clb=plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=-clbedge, vmax=clbedge)),ax=ax)
    clb.set_label(label="Tilting (degree)")
    if saveFigures:
        plt.savefig(figname, dpi=350,bbox_inches='tight')
    plt.show()


def properties_to_binned_grid(T,D,tcorr,bc,ss,bin_indices):
    """ 
    Assign tilting and distortion to 3D grids.  

    Args:
        T (numpy.ndarray): Tilt data.
        D (numpy.ndarray): Distortion data.
        tcorr (numpy.ndarray): Time correlation data.
        bc (numpy.ndarray): B-site displacement data.
        ss (int): Supercell size of the grid.
        bin_indices (numpy.ndarray): Binned indices of B-sites.

    Returns:
        tuple: Gridded tilt, distortion, time correlation, and B-site displacement values.
    """
    
    tgval = np.empty((ss,ss,ss,3))
    dgval = np.empty((ss,ss,ss,7))
    tcval = np.empty((ss,ss,ss,3))
    bgval = np.empty((ss,ss,ss,3))
    for j in range(T.shape[0]):
        tgval[bin_indices[0,j]-1,bin_indices[1,j]-1,bin_indices[2,j]-1,:] = T[j,:]
        dgval[bin_indices[0,j]-1,bin_indices[1,j]-1,bin_indices[2,j]-1,:] = D[j,:]
        tcval[bin_indices[0,j]-1,bin_indices[1,j]-1,bin_indices[2,j]-1,:] = tcorr[j,:]
        bgval[bin_indices[0,j]-1,bin_indices[1,j]-1,bin_indices[2,j]-1,:] = bc[j,:]

    return tgval, dgval, tcval, bgval


def compute_tilt_domain(Corr, timestep, uniname, saveFigures, n_bins=42, tol=0, smoother=5):
    """ 
    Compute tilt domain lifetime.  

    Args:
        Corr (numpy.ndarray): Correlation data.
        timestep (float): Time step in picoseconds.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for the histogram.
        tol (float): Tolerance for zero crossing.
        smoother (int): Smoothing window size.

    Returns:
        None
    """
    
    fig_name = f"traj_tilt_domain_time_{uniname}.png"
    
    time_window = smoother # picosecond
    sgw = round(time_window/timestep)
    if sgw<5: sgw = 5
    if sgw%2==0: sgw+=1
    
# =============================================================================
#     fig,ax = plt.subplots()
#     plt.plot(np.array(list(range(Corr.shape[0])))*timestep,Corr[:,100,5])
#     plt.plot(np.array(list(range(Corr.shape[0])))*timestep,savitzky_golay(Corr[:,100,5],window_size=sgw))
#     ax.set_xlabel("Time (ps)")
#     ax.set_ylabel("TCP")
#     plt.axhline(y=0,linestyle="dashed",color="k")
# =============================================================================
    
    dom = [[],[],[],[],[],[]] # [a+,b+,c+,a-,b-,c-]
    for i in range(6):
        for j in range(Corr.shape[1]):
            tcline = Corr[:,j,i]
            if smoother != 0: 
                tcline = savitzky_golay(tcline,window_size=sgw)
            crosszero = np.diff(np.sign(tcline+tol))
            neg2pos = np.where(crosszero==2)[0]
            pos2neg = np.where(crosszero==-2)[0]
            if len(neg2pos) == 0 or len(pos2neg) == 0:
                if tol != 0:
                    raise ValueError("Tilt_domain: No zero crossing is found, lower 'tol' value. ")
                else:
                    neg = []
            else:
                if neg2pos[0] > pos2neg[0]:
                    t1 = neg2pos
                    t2 = pos2neg
                    ts = min(t1.shape[0],t2.shape[0])
                    neg = t1[:ts] - t2[:ts]
                else:
                    t1 = neg2pos[1:]
                    t2 = pos2neg
                    ts = min(t1.shape[0],t2.shape[0])
                    neg = t1[:ts] - t2[:ts]
                
            crosszero = np.diff(np.sign(tcline-tol))
            neg2pos = np.where(crosszero==2)[0]
            pos2neg = np.where(crosszero==-2)[0]
            if len(neg2pos) == 0 or len(pos2neg) == 0:
                if tol != 0:
                    raise ValueError("Tilt_domain: No zero crossing is found, lower 'tol' value. ")
                else:
                    pos = []
            else:
                if neg2pos[0] > pos2neg[0]:
                    t1 = pos2neg[1:]
                    t2 = neg2pos
                    ts = min(t1.shape[0],t2.shape[0])
                    pos = t1[:ts] - t2[:ts]
                else:
                    t1 = pos2neg
                    t2 = neg2pos
                    ts = min(t1.shape[0],t2.shape[0])
                    pos = t1[:ts] - t2[:ts]
            
            if i in (0,1):
                if len(pos) != 0:
                    dom[0].extend(list(pos*timestep))
                if len(neg) != 0:
                    dom[3].extend(list(neg*timestep))
            elif i in (2,3):
                if len(pos) != 0:
                    dom[1].extend(list(pos*timestep))
                if len(neg) != 0:
                    dom[4].extend(list(neg*timestep))
            elif i in (4,5):
                if len(pos) != 0:
                    dom[2].extend(list(pos*timestep))
                if len(neg) != 0:
                    dom[5].extend(list(neg*timestep))
    
    hist_filt = 0            
    
    legs = ["$a^{+}$","$b^{+}$","$c^{+}$","$a^{-}$","$b^{-}$","$c^{-}$"]
    colors = ["C0","C1","C2"]
    maxis = []
    for i in range(6):
        maxis.append(max(dom[i]))
    m = max(maxis)*1.1
    
# =============================================================================
#     plt.subplots(1,1)
#     ax = plt.gca()
#     for i in range(6):
#         temp,binEdges = np.histogram(dom[i],bins=n_bins,range=[hist_filt,m])
#         bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#         if i in (0,1,2):
#             ax.plot(bincenters,temp,label=legs[i],linewidth=2,color=colors[i%3],linestyle="solid")
#         else:
#             ax.plot(bincenters,temp,label=legs[i],linewidth=2,color=colors[i%3],linestyle="dotted")
#     
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     #ax.set_xlim([0,m])
#     plt.xlabel('Time (ps)', fontsize=14)
#     plt.ylabel('Count (a.u.)', fontsize=14)
#     #plt.legend(prop={'size': 11})
#     legend = plt.legend(prop={'size': 11},frameon = True, loc="upper right", ncol=2)
#     legend.get_frame().set_alpha(0.8)
# =============================================================================
    
    cpower = 1
    
    w, h = figaspect(1.2/1.5)
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False,figsize=(w,h))
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2"]
    for i in range(3):
        #axs[i].hist(dom[i],bins=n_bins,range=[hist_filt,m], color=colors[0], label="positive", alpha=0.5)
        #axs[i].hist(dom[i+3],bins=n_bins,range=[hist_filt,m], color=colors[1], label="negative", alpha=0.5)
        temp1,binEdges = np.histogram(dom[i],bins=n_bins,range=[hist_filt,m])
        temp2,binEdges = np.histogram(dom[i+3],bins=n_bins,range=[hist_filt,m])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        temp1 = np.multiply(temp1,np.power(bincenters,cpower))
        temp2 = np.multiply(temp2,np.power(bincenters,cpower))
        axs[i].fill_between(bincenters, temp1, 0, facecolor=colors[0], interpolate=True, alpha = 0.4, label="positive")
        axs[i].fill_between(bincenters, temp2, 0, facecolor=colors[1], interpolate=True, alpha = 0.4, label="negative")
        #axs[i].plot(bincenters, temp1, linewidth = 1.8, linestyle = 'solid', color=colors[0], label="positive")
        #axs[i].plot(bincenters, temp2, linewidth = 1.8, linestyle = 'solid', color=colors[1], label="negative")
        axs[i].text(0.04, 0.84, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes)

    for ax in axs.flat:
        ax.set_ylim(ymin=0)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xlim([hist_filt,m])
        ax.set_yticks([])
    
    axs[0].legend(prop={'size': 10}, loc="upper right")
    fig.text(0.5, 0.01, 'Time (ps)', ha='center', fontsize=14)
    fig.text(0.07, 0.5, 'Counts (a.u.)', va='center', rotation='vertical', fontsize=14)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    

def spherical_coordinates(cn):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        cn (numpy.ndarray): Cartesian coordinates (x, y, z).

    Returns:
        tuple: Spherical coordinates (theta, phi).
    """

#    cn=sorted(abs(cn),reverse=True) #This forces symmetry exploitation. Used for figuring out what [x,y,z] corresponds to which point in the figure
    l=np.linalg.norm(cn)
    x=cn[0]
    y=cn[1]
    z=cn[2]
    theta = math.acos(z/l)
    phi   = math.atan2(y,x)
    theta   = theta - math.pi/2 #to agree with Matplotlib view of angles...
    return (theta,phi)


def MO_correlation(cnsn,MDTimestep,SaveFigures,uniname):
    """
    Calculate the self-correlation of molecular vectors.

    Args:
        cnsn (numpy.ndarray): Molecular vectors.
        MDTimestep (float): Molecular dynamics time step.
        SaveFigures (bool): Whether to save the figure.
        uniname (str): User-defined name for printing and figure saving.

    Returns:
        tuple: Time and correlation values.
    """

    if cnsn.shape[0] > 1500:
        T=1000
    else:
        T=round(cnsn.shape[0]/2) 
    correlation=np.zeros(T)

    for carbon in range(cnsn.shape[1]): 
        for i in range(cnsn.shape[0]-T): 
            for dt in range(T): 
                correlation[dt]=correlation[dt]+np.dot(cnsn[i,carbon],cnsn[i+dt,carbon]) 
    correlation=correlation/((cnsn.shape[0]-T)) 

# =============================================================================
#     fig=plt.figure()
#     ax=fig.add_subplot(111)
# 
#     plt.plot(np.array(range(T))*MDTimestep,correlation)
# 
#     ax.set_title("Dot Product correlation of molecular vector")
#     ax.set_xlabel(r'$\Delta t$ (ps)')
#     ax.set_ylabel(r'$r_{T}.r_{T+\Delta t}$')
# 
#     plt.show()
#     
#     if (SaveFigures):
#         fig.savefig("%s_MO_correlation_averages.png"%uniname, dpi=300)
# =============================================================================
    
    x = np.array(range(T))*MDTimestep
    x = x.reshape(1,x.shape[0])
    y = correlation
    y = y.reshape(1,y.shape[0])
    y = y/np.amax(y)
    
    return x, y


def orientation_density_3D_dots(cnsn,moltype,SaveFigures,uniname,title=None):
    """
    Visualize the orientation density of molecular vectors in 3D.

    Args:
        cnsn (numpy.ndarray): Molecular vectors.
        moltype (str): Molecular type name.
        SaveFigures (bool): Whether to save the figure.
        uniname (str): User-defined name for printing and figure saving.
        title (str): Title for the plot.

    Returns:
        None
    """
    
    def fibonacci_sphere(samples):

        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append((x, y, z))

        return np.array(points)
    
    n_sample = 2000
    
    p = fibonacci_sphere(n_sample)
    sims = np.dot(cnsn.reshape(-1,3),p.T)
    pto = np.argmax(sims,axis=1)

    u,ind = np.unique(pto,return_counts=True)
    counts = np.zeros((n_sample,))
    for i,j in enumerate(u):
        counts[j] = ind[i]
    counts = counts.astype(int)
    cnorm = counts/np.amax(counts)
    
    #pnorm = np.multiply(cnorm.reshape(-1,1),p)
    #ax = plt.figure().add_subplot(projection='3d')
    #ax.plot_trisurf(pnorm[:,0], pnorm[:,1], pnorm[:,2], linewidth=0.2, antialiased=True)
    
    pplot = np.empty((0,3))
    for ip in range(n_sample):
        muls = np.linspace(0,cnorm[ip],1+round(cnorm[ip]/0.08))[1:]
        pplot = np.concatenate((pplot,muls.reshape(-1,1)*p[ip].reshape(1,3)),axis=0)
    
    box = 0.7

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(pplot[:,0],pplot[:,1],pplot[:,2],s = 7,alpha=0.5)
    
    limits = np.array([[-box, box],
                       [-box, box],
                       [-box, box]])

    v, e, f = get_cube(limits)
    ax.plot(*v.T, marker='o', color='k', ls='', markersize=10, alpha=0.5)
    for i, j in e:
        ax.plot(*v[[i, j], :].T, color='k', ls='-', lw=2, alpha=0.5)
    
    tol = 0.0001
    ax.set_xlim([-box-tol,box+tol])
    ax.set_ylim([-box-tol,box+tol])
    ax.set_zlim([-box-tol,box+tol])
    ax.view_init(30, 25)
    set_axes_equal(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax._axis3don = False

    plt.show()
    if (SaveFigures):
        fig.savefig(f"MO_{moltype}_orientation_density_3D_{uniname}.png",bbox_inches='tight', pad_inches=0,dpi=350)


def sphere_mesh(res = 80):
    """
    Create a sphere mesh for visualization.
    
    Args:
        res (int): Resolution of the sphere mesh.

    Returns:
        tuple: Mesh coordinates (x, y, z) and sampled points.
    """

    u = np.linspace(0, 2*np.pi, 2*res)
    v = np.linspace(0, np.pi, res )
    # create the sphere surface
    xmesh = np.outer(np.cos(u), np.sin(v))
    ymesh = np.outer(np.sin(u), np.sin(v))
    zmesh = np.outer(np.ones(np.size(u)), np.cos(v))
    points = np.concatenate((xmesh[:,:,np.newaxis],ymesh[:,:,np.newaxis],zmesh[:,:,np.newaxis]),axis=2)
    return xmesh, ymesh, zmesh, points


def sphere_bin_count(cnsn):
    """
    Count the number of molecular vectors in spherical bins.

    Args:
        cnsn (numpy.ndarray): Molecular vectors.

    Returns:
        tuple: Mesh coordinates (x, y, z), counts array, and variance (spread metric).
    """

    xmesh, ymesh, zmesh, points = sphere_mesh(80)
    #points = np.moveaxis(points,[2],[0])
    s = list(points.shape[:2])
    counting = np.zeros((s[0],s[1]))
    cnsn = cnsn.reshape(-1,3)
    for i in range(cnsn.shape[0]):
        dots = np.dot(points,cnsn[i,:])
        maxx = np.where(dots > 0.995)
        #maxx = np.where(dots==np.amax(dots))
        counting[list(maxx[0]),list(maxx[1])] += 1
    return xmesh, ymesh, zmesh, counting, np.var(counting/np.mean(counting))


def orientation_density_3D_sphere(cnsn,moltype,SaveFigures,uniname,title=None):
    """
    Visualize the orientation density of molecular vectors in 3D using a sphere.

    Args:
        cnsn (numpy.ndarray): Molecular vectors.
        moltype (str): Molecular type name.
        SaveFigures (bool): Whether to save the figure.
        uniname (str): User-defined name for printing and figure saving.
        title (str): Title for the plot.

    Returns:
        float: Variance of the counts (spread metric).
    """

    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib import cm

    def init():
        #v = np.array([[1, 1, 1], [1, 1, -1],
        #              [1, -1, 1], [-1, 1, 1],
        #              [-1, -1, 1], [-1, 1, -1],
        #              [1, -1, -1], [-1, -1, -1]], dtype=int)
        #ax.scatter(*v.T, color='k', s=100, alpha=0.25) #, depthshade=False
        ax.plot_surface(xmesh, ymesh, zmesh, alpha=0.8, cstride=1, rstride=1, facecolors=cm.plasma(myheatmap))
        return fig,

    def animate(i):
        ax.view_init(elev=20, azim=i)
        #ax.set_title(moltype,fontsize=14,animated=True)
        return fig,
    
    xmesh, ymesh, zmesh, points = sphere_mesh(80)
    s = list(points.shape[:2])
    counting = np.zeros((s[0],s[1]))
    cnsn = cnsn.reshape(-1,3)
    for i in range(cnsn.shape[0]):
        dots = np.dot(points,cnsn[i,:])
        maxx = np.where(dots > 0.995)
        counting[list(maxx[0]),list(maxx[1])] += 1
    myheatmap = counting / np.amax(counting)
    movar = np.var(counting/np.mean(counting))
    #xmesh, ymesh, zmesh, portion, movar = sphere_bin_count(cnsn)
    #myheatmap = portion / np.amax(portion)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(True)
    #ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([-1.05,1.05])
    ax.set_ylim([-1.05,1.05])
    ax.set_zlim([-1.05,1.05])
    anim = FuncAnimation(fig, animate, init_func=init, frames=list(np.linspace(0,360,91)), interval=30, blit=True)
    writer = PillowWriter(fps=25)
    anim.save(f"MO_{moltype}_orientation_density_3D_{uniname}.gif", writer=writer)
    plt.show()
    
    # also save a snapshot
    framecolor = 'grey'
    framelw = 1
    framebound = 1
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(True)
    ax.set_axis_off()
    ax.plot_surface(xmesh, ymesh, zmesh, alpha=0.8, cstride=1, rstride=1, facecolors=cm.plasma(myheatmap))
    ax.plot([-framebound,-framebound],[1.04,framebound],[-framebound,framebound],color=framecolor,linewidth=framelw)
    ax.plot([framebound,framebound],[framebound,framebound],[-framebound,framebound],color=framecolor,linewidth=framelw)
    ax.plot([framebound,framebound],[-framebound,-framebound],[-framebound,framebound],color=framecolor,linewidth=framelw)
    ax.plot([framebound,-framebound],[-framebound,-framebound],[-framebound,-framebound],color=framecolor,linewidth=framelw)
    ax.plot([framebound,-framebound],[framebound,framebound],[-framebound,-framebound],color=framecolor,linewidth=framelw)
    ax.plot([-framebound,-framebound],[framebound,-framebound],[-framebound,-framebound],color=framecolor,linewidth=framelw)
    ax.plot([framebound,framebound],[framebound,-framebound],[-framebound,-framebound],color=framecolor,linewidth=framelw)
    ax.plot([framebound,framebound],[framebound,-framebound],[framebound,framebound],color=framecolor,linewidth=framelw)
    ax.plot([framebound,-framebound],[framebound,framebound],[framebound,framebound],color=framecolor,linewidth=framelw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([-1.05,1.05])
    ax.set_ylim([-1.05,1.05])
    ax.set_zlim([-1.05,1.05])
    ax.view_init(elev=20, azim=225)
    fig.savefig(f"MO_{moltype}_3D_sphere_{uniname}.png",bbox_inches='tight', pad_inches=0,dpi=350)
    
    return movar

    
def orientation_density(cnsn,moltype,SaveFigures,uniname,title=None,miller_mask=False):
    """
    Visualize the orientation density of molecular vectors in 2D polar plot.

    Args:
        cnsn (numpy.ndarray): Molecular vectors.
        moltype (str): Molecular type name.
        SaveFigures (bool): Whether to save the figure.
        uniname (str): User-defined name for printing and figure saving.
        title (str): Title for the plot.
        miller_mask (bool): Whether to apply Miller indices masking.

    Returns:
        None
    """
    
    thetas=[] # List to collect data for later histogramming
    phis=[]

    thetasOh=[]
    phisOh=[]

    for frame in cnsn[:,:]:
        for cn in frame:

            theta,phi = spherical_coordinates(np.array(cn)) # Values used for ORIENTATION 
            thetas.append(theta) #append this data point to lists
            phis.append(phi)
 
            cn=abs(cn)
            cn=sorted(abs(cn),reverse=True) #Exploit Oh symmetry - see workings in Jarv's notebooks
            
            thetaOh,phiOh=spherical_coordinates(np.array(cn))
            thetasOh.append(thetaOh)
            phisOh.append(phiOh)

    w, h = figaspect(1/1.6)
    fig, ax = plt.subplots(figsize=(w,h))

    plt.hexbin(phis,thetas,gridsize=36,marginals=False,cmap=plt.cm.cubehelix_r) #PuRd) #cmap=plt.cm.jet)
    plt.title(title, fontsize = 16)
    
    if miller_mask:
        mil_111 = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,-1,-1]])
        mil_100 = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[-0.99,-0.01,0]])
        
        mil111thetas=[] # List to collect data for later histogramming
        mil111phis=[]
        for frame in mil_111[:,:]:
            miltheta,milphi = spherical_coordinates(frame) # Values used for ORIENTATION 
            mil111thetas.append(miltheta) #append this data point to lists
            mil111phis.append(milphi)
        mil100thetas=[] # List to collect data for later histogramming
        mil100phis=[]
        for frame in mil_100[:,:]:
            miltheta,milphi = spherical_coordinates(frame) # Values used for ORIENTATION 
            mil100thetas.append(miltheta) #append this data point to lists
            mil100phis.append(milphi)
        ax.scatter(mil111phis,mil111thetas,label='(111)',s=10,color='lime')
        ax.scatter(mil100phis,mil100thetas,label='(100)',s=10,color='gold')
        legend = ax.legend(prop={'size': 12},frameon = True, loc="upper right")
        legend.get_frame().set_alpha(0.7)
    
    cbar = plt.colorbar()
    #cbar.ax.set_yaxis('Intensity (a.u.)', fontsize=15)
    cbar.ax.set_ylabel('Intensity (a.u.)', fontsize=15, rotation=270, labelpad=20)
    cbar.set_ticks([])
    pi=np.pi

    plt.xticks( [-pi,-pi/2,0,pi/2,pi],
                [r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],
                fontsize=15)
    plt.yticks( [-pi/2,0,pi/2],
                [r'$-\pi/2$',r'$0$',r'$\pi/2$'],
                fontsize=15)
    ax.set_xlabel(r'$\mathit{\theta}$',fontsize=17)
    ax.set_ylabel(r'$\mathit{\phi}$',fontsize=17, rotation=0)
    ax.yaxis.set_label_coords(-0.12,0.46)
    plt.show()
    if (SaveFigures):
        fig.savefig("MO_MA_orientation_density_nosymm_%s.png"%uniname,bbox_inches='tight', pad_inches=0,dpi=350)
    
    # 2D density plot of the theta/phi information - MOLLWEIDE projection
# =============================================================================
#     fig=plt.figure()
#     ax=fig.add_subplot(111,projection = 'mollweide')
# 
#     plt.hexbin(phis,thetas,gridsize=36,marginals=False,cmap=plt.cm.cubehelix_r) #PuRd) #cmap=plt.cm.jet)
# 
#     cbar = plt.colorbar()
#     cbar.set_ticks([])
#     pi=np.pi
# 
#     plt.xticks([],[])
# 
#     plt.show()
# =============================================================================

    fig=plt.figure()
    fig.add_subplot(111)

    plt.hexbin(phisOh,thetasOh,gridsize=36,marginals=False,cmap=plt.cm.cubehelix_r) #PuRd) #cmap=plt.cm.jet)
    cbar = plt.colorbar()
    cbar.set_ticks([])
    pi=np.pi
   
    plt.xticks( [0.01,pi/4], 
                [r'$0$',r'$\pi/4$'],
                fontsize=14)

    plt.yticks( [-0.6154797,-0.01],
                [r'$-0.62$',r'$0$'],
                fontsize=14)
    
    plt.show()
    if (SaveFigures):
        fig.savefig(f"MO_{moltype}_orientation_density_Oh_symm_{uniname}.png",bbox_inches='tight', pad_inches=0,dpi=350)


def orientation_density_2pan(cnsn,nnsn,moltype,SaveFigures,uniname,title=None,miller_mask=True):
    """
    Visualize the orientation density of molecular vectors in 2D with two panels.

    Args:
        cnsn (numpy.ndarray): First molecular vectors.
        nnsn (numpy.ndarray): Secondary molecular vectors.
        moltype (str): Molecular type name.
        SaveFigures (bool): Whether to save the figure.
        uniname (str): User-defined name for printing and figure saving.
        title (str): Title for the plot.
        miller_mask (bool): Whether to apply Miller indices masking.

    Returns:
        None
    """
    
    thetas=[] # List to collect data for later histogramming
    phis=[]

    thetasOh=[]
    phisOh=[]

    for frame in cnsn[:,:]:
        for cn in frame:

            theta,phi = spherical_coordinates(np.array(cn)) # Values used for ORIENTATION 
            thetas.append(theta) #append this data point to lists
            phis.append(phi)
 
            cn=abs(cn)
            cn=sorted(abs(cn),reverse=True) #Exploit Oh symmetry - see workings in Jarv's notebooks
            
            thetaOh,phiOh=spherical_coordinates(np.array(cn))
            thetasOh.append(thetaOh)
            phisOh.append(phiOh)
            
    
    nnthetas=[] # List to collect data for later histogramming
    nnphis=[]

    nnthetasOh=[]
    nnphisOh=[]

    for frame in nnsn[:,:]:
        for nn in frame:

            nntheta,nnphi = spherical_coordinates(np.array(nn)) # Values used for ORIENTATION 
            nnthetas.append(nntheta) #append this data point to lists
            nnphis.append(nnphi)
 
            nn=abs(nn)
            nn=sorted(abs(nn),reverse=True) #Exploit Oh symmetry - see workings in Jarv's notebooks
            
            nnthetaOh,nnphiOh=spherical_coordinates(np.array(nn))
            nnthetasOh.append(nnthetaOh)
            nnphisOh.append(nnphiOh)

    #w, h = figaspect(2/1.2)
    #fig, axs = plt.subplots(figsize=(w,h),nrows=2, ncols=1,sharey=True)
    fig, axs = plt.subplots(nrows=2, ncols=1,sharey=True)

    axs[0].hexbin(phis,thetas,gridsize=42,marginals=False,cmap=plt.cm.cubehelix_r) #PuRd) #cmap=plt.cm.jet)
    axs[1].hexbin(nnphis,nnthetas,gridsize=42,marginals=False,cmap=plt.cm.cubehelix_r)
    
    if miller_mask:
        mil_111 = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,-1,-1]])
        mil_100 = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[-0.99,-0.01,0]])
        
        mil111thetas=[] # List to collect data for later histogramming
        mil111phis=[]
        for frame in mil_111[:,:]:
            miltheta,milphi = spherical_coordinates(frame) # Values used for ORIENTATION 
            mil111thetas.append(miltheta) #append this data point to lists
            mil111phis.append(milphi)
        mil100thetas=[] # List to collect data for later histogramming
        mil100phis=[]
        for frame in mil_100[:,:]:
            miltheta,milphi = spherical_coordinates(frame) # Values used for ORIENTATION 
            mil100thetas.append(miltheta) #append this data point to lists
            mil100phis.append(milphi)
        axs[0].scatter(mil111phis,mil111thetas,label='(111)',s=6,color='lime')
        axs[1].scatter(mil111phis,mil111thetas,s=6,color='lime')
        axs[0].scatter(mil100phis,mil100thetas,label='(100)',s=6,color='gold')
        axs[1].scatter(mil100phis,mil100thetas,s=6,color='gold')
        legend = axs[0].legend(prop={'size': 9},frameon = True, loc="upper right", ncol=2)
        legend.get_frame().set_alpha(0.5)
    
    axs[0].set_title(title, fontsize = 13)
    #cbar = plt.colorbar()
    #cbar.set_ticks([])
    pi=np.pi

    plt.sca(axs[0])
    plt.xticks( [-pi,-pi/2,0,pi/2,pi],
                [],
                fontsize=11)
    plt.yticks( [-pi/2,0,pi/2],
                [r'$-\pi/2$',r'$0$',r'$\pi/2$'],
                fontsize=11)
    plt.sca(axs[1])
    plt.xticks( [-pi,-pi/2,0,pi/2,pi],
                [r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'],
                fontsize=11)
    plt.yticks( [-pi/2,0,pi/2],
                [r'$-\pi/2$',r'$0$',r'$\pi/2$'],
                fontsize=11)
    axs[0].set_ylabel(r'$\mathit{\phi_{1}}$',fontsize=13, rotation=0)
    axs[1].set_ylabel(r'$\mathit{\phi_{2}}$',fontsize=13, rotation=0)
    axs[0].yaxis.set_label_coords(-0.12,0.42)
    axs[1].yaxis.set_label_coords(-0.12,0.42)
    axs[1].set_xlabel(r'$\mathit{\theta}$',fontsize=13, rotation=0)
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    fig.subplots_adjust(hspace = 0.06,left=0.12,right=0.90)
    plt.show()
    if (SaveFigures):
        fig.savefig(f"MO_{moltype}_orientation_density_nosymm_{uniname}.png",bbox_inches='tight', pad_inches=0,dpi=350)

    
    w, h = figaspect(2/1.2)
    fig, axs = plt.subplots(figsize=(w,h),nrows=2, ncols=1)

    axs[0].hexbin(phisOh,thetasOh,gridsize=36,marginals=False,cmap=plt.cm.cubehelix_r) #PuRd) #cmap=plt.cm.jet)
    axs[1].hexbin(nnphisOh,nnthetasOh,gridsize=36,marginals=False,cmap=plt.cm.cubehelix_r)
    #plt.title(title, fontsize = 16)
    #cbar = plt.colorbar()
    #cbar.set_ticks([])
    pi=np.pi

    plt.sca(axs[0])
    plt.xticks( [0.01,pi/4], 
                [r'$0$',r'$\pi/4$'],
                fontsize=14)
    plt.yticks( [-0.6154797,-0.01],
                [r'$-0.62$',r'$0$'],
                fontsize=14)
    plt.sca(axs[1])
    plt.xticks( [0.01,pi/4], 
                [r'$0$',r'$\pi/4$'],
                fontsize=14)
    plt.yticks( [-0.6154797,-0.01],
                [r'$-0.62$',r'$0$'],
                fontsize=14)

    #cbar = plt.colorbar()
    #cbar.set_ticks([])
    #plt.xlabel(r'$\mathit{\theta}$',fontsize=17)
    plt.show()
    if (SaveFigures):
        fig.savefig(f"MO_{moltype}_orientation_density_Oh_symm_{uniname}.png",bbox_inches='tight', pad_inches=0,dpi=350)


def get_norm_corr(TC,T):
    """
    Calculate normalized correlation from tensor components, converting 6 neighbours to three principle directions.

    Args:
        TC (numpy.ndarray): Tilting correlation components.
        T (numpy.ndarray): Tilting magnitudes.

    Returns:
        numpy.ndarray: Normalized correlation values.
    """
    T = np.abs(T)
    v1 = np.mean(np.divide(TC[:,:,[0,1]],T[:,:,[0]]),axis=2)[:,:,np.newaxis]
    v2 = np.mean(np.divide(TC[:,:,[2,3]],T[:,:,[1]]),axis=2)[:,:,np.newaxis]
    v3 = np.mean(np.divide(TC[:,:,[4,5]],T[:,:,[2]]),axis=2)[:,:,np.newaxis]
    return np.concatenate((v1,v2,v3),axis=2)


def get_tcp_from_list(TC):
    """
    Calculate TCP from a list of tensor components.

    Args:
        TC (list of numpy.ndarray): Tilting correlation components.

    Returns:
        numpy.ndarray: TCP values.
    """
    corr_power = 2.5
    p = []
    for g in TC:
        v = []
        v.append(g[:,:,[0,1]].reshape(-1,))
        v.append(g[:,:,[2,3]].reshape(-1,))
        v.append(g[:,:,[4,5]].reshape(-1,))
        por = []
        for vi in v:
            vmax = np.amax(np.abs(vi))
            yi,binEdges=np.histogram(vi,bins=100,range=[-vmax,vmax]) 
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            parneg = np.sum(np.power(yi,corr_power)[bincenters<0])
            parpos = np.sum(np.power(yi,corr_power)[bincenters>0])
            por.append((-parneg+parpos)/(parneg+parpos))
        p.append(por)
    p = np.array(p)
    return p
    

def fit_exp_decay(x,y,allow_redo=True):
    """
    Fit an exponential decay to the given data.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.
        allow_redo (bool): Whether to allow a second fitting attempt.

    Returns:
        float: The fitted decay constant.
    """

    from scipy.optimize import curve_fit
    def model_func(x, a, k1, k2):
        return a * np.exp(-k1*x) + (1-a) * np.exp(-k2*x)
        #return a * np.exp(-k*x) + b
    def model_func1(x, a, k1):
        return a * np.exp(-k1*x)
        #return a * np.exp(-k*x) + b
        
    x = np.squeeze(x)
    y = np.squeeze(y)/np.amax(y)
    
    fitrange = 1
    xc = x[:round(x.shape[0]*fitrange)]
    yc = y[:round(y.shape[0]*fitrange)]
    
    p0 = (0.90,20,0.5) # starting search coeffs
    opt, pcov = curve_fit(model_func, xc, yc, p0)
    a, k1 ,k2= opt

    y2 = model_func(x, a, k1 ,k2)
    fig, ax = plt.subplots()
    #ax.plot(x, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
    ax.plot(x, y2, color='r',label='fit',linewidth=3)
    ax.plot(x[::10], y[::10], 'bo', label='raw data',alpha=0.63,markersize=4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.title('Two-component fitting', fontsize=14)
    plt.show()
    
    redo = False
    if a < 0.35:
        k = k2
    elif a < 0.65:
        print(f"!fit_exp_decay: The fitted coefficient A={round(a,3)} is not reasonable, another fitting is performed.  ")
        redo = True
        if a > 0.5: 
            k = k1
        else:
            k = k2
    else:
        k = k1
    
    if redo and allow_redo:
        p0 = (0.90,5) # starting search coeffs
        opt, pcov = curve_fit(model_func1, xc, yc, p0)
        a, k= opt

        y2 = model_func1(x, a, k)
        fig, ax = plt.subplots()
        #ax.plot(x, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
        ax.plot(x, y2, color='r',label='fitted',linewidth=4)
        ax.plot(x, y, 'bo', label='raw data',alpha=0.13,markersize=4)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('Time (ps)', fontsize=14)
        plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
        plt.legend(prop={'size': 12})
        plt.title('One-component fitting', fontsize=14)
        plt.show()
        
    return 1/k


def fit_exp_decay_both(x,y):
    """
    Fit a two-component exponential decay to the given data.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.

    Returns:
        tuple: The fitted decay constants and coefficients.
    """

    from scipy.optimize import curve_fit
    def model_func(x, a, k1, k2):
        return a * np.exp(-k1*x) + (1-a) * np.exp(-k2*x)
        #return a * np.exp(-k*x) + b
        
    x = np.squeeze(x)
    y = np.squeeze(y)/np.amax(y)
    
    fitrange = 1
    xc = x[:round(x.shape[0]*fitrange)]
    yc = y[:round(y.shape[0]*fitrange)]
    
    p0 = (0.90,20,0.5) # starting search coeffs
    opt, pcov = curve_fit(model_func, xc, yc, p0)
    a, k1 ,k2= opt

    y2 = model_func(x, a, k1 ,k2)
    fig, ax = plt.subplots()
    #ax.plot(x, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
    ax.plot(x, y2, color='r',label='fitted',linewidth=4)
    ax.plot(x, y, 'bo', label='raw data',alpha=0.13,markersize=4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.title('Two-component fitting', fontsize=14)
    plt.show()
        
    return (1/k1,1/k2,a)


def fit_exp_decay_both_correct(x,y):
    """
    Fit a two-component exponential decay to the given data with correction term.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.

    Returns:
        tuple: The fitted decay constants, coefficients, and correction term.
    """

    from scipy.optimize import curve_fit
    def model_func(x, a, k1, k2, c):
        return a * np.exp(-k1*x) + (1-a) * np.exp(-k2*x) + c
        #return a * np.exp(-k*x) + b
        
    x = np.squeeze(x)
    y = np.squeeze(y)/np.amax(y)
    
    fitrange = 1
    xc = x[:round(x.shape[0]*fitrange)]
    yc = y[:round(y.shape[0]*fitrange)]
    
    p0 = (0.90,20,0.5,-0.03) # starting search coeffs
    opt, pcov = curve_fit(model_func, xc, yc, p0)
    a, k1 ,k2, c= opt

    y2 = model_func(x, a, k1 ,k2, c)
    fig, ax = plt.subplots()
    #ax.plot(x, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
    ax.plot(x, y2, color='r',label='fitted',linewidth=4)
    ax.plot(x, y, 'bo', label='raw data',alpha=0.13,markersize=4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.title('Two-component fitting', fontsize=14)
    plt.show()
    
    if abs(c) > 0.2:
        raise ValueError(f"The correction term c is too far from zero ({round(c,4)}).")
        
    return (1/k1,1/k2,a,c)


def fit_exp_decay_fixed(x,y,aconst = 0.9):
    """
    Fit a two-component exponential decay with a fixed coefficient to the given data.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.
        aconst (float): Fixed coefficient for the first exponential term.

    Returns:
        float: The fitted decay constant.
    """

    from scipy.optimize import curve_fit
    def model_func(x, k1, k2):
        return aconst * np.exp(-k1*x) + (1-aconst) * np.exp(-k2*x)
        #return a * np.exp(-k*x) + b
        
    x = np.squeeze(x)
    y = np.squeeze(y)/np.amax(y)
    
    fitrange = 1
    xc = x[:round(x.shape[0]*fitrange)]
    yc = y[:round(y.shape[0]*fitrange)]
    
    p0 = (5,0.1) # starting search coeffs
    opt, pcov = curve_fit(model_func, xc, yc, p0)
    k1 ,k2= opt

    y2 = model_func(x, k1 ,k2)
    fig, ax = plt.subplots()
    #ax.plot(x, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
    ax.plot(x, y2, color='r',label='fitted',linewidth=4)
    ax.plot(x, y, 'bo', label='raw data',alpha=0.13,markersize=4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.title('Two-component fitting', fontsize=14)
    plt.show()
    
    if k1 < k2:
        raise ValueError('The fitted Time constant k1 is smaller than k2. ')
    
    return 1/k1



def fit_exp_decay_single(x,y):
    """
    Fit a single exponential decay to the given data.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.

    Returns:
        float: The fitted decay constant.
    """

    from scipy.optimize import curve_fit
    def model_func1(x, a, k1):
        return a * np.exp(-k1*x)
        #return a * np.exp(-k*x) + b
        
    x = np.squeeze(x)
    y = np.squeeze(y)/np.amax(y)
    
    fitrange = 1
    xc = x[:round(x.shape[0]*fitrange)]
    yc = y[:round(y.shape[0]*fitrange)]
    
    p0 = (0.90,5) # starting search coeffs
    opt, pcov = curve_fit(model_func1, xc, yc, p0)
    a, k= opt

    y2 = model_func1(x, a, k)
    fig, ax = plt.subplots()
    #ax.plot(x, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
    ax.plot(x, y2, color='r',label='fitted',linewidth=4)
    ax.plot(x, y, 'bo', label='raw data',alpha=0.13,markersize=4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.title('One-component fitting', fontsize=14)
    plt.show()
        
    return 1/k


def fit_damped_oscillator(x,y):
    """
    Fit a damped oscillator model to the given data.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.

    Returns:
        float: The fitted frequency.
    """

    from scipy.optimize import curve_fit
    #def model_func1(x, omega, gamma):
    #    tau = 2/gamma
    #    omega_e = np.sqrt(omega**2-gamma**2/4)
    #    return np.exp(-x/tau) * (np.cos(omega_e*x)+gamma/(2*omega_e)*np.sin(omega_e*x))    
    #def model_func2(x, tau_l, tau_s):
    #    return 1/(tau_l-tau_s)*(tau_l*np.exp(-x/tau_l)-tau_s*np.exp(-x/tau_s))
    
    def model_func1(t, A, omega, gamma):
        return A * np.exp(-gamma * t / 2) * (np.cos(omega * t) + (gamma / (2 * omega)) * np.sin(omega * t))  
    def model_func2(t, A, omega, gamma):
        return A * np.exp(-gamma * t / 2) * (np.cosh(np.abs(omega) * t) + (gamma / (2 * np.abs(omega))) * np.sinh(np.abs(omega) * t))
    
    
    x = np.squeeze(x)
    y = np.squeeze(y)/np.amax(y)
    
    fitrange = 1
    xc = x[:round(x.shape[0]*fitrange)]
    yc = y[:round(y.shape[0]*fitrange)]
    
    p0 = (1,3,2) # starting search coeffs
    opt, pcov = curve_fit(model_func1, xc, yc, p0)
    A1, omega1, gamma1 = opt
    opt, pcov = curve_fit(model_func2, xc, yc, p0)
    A2, omega2, gamma2 = opt

    y1 = model_func1(x, A1, omega1, gamma1)
    y2 = model_func2(x, A2, omega2, gamma2)
    fig, ax = plt.subplots()
    ax.plot(x, y1, color='C0',label='underdamped',linewidth=4)
    ax.plot(x, y2, color='C1',label='overdamped',linewidth=4)
    ax.plot(x, y, 'bo', label='raw data',alpha=0.5,markersize=4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.title('Damped Harmonic Oscillator Fitting', fontsize=14)
    plt.show()
    
    if np.sum(np.abs(y1-y)) > np.sum(np.abs(y2-y)):
        return np.sqrt(omega2**2+(gamma2**2)/4)
    else:
        return np.sqrt(omega1**2+(gamma1**2)/4)
    


def draw_MO_spatial_corr_time(C, steps, uniname, saveFigures, smoother = 0, n_bins = 50):
    """
    Draw the evolution of spatial correlation of molecular orientations over time.

    Args:
        C (numpy.ndarray): Correlation data.
        steps (numpy.ndarray): Time steps.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): Smoothing window size.
        n_bins (int): Number of bins for histogram.

    Returns:
        tuple: The time axis and the correlation data.
    """
    
    fig_name = f"traj_MO_time_{uniname}.png"
    
    def three_diff(a,b,c):
        return(np.abs(a-b)**0.5+np.abs(c-b)**0.5+np.abs(a-c)**0.5)/3#**0.5
    def central_map(x,expo = 1.1):
        if x>0.5:
            #a=x-0.5
            #return 0.5+a**expo
            return x
        else:
            return 0.5-(0.5-x)**expo
    def hist_diff(h1,h2,expo = 1.0,standard = False):
        hs1 = np.power(h1,expo)
        hs2 = np.power(h2,expo)
        if standard:
            hs1=hs1/np.amax(hs1)
            hs2=hs2/np.amax(hs2)
        return sum(np.minimum(hs1,hs2))/sum(hs1)
        #return sum(np.minimum(hs1,hs2))/(sum(hs1)+sum(hs2)-sum(np.minimum(hs1,hs2)))
        
    plotobj = []
    time_window = smoother # picosecond
    ts = steps[1] - steps[0]
    sgw = round(time_window/ts)
    if sgw<5: sgw = 5
    if sgw%2==0: sgw+=1
    #Cline = np.empty((0,2,3))
    Cline = np.empty((0,4))

    aw = 11 # careful when tuning this
       
    for i in range(C.shape[2]-aw+1):
        temp00 = C[0,0,list(range(i,i+aw)),:].reshape(-1,)
        temp01 = C[0,1,list(range(i,i+aw)),:].reshape(-1,)
        temp02 = C[0,2,list(range(i,i+aw)),:].reshape(-1,)
        temp10 = C[1,0,list(range(i,i+aw)),:].reshape(-1,)
        temp11 = C[1,1,list(range(i,i+aw)),:].reshape(-1,)
        temp12 = C[1,2,list(range(i,i+aw)),:].reshape(-1,)

        y00,binEdges = np.histogram(temp00,bins=n_bins,range=[-1,1]) 
        y01,binEdges = np.histogram(temp01,bins=n_bins,range=[-1,1]) 
        y02,binEdges = np.histogram(temp02,bins=n_bins,range=[-1,1]) 
        y10,binEdges = np.histogram(temp10,bins=n_bins,range=[-1,1]) 
        y11,binEdges = np.histogram(temp11,bins=n_bins,range=[-1,1]) 
        y12,binEdges = np.histogram(temp12,bins=n_bins,range=[-1,1]) 
        #bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

        sim0 = (hist_diff(y00,y01)+hist_diff(y00,y02))/2
        sim1 = (hist_diff(y01,y00)+hist_diff(y01,y02))/2
        sim2 = (hist_diff(y02,y01)+hist_diff(y02,y00))/2
        invfac = (sum(np.minimum(y00,y10))/sum(y00)+sum(np.minimum(y01,y11))/sum(y01)+sum(np.minimum(y02,y12))/sum(y02))/3
        
        moavg = np.array([[central_map(sim0),central_map(sim1),central_map(sim2),invfac]])
 
        Cline = np.concatenate((Cline,moavg),axis=0)
    
    colors = ["C0","C1","C2","C3"]   
    labels = [r'$\mathit{a}$', r'$\mathit{b}$', r'$\mathit{c}$']
    lwid = 2.2
    
    w, h = figaspect(0.8/1.45)
    plt.subplots(figsize=(w,h))
    ax = plt.gca()
    
    plotobj.append(steps[:(len(steps)-aw+1)])
    for i in range(3):
        if smoother !=0:
            temp = savitzky_golay(Cline[:,i],window_size=sgw)
        else:
            temp = Cline[:,i]
        plt.plot(steps[:(len(steps)-aw+1)], temp, label = labels[i] ,color =colors[i], linewidth=lwid) 
        plotobj.append(temp)
        
    if smoother !=0 :
        temp = savitzky_golay(Cline[:,3],window_size=sgw)
    else:
        temp = Cline[:,3]
    plt.plot(steps[:(len(steps)-aw+1)], temp, label = 'CF' ,color ='k',linestyle='dashed', linewidth=lwid) 
    plotobj.append(temp)

    ax.set_xlim([0,np.amax(steps)])
    ax.set_ylim([0,1])
    #ax.set_yticks([-45,-30,-15,0,15,30,45])
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Time (ps)', fontsize=15)
    plt.ylabel('MO order (a.u.)', fontsize=15)
    plt.legend(prop={'size': 13})
    
    #print(np.mean(Cline[:,0]),np.mean(Cline[:,1]),np.mean(Cline[:,2]),np.mean(Cline[:,3]))
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    
    return tuple(plotobj)


def draw_MO_order_time(C, steps, uniname, saveFigures, smoother = 0, n_bins = 50):
    """
    Draw the evolution of molecular order over time.

    Args:
        C (numpy.ndarray): Correlation data.
        steps (numpy.ndarray): Time steps.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        smoother (int): Smoothing window size.
        n_bins (int): Number of bins for histogram.

    Returns:
        tuple: The time axis and the correlation data.
    """
    
    fig_name = f"traj_MO_order_time_{uniname}.png"
    
    def polar_param(y,bc,power=2):
        #parneg = sum(np.power(y,power)[bc<-0.5])+sum(np.power(y,power)[bc<0])
        #parpos = sum(np.power(y,power)[bc>0.5])+sum(np.power(y,power)[bc>0])
        parneg = sum(np.multiply(np.power(y,power)[bc<0],np.abs(bc[bc<0])))
        parpos = sum(np.multiply(np.power(y,power)[bc>0],np.abs(bc[bc>0])))
        return (-parneg+parpos)/(parneg+parpos)
    
    def contrast_corr(y00,y01,y02,y10,y11,y12,power=1.5):
        y00 = np.power(y00,power)
        y01 = np.power(y01,power)
        y02 = np.power(y02,power)
        y10 = np.power(y10,power)
        y11 = np.power(y11,power)
        y12 = np.power(y12,power)
        val = (sum(np.minimum(y00,y10))/sum(y00)+sum(np.minimum(y01,y11))/sum(y01)+sum(np.minimum(y02,y12))/sum(y02))/3
        return (val-0.5)*2
    
    ts = steps[1]-steps[0]
    time_window = smoother # picosecond
    sgw = round(time_window/ts)
    if sgw<5: sgw = 5
    if sgw%2==0: sgw+=1
    plotobj = []

    Cline = np.empty((0,4))

    aw = 11 # careful when tuning this
       
    for i in range(C.shape[2]-aw+1):
        temp00 = C[0,0,list(range(i,i+aw)),:].reshape(-1,)
        temp01 = C[0,1,list(range(i,i+aw)),:].reshape(-1,)
        temp02 = C[0,2,list(range(i,i+aw)),:].reshape(-1,)
        temp10 = C[1,0,list(range(i,i+aw)),:].reshape(-1,)
        temp11 = C[1,1,list(range(i,i+aw)),:].reshape(-1,)
        temp12 = C[1,2,list(range(i,i+aw)),:].reshape(-1,)

        y00,binEdges = np.histogram(temp00,bins=n_bins,range=[-1,1]) 
        y01,binEdges = np.histogram(temp01,bins=n_bins,range=[-1,1]) 
        y02,binEdges = np.histogram(temp02,bins=n_bins,range=[-1,1]) 
        y10,binEdges = np.histogram(temp10,bins=n_bins,range=[-1,1]) 
        y11,binEdges = np.histogram(temp11,bins=n_bins,range=[-1,1]) 
        y12,binEdges = np.histogram(temp12,bins=n_bins,range=[-1,1]) 
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

        sim0 = polar_param(y00,bincenters)
        sim1 = polar_param(y01,bincenters)
        sim2 = polar_param(y02,bincenters)
        invfac = contrast_corr(y00,y01,y02,y10,y11,y12)
        
        #moavg = np.array([[central_map(sim0),central_map(sim1),central_map(sim2),invfac]])
        moavg = np.array([[sim0,sim1,sim2,invfac]])
 
        Cline = np.concatenate((Cline,moavg),axis=0)
    
    colors = ["C0","C1","C2","C3"]   
    labels = [r'$\mathit{a}$', r'$\mathit{b}$', r'$\mathit{c}$']
    lwid = 2.2
    
    w, h = figaspect(0.8/1.45)
    plt.subplots(figsize=(w,h))
    ax = plt.gca()
    
    plotobj.append(steps[:(len(steps)-aw+1)])
    for i in range(3):
        if smoother != 0:
            temp = savitzky_golay(Cline[:,i],window_size=sgw)
        else:
            temp = Cline[:,i]
        plt.plot(steps[:(len(steps)-aw+1)], temp, label = labels[i] ,color =colors[i], linewidth=lwid) 
        plotobj.append(temp)
        
    if smoother != 0:
        temp = savitzky_golay(Cline[:,3],window_size=sgw)
    else:
        temp = Cline[:,3]
    plt.plot(steps[:(len(steps)-aw+1)], temp, label = 'CF' ,color ='k',linestyle='dashed', linewidth=lwid) 
    plotobj.append(temp)

    ax.set_xlim([0,np.amax(steps)])
    ax.set_ylim([-1,1])
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Time (ps)', fontsize=15)
    plt.ylabel('MO order (a.u.)', fontsize=15)
    legend = plt.legend(prop={'size': 13},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.8)
    
    #print(np.mean(Cline[:,0]),np.mean(Cline[:,1]),np.mean(Cline[:,2]),np.mean(Cline[:,3]))
        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return tuple(plotobj)


def draw_MO_spatial_corr_NN12(C, uniname, saveFigures, n_bins = 100):
    """ 
    Draw the spatial correlation of molecular orientations for NN1 and NN2.

    Args:
        C (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for histogram.

    Returns:
        None
    """
    
    fig_name = f"traj_MO_spatial_corr_NN12_{uniname}.png"
    
    if C.ndim == 3:
        pass
    elif C.ndim == 4:
        C = C.reshape(C.shape[0],C.shape[1],-1)
    else:
        raise TypeError("The dimension of C matrix is not correct. ")
        
    w, h = figaspect(1.2/1.5)
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False,figsize=(w,h))
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2","C3"]
    for i in range(3):
        temp1,binEdges = np.histogram(C[0][i],bins=n_bins,range=[-1,1])
        temp2,binEdges = np.histogram(C[1][i],bins=n_bins,range=[-1,1])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        
        temp1 = np.power(temp1,0.5) # normalise for better comparison
        temp2 = np.power(temp2,0.5)
        
        axs[i].fill_between(bincenters, temp1, 0,facecolor=colors[i], interpolate=True,alpha = 0.4)
        axs[i].plot(bincenters, temp2, linewidth = 2.4, linestyle = 'solid', color=colors[i])
        #axs[i].fill_between(bincenters, temp1, 0,facecolor=colors[i], interpolate=True)
        axs[i].text(0.04, 0.84, labels[i], horizontalalignment='center', fontsize=16, verticalalignment='center', transform=axs[i].transAxes)

    for ax in axs.flat:
        ax.set_ylim(ymin=0)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xlim([-1,1])
        ax.set_xticks([-1,-0.5,0,0.5,1])
        ax.set_yticks([])

    
    fig.text(0.5, 0.01, r'$\mathit{w}$', ha='center', fontsize=14)
    fig.text(0.03, 0.5, r'$\mathit{C_{\alpha}^{(k)}(w)}$ (a.u.)', va='center', rotation='vertical', fontsize=14)

        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()


def draw_MO_spatial_corr(C, uniname, saveFigures, n_bins = 50):
    """
    Draw the spatial correlation of molecular orientations.

    Args:
        C (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for histogram.

    Returns:
        None
    """
    
    fig_name = f"traj_MO_spatial_corr_{uniname}.png"
    
    if C.ndim == 3:
        pass
    elif C.ndim == 4:
        C = C.reshape(C.shape[0],C.shape[1],-1)
    else:
        raise TypeError("The dimension of C matrix is not correct. ")
        
    num_lens = C.shape[0]
    
    fig, axs = plt.subplots(nrows=3, ncols=num_lens, sharex=False, sharey=True)
    labels = [r'$\mathit{a}$',r'$\mathit{b}$',r'$\mathit{c}$']
    colors = ["C0","C1","C2","C3"]
    for i in range(3):
        for j in range(num_lens):
            axs[i,j].hist(C[j][i],bins=n_bins,range=[-1,1],orientation='horizontal')
            if j == 0:
                axs[i,j].text(0.1, 0.86, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i,j].transAxes)
            if i == 0:
                axs[i,j].text(0.16, 1.1, "NN"+str(j+1), horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i,j].transAxes)
        
    for ax in axs.flat:

        ax.set_ylim([-1.1,1.1])
        ax.set_yticks([-1,-0.5,0,0.5,1])
        ax.set_xticks([])

    
    fig.text(0.5, 0.07, 'Counts (a.u.)', ha='center', fontsize=12)
    fig.text(0.025, 0.5, 'MO correlation', va='center', rotation='vertical', fontsize=12)

        
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()


def draw_MO_spatial_corr_norm_var(C, uniname, saveFigures, n_bins=30):
    """
    Draw the normalized spatial correlation of molecular orientations.

    Args:
        C (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for histogram.

    Returns:
        numpy.ndarray: The normalized spatial correlation data.
    """
    
    fig_name = f"traj_MO_spatial_corr_norm_{uniname}.png"
    
    assert C.ndim == 4
    
    a1 = np.zeros((n_bins,))
    a1[0] = 1
    var0 = np.var(a1)
    
    m = np.empty((C.shape[0]+1,C.shape[1]))
    m[0,:] = var0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            temp,_ = np.histogram(C[i][j],bins=n_bins,range=[-1,1])
            temp = temp/np.sum(temp)
            m[i+1,j] = np.var(np.power(temp,0.5))
    m = m/np.amax(m)
            
    nns = C.shape[0]+1
    
    #thr = 0.015
    #sc[np.abs(sc)>thr] = (np.sqrt(np.abs(sc))*np.sign(sc))[np.abs(sc)>thr]
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),m[:,0],linewidth=1.5,label='axis 0')
    plt.plot(list(range(nns)),m[:,1],linewidth=1.5,label='axis 1')
    plt.plot(list(range(nns)),m[:,2],linewidth=1.5,label='axis 2')
    #plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    ax.set_ylim([0,1])
    ax.set_xticks(np.arange(C.shape[0]+1))
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    
# =============================================================================
#     from scipy.optimize import curve_fit
#     def model_func1(x, a, k1):
#         return a * np.exp(-k1*x)
#     
#     pop_warning = []
#     scabs = np.abs(m)
#     scdecay = np.empty((3,))
#     for i in range(3):
#         tc = scabs[:,i]
#         p0 = (0.90,5) # starting search coeffs
#         opt, pcov = curve_fit(model_func1, np.array(list(range(nns))), tc, p0)
#         a, k= opt
#         if a < 0.85 or a > 1.05:
#             pop_warning.append(a)
#             
#         scdecay[i] = (1/k)
#             
#     if pop_warning:
#         print(f"!Tilt Spatial Corr: fitting of decay length may be wrong, a value(s): {pop_warning}")
# =============================================================================
    
    return m

    
def draw_MO_spatial_corr_norm(C, uniname, saveFigures, n_bins=30):
    """
    Draw the normalized spatial correlation of molecular orientations.

    Args:   
        C (numpy.ndarray): Correlation data.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for histogram.

    Returns:
        numpy.ndarray: The normalized spatial correlation data.
    """
    
    fig_name = f"traj_MO_spatial_corr_norm_{uniname}.png"
    
    assert C.ndim == 4
    
    a1 = np.zeros((n_bins,))
    a1[0] = 1
    var0 = np.var(a1)
    
    m = np.empty((C.shape[0]+1,C.shape[1]))
    m[0,:] = var0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            temp,_ = np.histogram(C[i][j],bins=n_bins,range=[-1,1])
            temp = temp/np.sum(temp)
            m[i+1,j] = np.var(np.power(temp,0.5))
    m = m/np.amax(m)
            
    nns = C.shape[0]+1
    
    #thr = 0.015
    #sc[np.abs(sc)>thr] = (np.sqrt(np.abs(sc))*np.sign(sc))[np.abs(sc)>thr]
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),m[:,0],linewidth=1.5,label='axis 0')
    plt.plot(list(range(nns)),m[:,1],linewidth=1.5,label='axis 1')
    plt.plot(list(range(nns)),m[:,2],linewidth=1.5,label='axis 2')
    #plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    ax.set_ylim([0,1])
    ax.set_xticks(np.arange(C.shape[0]+1))
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    
# =============================================================================
#     from scipy.optimize import curve_fit
#     def model_func1(x, a, k1):
#         return a * np.exp(-k1*x)
#     
#     pop_warning = []
#     scabs = np.abs(m)
#     scdecay = np.empty((3,))
#     for i in range(3):
#         tc = scabs[:,i]
#         p0 = (0.90,5) # starting search coeffs
#         opt, pcov = curve_fit(model_func1, np.array(list(range(nns))), tc, p0)
#         a, k= opt
#         if a < 0.85 or a > 1.05:
#             pop_warning.append(a)
#             
#         scdecay[i] = (1/k)
#             
#     if pop_warning:
#         print(f"!Tilt Spatial Corr: fitting of decay length may be wrong, a value(s): {pop_warning}")
# =============================================================================
    
    return m


def draw_RDF(da, rdftype, uniname, saveFigures, n_bins=200):
    """
    Draw the radial distribution function (RDF) histogram.

    Args:
        da (numpy.ndarray): Data for RDF.
        rdftype (str): Type of RDF ('CN' or 'BX').
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for histogram.

    Returns:
        None
    """
    
    if rdftype == "CN":
        fig_name = f"RDF_CN_{uniname}.png"
        title = 'C-N RDF'
        histrange = [1.38,1.65]
    elif rdftype == "BX":
        fig_name = f"RDF_BX_{uniname}.png"
        title = 'B-X RDF'
        histrange = [0,5]
    
    fig, ax = plt.subplots()
    counts,binedge,_ = ax.hist(da,bins=n_bins,range=histrange)
    #mu, std = norm.fit(da)
    
    ax.set_yticks([])
    #ax.text(0.852, 0.94, 'Mean: %.4f' % mu, horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
    #ax.text(0.878, 0.84, 'SD: %.4f' % std, horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
    if not title is None:
        ax.text(0.14, 0.92, title, horizontalalignment='center', fontsize=15, verticalalignment='center', transform=ax.transAxes)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Bond Length ($\AA$)', fontsize=15)
    plt.ylabel('counts (a.u.)', fontsize=15)    
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight') 
    
    plt.show()


def fit_3D_disp_atomwise(disp,readTimestep,uniname,moltype,saveFigures,n_bins=50,title=None):
    """ 
    A-site displacement calculation to extract vibration of atoms about their average position.  

    Args:
        disp (numpy.ndarray): Displacement data.
        readTimestep (float): Time step for reading data.
        uniname (str): User-defined name for printing and figure saving.
        moltype (str): Type of molecule.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for histogram.
        title (str): Title for the plot.

    Returns:
        numpy.ndarray: The peaks of the displacement data.
    """
    from scipy.fft import fft, fftfreq
    fig_name=f"traj_A_vib_center_{moltype}_{uniname}.png"
    
    histlim = 1
    
    peaks = np.empty((disp.shape[1],3))
    for i in range(disp.shape[1]):
        for j in range(3):
            peaks[i,j] = norm.fit(disp[:,i,j].reshape(-1))[0]
    
    extremes = max(float(np.abs(np.amin(peaks))),float(np.amax(peaks)))
    
    if extremes > histlim:
        print(f"!A-site disp: Some displacement values ({round(extremes,4)}) are out of the histogram range {histlim}, consider increase 'histlim'. \n")

    valx,binEdges=np.histogram(peaks[:,0],bins=n_bins,range=[-histlim,histlim])
    valy,binEdges=np.histogram(peaks[:,1],bins=n_bins,range=[-histlim,histlim])
    valz,binEdges=np.histogram(peaks[:,2],bins=n_bins,range=[-histlim,histlim])
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    
    curvex = savitzky_golay(valx,window_size=11).clip(min=0)
    curvey = savitzky_golay(valy,window_size=11).clip(min=0)
    curvez = savitzky_golay(valz,window_size=11).clip(min=0)
     
    
    fig, ax = plt.subplots()
    ax.fill_between(bincenters, curvex, 0, color="C0", alpha=.5, label="X")
    ax.fill_between(bincenters, curvey, 0, color="C1", alpha=.5, label="Y")
    ax.fill_between(bincenters, curvez, 0, color="C2", alpha=.5, label="Z")
    ax.set_ylim(bottom=0)
    ax.set_xlim([-0.75,0.75])
        
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
    ax.set_xlabel(r'Position ($\AA$)', fontsize = 15) # X label
    ax.set_yticks([])
    plt.legend(prop={'size': 14})
    plt.title("Vibration Center", fontsize=16)
        
    if not title is None:
        ax.text(0.08, 0.90, title, horizontalalignment='center', fontsize=24, verticalalignment='center', transform=ax.transAxes)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    
    
    # compute oscillation
    vib = disp-peaks
    fig, ax = plt.subplots()
    valx,binEdges=np.histogram(vib[:,:,0],bins=100,range=[-1,1])
    valy,binEdges=np.histogram(vib[:,:,1],bins=100,range=[-1,1])
    valz,binEdges=np.histogram(vib[:,:,2],bins=100,range=[-1,1])
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    curvex = savitzky_golay(valx,window_size=21).clip(min=0)
    curvey = savitzky_golay(valy,window_size=21).clip(min=0)
    curvez = savitzky_golay(valz,window_size=21).clip(min=0)
    ax.fill_between(bincenters, curvex, 0, color="C0", alpha=.3, label="X")
    ax.fill_between(bincenters, curvey, 0, color="C1", alpha=.3, label="Y")
    ax.fill_between(bincenters, curvez, 0, color="C2", alpha=.3, label="Z")
    ax.set_ylim(bottom=0)
    ax.set_xlim([-1,1])
        
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
    ax.set_xlabel(r'Displacement ($\AA$)', fontsize = 15) # X label
    ax.set_yticks([])
    plt.legend(prop={'size': 14})
    plt.title("Individual Displacement", fontsize=16)
    
    if not title is None:
        ax.text(0.08, 0.90, title, horizontalalignment='center', fontsize=24, verticalalignment='center', transform=ax.transAxes)
    
    plt.show()
    
    # resolve vibbration with FFT
    Nstep = vib.shape[0]
    #sgws = round(Nstep/20)
    #if sgws%2==0:
    #    sgws+=1
    xf = fftfreq(Nstep, readTimestep)[:Nstep//2]
    yf = np.zeros((3,int(Nstep/2))) # FFT data in 3D
    magf = np.zeros((1,int(Nstep/2))) # FFT data in 3D
    for i in range(vib.shape[1]):
        #va = savitzky_golay(vib[:,i,0],window_size=sgws)
        #vb = savitzky_golay(vib[:,i,1],window_size=sgws)
        #vc = savitzky_golay(vib[:,i,2],window_size=sgws)
        va = vib[:,i,0]
        vb = vib[:,i,1]
        vc = vib[:,i,2]

        mag = np.linalg.norm(vib[:,i,:],axis=1)
        #mag = savitzky_golay(mag,window_size=sgws)
        mag = mag - np.mean(mag)
        mag = mag*np.amax(np.abs(mag))
        
        yf[0,:] = yf[0,:] + 2.0/Nstep * np.abs(fft(va)[0:Nstep//2]).reshape(1,-1)
        yf[1,:] = yf[1,:] + 2.0/Nstep * np.abs(fft(vb)[0:Nstep//2]).reshape(1,-1)
        yf[2,:] = yf[2,:] + 2.0/Nstep * np.abs(fft(vc)[0:Nstep//2]).reshape(1,-1)
        magf = magf + 2.0/Nstep * np.abs(fft(mag)[0:Nstep//2]).reshape(1,-1)
    
    yf = yf/Nstep/vib.shape[1]
    magf = magf/Nstep/vib.shape[1]
    
    fig, ax = plt.subplots()
    mag = np.linalg.norm(vib[:,i,:],axis=1)
    #mag = savitzky_golay(mag,window_size=sgws)
    mag = mag - np.mean(mag)
    ax.plot(readTimestep*np.array(list(range(mag.shape[0]))),mag)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel(r'Radial distance ($\AA$)', fontsize = 15) # Y label
    ax.set_xlabel('Time (ps)', fontsize = 15) # X label
    #ax.set_yticks([])
    #plt.legend(prop={'size': 14})
    #plt.title("Individual Displacement", fontsize=16)
    
    plt.show()
    
# =============================================================================
#     fig, ax = plt.subplots()
#     ax.plot(xf,yf[0,:])
#     ax.plot(xf,yf[1,:])
#     ax.plot(xf,yf[2,:])
#     ax.set_xlim([0,10])
#     plt.show()
#     
#     fig, ax = plt.subplots()
#     ax.plot(xf,magf.reshape(-1,))
# 
#     ax.set_xlim([0,10])
#     plt.show()
# =============================================================================
    
    return peaks


def fit_3D_disp_total(dispt,uniname,moltype,saveFigures,n_bins=100,title=None):
    """ 
    A-site displacement calculation (total displacement for all given sites). 

    Args:
        dispt (numpy.ndarray): Total displacement data.
        uniname (str): User-defined name for printing and figure saving.
        moltype (str): Type of molecule.
        saveFigures (bool): Whether to save the figure.
        n_bins (int): Number of bins for histogram.
        title (str): Title for the plot.

    Returns:
        None 
    """
    
    fig_name=f"traj_A_disp_{moltype}_{uniname}.png"
    histlim = 1.2
    
    #if np.amin(dispt) < -histlim:
    #    raise ValueError(f"Some displacement values ({np.amin(dispt)}) are out of the histogram range, increase 'histlim' \n")
    #if np.amax(dispt) > histlim:
    #    raise ValueError(f"Some displacement values ({np.amax(dispt)}) are out of the histogram range, increase 'histlim' \n")
        
    valx,binEdges=np.histogram(dispt[:,0],bins=n_bins,range=[-histlim,histlim])
    valy,binEdges=np.histogram(dispt[:,1],bins=n_bins,range=[-histlim,histlim])
    valz,binEdges=np.histogram(dispt[:,2],bins=n_bins,range=[-histlim,histlim])
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    
    curvex = savitzky_golay(valx,window_size=17).clip(min=0)
    curvey = savitzky_golay(valy,window_size=17).clip(min=0)
    curvez = savitzky_golay(valz,window_size=17).clip(min=0)
     
    
    fig, ax = plt.subplots()
    ax.fill_between(bincenters, curvex, 0, color="C0", alpha=.5, label="X")
    ax.fill_between(bincenters, curvey, 0, color="C1", alpha=.5, label="Y")
    ax.fill_between(bincenters, curvez, 0, color="C2", alpha=.5, label="Z")
    ax.set_ylim(bottom=0)
    ax.set_xlim([-1.2,1.2])
        
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Counts (a.u.)', fontsize = 15) # Y label
    ax.set_xlabel(r'Displacement ($\AA$)', fontsize = 15) # X label
    ax.set_yticks([])
    plt.legend(prop={'size': 14})
    plt.title("Total Displacement", fontsize=16)
        
    if not title is None:
        ax.text(0.08, 0.90, title, horizontalalignment='center', fontsize=24, verticalalignment='center', transform=ax.transAxes)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()


def peaks_3D_scatter(peaks, uniname, moltype, saveFigures):
    """
    Draw a 3D scatter plot of peaks.

    Args:
        peaks (numpy.ndarray): Peaks data.
        uniname (str): User-defined name for printing and figure saving.
        moltype (str): Type of molecule.
        saveFigures (bool): Whether to save the figure.

    Returns:
        None
    """

    from scipy.spatial import distance_matrix as scipydm
    
    fig_name=f"traj_A_vib_center_3D_{moltype}_{uniname}.png"
    
    box = 0.5
    radial = np.empty((peaks.shape[0],1))
    for i in range(peaks.shape[0]):
        #radial[i] = np.linalg.norm(peaks[i,:])
        radial[i]=np.log10(np.sum(np.reciprocal(np.square(scipydm(peaks[i,:].reshape(1,3),np.delete(peaks,i,axis=0)).T))))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(peaks[:,0], peaks[:,1], peaks[:,2], c=radial, s=20 , cmap='YlOrRd')
    
    limits = np.array([[-box, box],
                       [-box, box],
                       [-box, box]])

    v, e, f = get_cube(limits)
    ax.plot(*v.T, marker='o', color='k', ls='', markersize=10, alpha=0.5)
    for i, j in e:
        ax.plot(*v[[i, j], :].T, color='k', ls='-', lw=2, alpha=0.5)
    
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    tol = 0.0001
    ax.set_xlim([-box-tol,box+tol])
    ax.set_ylim([-box-tol,box+tol])
    ax.set_zlim([-box-tol,box+tol])
    ax.view_init(30, 65)
    set_axes_equal(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax._axis3don = False
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    
    plt.show()


def defect_3D_scatter(peaks, uniname, deftype, saveFigures):
    """
    Draw a 3D scatter plot of defects.

    Args:
        peaks (numpy.ndarray): Peaks data.
        uniname (str): User-defined name for printing and figure saving.
        deftype (str): Type of defect.
        saveFigures (bool): Whether to save the figure.

    Returns:
        None
    """

    from scipy.spatial import distance_matrix as scipydm
    
    fig_name=f"traj_defect_3D_{deftype}_{uniname}.png"
    
    radial = np.empty((peaks.shape[0],1))
    for i in range(peaks.shape[0]):
        #radial[i] = np.linalg.norm(peaks[i,:])
        radial[i]=np.log10(np.sum(np.reciprocal(np.square(scipydm(peaks[i,:].reshape(1,3),np.delete(peaks,i,axis=0)).T))))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(peaks[:,0], peaks[:,1], peaks[:,2], c=radial, s=20 , cmap='YlOrRd')
    
    limits = np.array([[0, 1],
                       [0, 1],
                       [0, 1]])

    v, e, f = get_cube(limits)
    ax.plot(*v.T, marker='o', color='k', ls='', markersize=10, alpha=0.5)
    for i, j in e:
        ax.plot(*v[[i, j], :].T, color='k', ls='-', lw=2, alpha=0.5)
    
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    tol = 0.0001
    ax.set_xlim([0-tol,1+tol])
    ax.set_ylim([0-tol,1+tol])
    ax.set_zlim([0-tol,1+tol])
    ax.view_init(30, 65)
    set_axes_equal(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax._axis3don = False
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    
    plt.show()


def draw_transient_properties(Lobj,Tobj,Cobj,Mobj,uniname,saveFigures):
    """
    Draw transient properties of the system.

    Args:
        Lobj (list): List containing lattice parameters.
        Tobj (list): List containing tilting properties.
        Cobj (list): List containing tilting correlation properties.
        Mobj (list): List containing molecular properties.
        uniname (str): User-defined name for printing and figure saving.
        saveFigures (bool): Whether to save the figure.

    Returns:
        None
    """
    
    lwid = 2
    colors = ["C0","C1","C2","C3","C4","C5","C6"]
    xlimmax = min(max(Lobj[0]),max(Tobj[0]),max(Cobj[0]),max(Mobj[0]))*0.9
    #xlimmax = 40
    
    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False)
    fig.set_size_inches(6.5,5.5)

    axs[0].plot(Lobj[0],Lobj[1],label = r'$\mathit{a}$',linewidth=lwid,color=colors[0])
    axs[0].plot(Lobj[0],Lobj[2],label = r'$\mathit{b}$',linewidth=lwid,color=colors[1])
    axs[0].plot(Lobj[0],Lobj[3],label = r'$\mathit{c}$',linewidth=lwid,color=colors[2])
    #ax.text(0.2, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
    pmax = max(max(Lobj[1]),max(Lobj[2]),max(Lobj[3]))
    pmin = min(min(Lobj[1]),min(Lobj[2]),min(Lobj[3]))
    axs[0].set_ylim([pmin-(pmax-pmin)*0.2,pmax+(pmax-pmin)*0.2])
    axs[0].set_xlim([0,xlimmax])
    axs[0].tick_params(axis='both', which='major', labelsize=12.5)
    axs[0].set_ylabel("pc-lp "+r'($\mathrm{\AA}$)', fontsize = 14) # Y label
    #axs[0].set_xlabel("Time (ps)", fontsize = 12) # X label
    #axs[0].set_xticklabels([])
    
    
    axs[1].plot(Tobj[0],Tobj[1],label = r'$\mathit{a}$',linewidth=lwid,color=colors[0])
    axs[1].plot(Tobj[0],Tobj[2],label = r'$\mathit{b}$',linewidth=lwid,color=colors[1])
    axs[1].plot(Tobj[0],Tobj[3],label = r'$\mathit{c}$',linewidth=lwid,color=colors[2])   
    axs[1].set_xlim([0,xlimmax])
    axs[1].set_ylim([0,17])
    axs[1].set_yticks([0,5,10,15])
    axs[1].tick_params(axis='both', which='major', labelsize=12.5)
    axs[1].set_ylabel(r'Tilt Angle ($\degree$)', fontsize = 14) # Y label
    #axs[1].set_xlabel("Time (ps)", fontsize = 12) # X label
    #axs[1].set_xticklabels([])
    #axs[1].legend(prop={'size': 10},ncol=3)
    
    
    axs[2].plot(Cobj[0],Cobj[1],label = r'$\mathit{a}$',linewidth=lwid,color=colors[0])
    axs[2].plot(Cobj[0],Cobj[2],label = r'$\mathit{b}$',linewidth=lwid,color=colors[1])
    axs[2].plot(Cobj[0],Cobj[3],label = r'$\mathit{c}$',linewidth=lwid,color=colors[2])   
    axs[2].set_xlim([0,xlimmax])
    axs[2].set_ylim([-1.1,1.2])
    axs[2].set_yticks([-1,0,1])
    axs[2].tick_params(axis='both', which='major', labelsize=12.5)
    axs[2].set_ylabel('TCP', fontsize = 14) # Y label
    
    
    #colors = ["C0","C1","C2","C3"]   
    axs[3].plot(Mobj[0], Mobj[1], label = r'$\mathit{a}$' , linewidth=lwid,color=colors[0]) 
    axs[3].plot(Mobj[0], Mobj[2], label = r'$\mathit{b}$' , linewidth=lwid,color=colors[1]) 
    axs[3].plot(Mobj[0], Mobj[3], label = r'$\mathit{c}$' , linewidth=lwid,color=colors[2]) 
    axs[3].plot(Mobj[0], Mobj[4], label = 'CF' ,color ='k', linewidth=lwid, linestyle='dashed') 
    #axs[2].plot(Mobj[0], Mobj[1], label = 'a-NN1' ,color =colors[0], linewidth=lwid) 
    #axs[2].plot(Mobj[0], Mobj[2], label = 'b-NN1' ,color =colors[1], linewidth=lwid) 
    #axs[2].plot(Mobj[0], Mobj[3], label = 'c-NN1' ,color =colors[2], linewidth=lwid) 
    #axs[2].plot(Mobj[0], Mobj[4], label = 'a-NN2' ,color =colors[0], linestyle = 'dashed', linewidth=lwid) 
    #axs[2].plot(Mobj[0], Mobj[5], label = 'b-NN2' ,color =colors[1], linestyle = 'dashed', linewidth=lwid) 
    #axs[2].plot(Mobj[0], Mobj[6], label = 'c-NN2' ,color =colors[2], linestyle = 'dashed', linewidth=lwid) 
    
    axs[3].set_xlim([0,xlimmax])
    axs[3].tick_params(axis='both', which='major', labelsize=12.5)
    axs[3].set_ylim([-1.1,1.1])
    axs[3].set_yticks([-1,0,1])
    axs[3].set_ylabel('MO order', fontsize = 15) # Y label
    axs[3].set_xlabel("Time (ps)", fontsize = 14) # X label
    axs[3].legend(prop={'size': 12.2},ncol=4,loc=0)
    #axs[2].legend(prop={'size': 10},ncol=3)

    fig.subplots_adjust(hspace = 0.1,left=0.12,right=0.90)
    
    for ai in range(4):
        for axis in ['top', 'bottom', 'left', 'right']:
            axs[ai].spines[axis].set_linewidth(1)      

    if saveFigures:
        plt.savefig(f"quench_trimetric_{uniname}.png", dpi=350,bbox_inches='tight')
    plt.show()    


def get_cube(limits=None):
    """
    Get the vertices, edges, and faces of a cuboid defined by its limits.

    Args:
        limits (numpy.ndarray): A 2D array of shape (3, 2) defining the limits of the cuboid.
            Example:
                limits = np.array([[x_min, x_max],
                                   [y_min, y_max],
                                   [z_min, z_max]])

    Returns:
        tuple: A tuple containing:
            - vertices (numpy.ndarray): An array of the coordinates of the cuboid vertices.
            - edges (numpy.ndarray): An array of paired indices for connecting vertices that form an edge.
            - faces (numpy.ndarray): An array of groups of four indices that form a face.
    """

    v = np.array([[0, 0, 0], [0, 0, 1],
                  [0, 1, 0], [0, 1, 1],
                  [1, 0, 0], [1, 0, 1],
                  [1, 1, 0]], dtype=int)

    if limits is not None:
        v = limits[np.arange(3)[np.newaxis, :].repeat(7, axis=0), v]

    e = np.array([[0, 1], [0, 2], [0, 4],
                  [1, 3], [1, 5],
                  [2, 3], [2, 6],
                  [4, 5], [4, 6]], dtype=int)

    f = np.array([[0, 2, 3, 1],
                  [0, 4, 5, 1],
                  [0, 4, 6, 2],
                  [1, 5, 7, 3],
                  [2, 6, 7, 3],
                  [4, 6, 7, 5]], dtype=int)

    return v, e, f


def set_axes_equal(ax):
    """
    Set the aspect ratio of a 3D plot to be equal.

    Args:
        ax (matplotlib.axes._axes.Axes): The axes object to set the aspect ratio for.

    Returns:
        None
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


