import os
import math
import numpy as np
from math import factorial
from scipy.stats import norm
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt

if os.path.exists("style.mplstyle"):
    plt.style.use("style.mplstyle")

def savitzky_golay(y, window_size=51, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the Time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
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



def draw_lattice_evolution(dm, steps, Tgrad, uniname, saveFigures = False, smoother = False, xaxis_type = 'N', Ti = None, x_lims = None, y_lims = None, invert_x = False, num_crop = 0):
    
    fig_name = f"lattice_evo_{uniname}.png"
    
    assert dm.shape[0] == len(steps)
    if dm.ndim == 3:
        La, Lb, Lc = np.nanmean(dm[:,:,0],axis=1), np.nanmean(dm[:,:,1],axis=1), np.nanmean(dm[:,:,2],axis=1)
        
    elif dm.ndim == 2:
        La, Lb, Lc = dm[:,0], dm[:,1], dm[:,2]
    
    if smoother:
        Nwindows = round(dm.shape[0]*0.04)
        if Nwindows%2 == 0:
            Nwindows+=1
        La = savitzky_golay(La,window_size=Nwindows)
        Lb = savitzky_golay(Lb,window_size=Nwindows)
        Lc = savitzky_golay(Lc,window_size=Nwindows)
    
    if num_crop != 0:
        La = La[num_crop:-num_crop]
        Lb = Lb[num_crop:-num_crop]
        Lc = Lc[num_crop:-num_crop]
        steps = steps[num_crop:-num_crop]
    
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
            ax.text(0.2, 0.95, f'Cooling ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
        elif steps[0] < steps[-1]:
            ax.text(0.2, 0.95, f'Heating ({round(Tgrad,1)} K/ps)', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
        else:
            ax.text(0.2, 0.95, f'Heat bath at {Ti}K', horizontalalignment='center', fontsize=14, verticalalignment='center', transform=ax.transAxes)
    
    
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
    plt.ylabel('Lattice Parameter ($\mathrm{AA}$)', fontsize=14)
    plt.legend(prop={'size': 12})
    

    if not x_lims is None:
        ax.set_xlim(x_lims)
    
    if not y_lims is None:
        ax.set_ylim(y_lims)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return (steps,La,Lb,Lc)


def draw_tilt_evolution_time(T, steps, uniname, saveFigures, smoother = 0, y_lim = None):
    
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


def compute_tilt_density(T,method = "curve"):
    
    init_arr = np.array([-0.1,0.1])
    T_a = T[:,:,0].reshape((-1,))
    T_b = T[:,:,1].reshape((-1,))
    T_c = T[:,:,2].reshape((-1,))
    tup_T = (T_a,T_b,T_c)
    
    if method == "curve":
        n_bins = 200
        
        Y = []
        for i in range(3):
            y,binEdges=np.histogram(np.abs(tup_T[i]),bins=n_bins,range=[0,45])
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                Y.append(bincenters)
            Y.append(y)
        Y = np.transpose(np.array(Y))
        
        maxs = []
        window_size = n_bins/4
        if window_size%2==0:
            window_size+=1
        for i in range(3):
            temp = savitzky_golay(Y[:,i+1], window_size=window_size, order=3, deriv=0, rate=1)
            maxs.append(abs(Y[:,0][np.argmax(temp)]))
    
    elif method == "kmean":
        from scipy.cluster.vq import kmeans
        maxs = []
        for i in range(3):
            centers = kmeans(tup_T[i], k_or_guess=init_arr, iter=20, thresh=1e-05)[0]
            if abs(np.abs(centers[1])-np.abs(centers[0])) > 0.3:
                print("!Tilting-Kmeans: Fit error above threshold, turned to curve fitting, see difference below. ")
            maxs.append((np.abs(centers[1])+np.abs(centers[0]))/2)

    return maxs


def draw_distortion_evolution_sca(D, steps, uniname, saveFigures, xaxis_type = 'N', scasize = 2.5, y_lim = 0.4):
    
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

    if gaus_fit:
        return Mu, Std


def draw_dist_density_frame(D, uniname, saveFigures, n_bins = 100, xrange = [0,0.5]):
    
    fig_name = f"frame_dist_{uniname}.png"
    
    assert D.ndim == 2
    
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
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
        
    plt.show()
    return Mu


def draw_tilt_density(T, uniname, saveFigures, n_bins = 100, symm_n_fold = 4, title = None):
    
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


def draw_octatype_tilt_density(Ttype, config_types, uniname, saveFigures, n_bins = 100, symm_n_fold = 4):
    """ 
    Isolate tilting pattern wrt. the local halide configuration.  
    """
    
    from pdyna.structural import periodicity_fold
    fig_name=f"tilt_octatype_density_{uniname}.png"
    
    if symm_n_fold == 2:
        hrange = [-90,90]
        tlabel = [-90,-60,-30,0,30,60,90]
    elif symm_n_fold == 4:
        hrange = [-45,45]
        tlabel = [-45,-30,-15,0,15,30,45]
    elif symm_n_fold == 8:
        hrange = [0,45]
        tlabel = [0,15,30,45]
    
    typesname = ["I6 Br0","I5 Br1","I4 Br2: right-angle","I4 Br2: linear","I3 Br3: right-angle",
                 "I3 Br3: planar","I2 Br4: right-angle","I2 Br4: linear","I1 Br5","I0 Br6"]
    typexval = [0,1,1.8,2.2,2.8,3.2,3.8,4.2,5,6]
    typextick = ['0','1','2r','2l','3r','3p','4r','4l','5','6']
    
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
        
        m1 = np.array(compute_tilt_density(T)).reshape(1,-1)
        maxs = np.concatenate((maxs,m1),axis=0)
    
    # plot type dependence   
    plotx = np.array([typexval[i] for i in config_types])
    plotxlab = [typextick[i] for i in config_types]
    
    scaalpha = 0.9
    scasize = 50
    plt.subplots(1,1)
    ax = plt.gca()
    ax.scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    ax.scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    ax.scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    plt.legend(prop={'size': 12})
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Tilting (deg)', fontsize = 15) # Y label
    ax.set_xlabel('Br content', fontsize = 15) # X label
    ax.set_xticks(plotx)
    ax.set_xticklabels(plotxlab)
    ax.set_ylim(bottom=0)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return maxs


def draw_octatype_dist_density(Dtype, config_types, uniname, saveFigures, n_bins = 100, xrange = [0,0.5]):
    """ 
    Isolate distortion mode wrt. the local halide configuration.  
    """
    
    fig_name=f"dist_octatype_density_{uniname}.png"
    
    typesname = ["I6 Br0","I5 Br1","I4 Br2: right-angle","I4 Br2: linear","I3 Br3: right-angle",
                 "I3 Br3: planar","I2 Br4: right-angle","I2 Br4: linear","I1 Br5","I0 Br6"]
    typexval = [0,1,1.8,2.2,2.8,3.2,3.8,4.2,5,6]
    typextick = ['0','1','2r','2l','3r','3p','4r','4l','5','6']
    
    config_types = list(config_types)
    config_involved = []
    
    Dgauss = np.empty((0,4))
    Dgaussstd = np.empty((0,4))
    
    for di, D in enumerate(Dtype):
        if D.ndim == 3:
            D = D.reshape(D.shape[0]*D.shape[1],4)
        
        figs, axs = plt.subplots(4, 1)
        labels = ["Eg","T2g","T1u","T2u"]
        colors = ["C3","C4","C5","C6"]
        for i in range(4):
            y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
            axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
        
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
            ax.set_ylabel('counts (a.u.)', fontsize = 15) # Y label
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

        axs[0].set_title(typesname[config_types[di]],fontsize=16)
        config_involved.append(typesname[config_types[di]])    
            
        plt.show()
        
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
    
    return Dgauss, Dgaussstd


def draw_halideconc_tilt_density(Tconc, concent, uniname, saveFigures, n_bins = 100, symm_n_fold = 4):
    """ 
    Isolate tilting pattern wrt. the local halide concentration.  
    """
    
    from pdyna.structural import periodicity_fold
    fig_name=f"tilt_halideconc_density_{uniname}.png"
    
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
        
        m1 = np.array(compute_tilt_density(T)).reshape(1,-1)
        maxs = np.concatenate((maxs,m1),axis=0)
    
    # plot type dependence   
    plotx = concent
    
    scaalpha = 0.9
    scasize = 50
    plt.subplots(1,1)
    ax = plt.gca()
    ax.scatter(plotx,maxs[:,0],label=r'$\mathit{a}$',alpha=scaalpha,s=scasize)
    ax.scatter(plotx,maxs[:,1],label=r'$\mathit{b}$',alpha=scaalpha,s=scasize)
    ax.scatter(plotx,maxs[:,2],label=r'$\mathit{c}$',alpha=scaalpha,s=scasize)
    plt.legend(prop={'size': 12})
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Tilting (deg)', fontsize = 15) # Y label
    ax.set_xlabel('Br content', fontsize = 15) # X label
    ax.set_ylim(bottom=0)
    
    if saveFigures:
        plt.savefig(fig_name, dpi=350,bbox_inches='tight')
    plt.show()
    
    return maxs


def draw_halideconc_dist_density(Dconc, concent, uniname, saveFigures, n_bins = 100, xrange = [0,0.5]):
    """ 
    Isolate distortion mode wrt. the local halide concentration.  
    """
    
    fig_name=f"dist_halideconc_density_{uniname}.png"
    
    Dgauss = np.empty((0,4))
    Dgaussstd = np.empty((0,4))
    
    for di, D in enumerate(Dconc):
        if D.ndim == 3:
            D = D.reshape(D.shape[0]*D.shape[1],4)
        
        figs, axs = plt.subplots(4, 1)
        labels = ["Eg","T2g","T1u","T2u"]
        colors = ["C3","C4","C5","C6"]
        for i in range(4):
            y,binEdges=np.histogram(D[:,i],bins=n_bins,range=xrange)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axs[i].plot(bincenters,y,label = labels[i],color = colors[i],linewidth = 2.4)
            axs[i].text(0.05, 0.82, labels[i], horizontalalignment='center', fontsize=14, verticalalignment='center', transform=axs[i].transAxes)
        
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
            ax.set_ylabel('counts (a.u.)', fontsize = 15) # Y label
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

        axs[0].set_title('Br concentration: '+str(round(concent[di],4)),fontsize=16)  
            
        plt.show()
        
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
    
    return Dgauss, Dgaussstd


def abs_sqrt(m):
    return np.sqrt(np.abs(m))*np.sign(m)


def draw_tilt_corr_evolution_sca(T, steps, uniname, saveFigures, xaxis_type = 't', scasize = 1.5, y_lim = [-1,1]):
    
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


def draw_tilt_and_corr_density_shade_frame(T, Corr, uniname, saveFigures, n_bins = 100):
    """ 
    Generate the Glazer plot. 
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


def draw_tilt_coaxial(T, uniname, saveFigures, n_bins = 71, title = None):
    
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

    ax.set_ylim([-1,1])
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
    #Tmean=0
    
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


def quantify_tilt_domain(sc,scnorm):
    """ 
    Compute spatial coorelation of tilting.  
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
            opt, pcov = curve_fit(model_func, np.array(list(range(sc.shape[1]))), tc, p0)
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
        print(f"!Tilt Spatial Corr: fitting of decay length may be wrong, a value(s): {pop_warning}")
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),sc[0,:,0],linewidth=1.5,label='axis 0')
    plt.plot(list(range(nns)),sc[1,:,0],linewidth=1.5,label='axis 1')
    plt.plot(list(range(nns)),sc[2,:,0],linewidth=1.5,label='axis 2')
    plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    plt.title("Along axis 0",fontsize=15)
    ax.set_ylim([-1,1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),sc[0,:,1],linewidth=1.5,label='axis 0')
    plt.plot(list(range(nns)),sc[1,:,1],linewidth=1.5,label='axis 1')
    plt.plot(list(range(nns)),sc[2,:,1],linewidth=1.5,label='axis 2')
    plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    plt.title("Along axis 1",fontsize=15)
    ax.set_ylim([-1,1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    fig,ax = plt.subplots()
    plt.plot(list(range(nns)),sc[0,:,2],linewidth=1.5,label='axis 0')
    plt.plot(list(range(nns)),sc[1,:,2],linewidth=1.5,label='axis 1')
    plt.plot(list(range(nns)),sc[2,:,2],linewidth=1.5,label='axis 2')
    plt.axhline(y=0,linestyle='dashed',linewidth=1,color='k')
    plt.title("Along axis 2",fontsize=15)
    ax.set_ylim([-1,1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Distance (unit cell)', fontsize=14)
    plt.ylabel('Spatial correlation (a.u.)', fontsize=14)
    legend = plt.legend(prop={'size': 12},frameon = True, loc="upper right")
    legend.get_frame().set_alpha(0.7)
    
    
    
    return scdecay


def vis3D_domain_anime(cfeat,frs,tstep,ss,bin_indices,figname):
    """ 
    Visualise tilting in 3D animation.  
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


def compute_tilt_domain(Corr, timestep, uniname, saveFigures, n_bins=42, tol=0, smoother=5):
    """ 
    Compute tilt domain lifetime.  
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
#    cn=sorted(abs(cn),reverse=True) #This forces symmetry exploitation. Used for figuring out what [x,y,z] corresponds to which point in the figure
    l=np.linalg.norm(cn)
    x=cn[0]
    y=cn[1]
    z=cn[2]
    theta = math.acos(z/l)
    phi   = math.atan2(y,x)
    theta   = theta - math.pi/2 #to agree with Matplotlib view of angles...
    return (theta,phi)


### ANALYSIS code below here; from merged files

def MO_correlation(cnsn,MDTimestep,SaveFigures,uniname):

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


def orientation_density_3D_sphere(cnsn,moltype,SaveFigures,uniname,title=None):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib import cm
    
    def sphere_mesh(res = 80):
        u = np.linspace(0, 2*np.pi, 2*res)
        v = np.linspace(0, np.pi, res )
        # create the sphere surface
        xmesh = np.outer(np.cos(u), np.sin(v))
        ymesh = np.outer(np.sin(u), np.sin(v))
        zmesh = np.outer(np.ones(np.size(u)), np.cos(v))
        points = np.concatenate((xmesh[:,:,np.newaxis],ymesh[:,:,np.newaxis],zmesh[:,:,np.newaxis]),axis=2)
        return xmesh, ymesh, zmesh, points

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
        maxx = np.where(dots > 0.98)
        counting[list(maxx[0]),list(maxx[1])] += 1
    myheatmap = counting / np.amax(counting)
    
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
    #anim.save("link-anim.gif", writer=writer)
    plt.show()
    
    # also save a snapshot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(True)
    ax.plot_surface(xmesh, ymesh, zmesh, alpha=0.8, cstride=1, rstride=1, facecolors=cm.plasma(myheatmap))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([-1.05,1.05])
    ax.set_ylim([-1.05,1.05])
    ax.set_zlim([-1.05,1.05])
    ax.view_init(elev=20, azim=0)
    fig.savefig(f"MO_{moltype}_3D_sphere_{uniname}.png",bbox_inches='tight', pad_inches=0,dpi=350)

    
def orientation_density(cnsn,moltype,SaveFigures,uniname,title=None,miller_mask=False):
    
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


def fit_exp_decay(x,y,allow_redo=True):
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


def fit_exp_decay_fixed(x,y,aconst = 0.9):
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
    from scipy.optimize import curve_fit
    def model_func1(x, omega, gamma):
        tau = 2/gamma
        omega_e = np.sqrt(omega**2-gamma**2/4)
        return np.exp(-x/tau) * (np.cos(omega_e*x)+gamma/(2*omega_e)*np.sin(omega_e*x))
    
    def model_func2(x, tau_l, tau_s):
        return 1/(tau_l-tau_s)*(tau_l*np.exp(-x/tau_l)-tau_s*np.exp(-x/tau_s))
    
    
    x = np.squeeze(x)
    y = np.squeeze(y)/np.amax(y)
    
    fitrange = 1
    xc = x[:round(x.shape[0]*fitrange)]
    yc = y[:round(y.shape[0]*fitrange)]
    
    p0 = (0.2,0.01) # starting search coeffs
    opt, pcov = curve_fit(model_func1, xc, yc, p0)
    omega, gamma= opt
    
    p1 = (50,0.5) # starting search coeffs
    opt, pcov = curve_fit(model_func2, xc, yc, p0)
    tau_l, tau_s = opt

    y1 = model_func1(x, omega, gamma)
    y2 = model_func2(x, tau_l, tau_s)
    fig, ax = plt.subplots()
    #ax.plot(x, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
    ax.plot(x, y2, color='r',label='fitted',linewidth=4)
    ax.plot(x, y, 'bo', label='raw data',alpha=0.13,markersize=4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('Autocorrelation (a.u.)', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.title('Damped Oscillator Fitting', fontsize=14)
    plt.show()
        
    return tau


def draw_MO_spatial_corr_time(C, steps, uniname, saveFigures, smoother = 0, n_bins = 50):
    
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
    
    if rdftype == "CN":
        fig_name = f"RDF_CN_{uniname}.png"
        title = 'C-N RDF'
    elif rdftype == "BX":
        fig_name = f"RDF_BX_{uniname}.png"
        title = 'B-X RDF'
    
    fig, ax = plt.subplots()
    counts,binedge,_ = ax.hist(da,bins=n_bins)
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
    A-site displacement calculation.  
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
    A-site displacement calculation (total displacement).  
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
    """get the vertices, edges, and faces of a cuboid defined by its limits

    limits = np.array([[x_min, x_max],
                   [y_min, y_max],
                   [z_min, z_max]])
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


