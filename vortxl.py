#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:34:02 2022

@author: ftrigaux
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


def vortxl(x,x1,x2,Gamma=1,axis=0):
    '''
    A function to compute the velocity induced on point x by a vortex segment x1-x2 of intensity Gamma 
    Based on Katz & Plotkin, Low speed aerodynamics

    Parameters
    ----------
    x : np.array [3xn]
        A numpy array with the 3 coordinates (x,y,z) of the point.
    x1 : np.array [3xn]
        A numpy array with the 3 coordinates (x,y,z) of the vortex segment starting point.
    x2 : np.array [3xn]
        A numpy array with the 3 coordinates (x,y,z) of the vortex segment ending point..
    Gamma : scalar or np.array [n], optional
        The circulation of the vortex segments. The default is 1.
    axis : scalar, optional
        The no.axis on along which the vectorial operations should be carried. The default is 0.

    Returns
    -------
    v : np.array [3xn]
        the velocity (u,v,w) induced at the point by the vortex segment

    '''
    
    delta = 1e-8; #regularization parameter -- avoids divide by zero errors
    
    r1 = x-x1;
    r2 = x-x2;
    r0 = x2-x1;
    r1cr2 = np.cross(r1,r2,axis=axis);
    
    norm_r1cr2 = np.sqrt(np.sum(r1cr2**2,axis=axis)) + delta;
    norm_r0    = np.sqrt(np.sum(r0**2,axis=axis))    + delta;
    norm_r1    = np.sqrt(np.sum(r1**2,axis=axis))    + delta;
    norm_r2    = np.sqrt(np.sum(r2**2,axis=axis))    + delta;
    
    dist = norm_r1cr2/norm_r0              + delta;
    cos1 = np.sum(r1*r0,axis=axis)/norm_r0/norm_r1;
    cos2 = np.sum(r2*r0,axis=axis)/norm_r0/norm_r2;
    drct = r1cr2 / norm_r1cr2;
    
    v = Gamma/4/np.pi/dist * ( cos1 - cos2 ) * drct;
    
    return v;

# ~~ Example of usage ~~
# This gives 3 examples of usage of the function

if __name__=="__main__":
    # 1/
    #
    # Velocity induced by a point close to a vortex segment.
    # 
    # x1 -------- x2
    #       |
    #       |  d
    #       |
    #       x
    # 
    # The distance d from the point x to the vortex segment x1-x2 varies linearly.
    # The function is compared to the analytical solution
    
    n  = 100;
    
    # List of points at which we compute the induction - varies along the x-axis
    x   = np.zeros((3,n+1));
    x[0,:] = np.linspace(0,1e-3,n+1);
    
    # Starting point of the vortex segment
    x1  = np.zeros((3,n+1));
    x1[1,:] = 1.0;
    
    # Ending point of the vortex segment
    x2  = np.zeros((3,n+1));
    x2[1,:] = -1.0;
    
    # Call the function
    v  = vortxl(x, x1, x2);
    
    # Show the results - the velocity is induced in the z direction
    plt.figure()
    plt.semilogy(x[0,:],v[2,:],'.',label='Katz&Plotkin'); # Katz and Plotkin implementation
    
    # Analytical solution
    plt.semilogy(x[0,1:],1/4/np.pi/x[0,1:]*2/(x[0,1:]**2+1)**(1.0/2.0),'k--',label='Theory');
    
    plt.grid(True);
    plt.xlabel('Distance d [m]');
    plt.ylabel('Induced vel [m/s]');
    plt.legend()
    
    
    # 2/
    # Velocity induced on a point by multiple vortex segments.
    #
    # A line of 10 vortex segments with varying circulation induce velocity on the origin
    
    x  = np.zeros((3,n));
    x1 = np.zeros((3,n));
    x2 = np.zeros((3,n));
    
    x1[0,:] = 1.0
    x1[1,:] = np.linspace(-10,10,n);
    x2[0,:] = 1.0
    x2[1,:] = x1[1,:] + x1[1,1]-x1[1,0];
    
    Gamma = np.linspace(0,1,n);
    
    # Call the function
    v = vortxl(x, x1, x2);
    
    # Show the results - the velocity is induced in the z direction
    plt.figure()
    plt.semilogy(x1[1,:],v[2,:],'-');
    
    plt.grid(True);
    plt.xlabel('Distance d [m]');
    plt.ylabel('Induced vel [m/s]');
    
    v_tot = np.sum(v,axis=1);
    
    print("Velocity induced (u-v-w) [m/s] = (%1.2e, %1.2e, %1.2e)"%(v_tot[0],v_tot[1],v_tot[2]));
    
    # 3/
    # Velocity induced by 6 vortex rings
    # This can help you start with the lifting surface implementation
    #
    # 1---2---3---4 ----> x
    # | x |   |   |
    # 5---6---7---8
    # |   |   |   |
    # 9--10--11--12
    # |
    # |
    # y
    # 

    nx = 16;          # number of rings in the x direction
    ny = 8;          # number of rings in the y direction
    nring = nx*ny;   # total number of rings
    nsegm = nring*4; # total number of segments
    
    xv = np.linspace(0,nx+1,nx+1); # position of the nodes in the x direction
    yv = np.linspace(0,ny+1,ny+1); # position of the nodes in the y direction
    XV,YV = np.meshgrid(xv,yv);
    
    # For each ring, create 4 segments and fill their start and their end
    xs_start = np.zeros((3,nsegm)); # coordinates of the starting point of each seg
    xs_end   = np.zeros((3,nsegm)); # coordinates of the ending point of each seg
    xptr = np.array([0,1,1,0]);     # pattern in the x direction
    yptr = np.array([0,0,1,1]);     # pattern in the y direction
    for iy in range(ny):
        for ix in range(nx):
            idx = (iy*nx+ix)*4;
            xs_start[:,idx:idx+4] = np.array([XV[iy+yptr,ix+xptr],YV[iy+yptr,ix+xptr],np.zeros(4)]); # fill the coords of the starting points
            xs_end[:,idx:idx+3]   = np.copy(xs_start[:,idx+1:idx+4]); # fill the coords of the ending points
            xs_end[:,idx+3]       = np.copy(xs_start[:,idx]);
    
    # Compute the velocity at each center point vel is computed
    v_tot = np.zeros((3,nring));
    for iy in range(ny):
        for ix in range(nx):
            x = np.array([(xv[ix]+xv[ix+1])/2.0,(yv[iy]+yv[iy+1])/2.0,0.0]);
            x = np.outer(x,np.ones(nsegm)); # create a (3xN) array (N = number of segments)
    
            # computed the induced vel of Gamma=1 for all rings
            v = vortxl(x,xs_start,xs_end);
            v_tot[:,iy*nx+ix] = np.sum(v,axis=1);
    
            #print("Velocity induced by 4 rings in square %d (u-v-w) [m/s] = (%1.2e, %1.2e, %1.2e)"%(iy*nx+ix,v_tot[0],v_tot[1],v_tot[2]));
    plt.figure()
    plt.pcolormesh(v_tot[2,:].reshape((ny,nx)),shading='flat',cmap='Blues');
    plt.colorbar();
    plt.axis('scaled');
    