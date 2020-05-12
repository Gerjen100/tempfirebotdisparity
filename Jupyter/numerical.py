#!/usr/bin/env python
# coding: utf-8

# # Stereo setup
# 
# 
# <img src="img/stereo_setup.svg" width="80%"/>

# The setup we discuss in this notebook is shown above. Without loss of generality we assume that the cameras have identical parameters focal length $f$, angular horizontal field of view (afovh), and horizontal en vertical resolution $R_h$ and $R_v$. The relative pose between the cameras is assumed to only have a positive translation in $x$ direction, and the coordinate system we use as reference (also known as the world frame) is equal to the camera frame of the left camera (camera 1).

# In[1]:


import numpy as np


# # Numerical approach
# 
# First set the stereo setup's defining parameters. The angular field of view (`afovh`, $\alpha_h$) is set to 50°, the focal length (`f`, $f$) is set to 4.3mm (0.0043m). The horizontal resolution (`R_h`, $R_h$) of both cameras is set to 320 pixels. The distance between the camera's (`b`, $b$) can be varied but should be expressed in meters. The vertical resolution ('R_v', $R_v$) is also declared, this way we have the complete camera defined even if we only want to 'feed' it 2D coordinates.

# In[2]:


afovh = 50.  # in degrees
f = 0.0043  # in meters
R_h = 320  # in number of pixels
R_v = 240  # in number of pixels
b = 6.  # in meters


# In an ideal camera it can be assumed that the principal point of the image sensor is exactly in its center, so we can set the row ($v_0$) and column ($u_0$) displacement of the projection as follows:

# In[3]:


u_0 = R_h // 2  # Double slash to get an integer back instead of float
v_0 = R_v // 2


# Next the image sensor width can be determined with the angular field of view and the focal length, knowing that the focal length and the image sensor are perpendicular. Half of the image sensor forms a right-angled triangle with the focal length and with half of the angular field of view as angle, therefore the following holds:
# $$ I_w = 2\tan\left(\frac{\alpha_h}{2}\right)f$$
# We also define the image height to get the complete picture (no pun intended).

# In[4]:


I_w = 2 * np.tan(np.radians(afovh/2)) * f
I_h = I_w / R_h * R_v


# Next the pixel width (`rho_w`, $\rho_w$) can be determined by dividing the image sensor width by the horizontal resolution. The same can be done for the pixel heigth, but by definition that will be the same in our case.

# In[5]:


rho_w = I_w/R_h
rho_h = I_h/R_v


# Since we have two cameras which are only translated with respect to each other, we take the the camera frame of the first (left) camera as a world frame. Then we need to define the pose of the second camera as expressed in the world frame. This can be expressed as a 4x4 homogeneous transformation matrix.

# In[6]:


T = np.array([[1, 0, 0, b],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])


# Actually to transform coordinates from the world frame (which is the first camera frame) to the second camera frame, in which the projection works, we need the inverse of this relative pose. For homogeneous transformation matrices, the inverse can be determined quite simply, if
# $$ \mathbf{T} = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{pmatrix} $$
# where $\mathbf{R}$ is the 3x3 rotation matrix and $\mathbf{t}$ is the 3x1 translation vector, then the inverse of $\mathbf{T}$ is:
# $$ \mathbf{T}^{-1} = \begin{pmatrix} \mathbf{R}^{\mathrm{T}} & -\mathbf{R}^{\mathrm{T}}\mathbf{t} \\ \mathbf{0} & 1 \end{pmatrix} $$
# For convenience, we define a function to do this:

# In[7]:


def inverse_transformation(T):
    R, t = T[:-2, :-2], T[:-2, -1]
    T_inv = np.array(T)
    T_inv[:-2, :-2] = R.T
    T_inv[:-2, -1] = np.dot(-R.T, t)
    return T_inv

T_inv = inverse_transformation(T)
T_inv


# Now with all camera parameters set, we can define the projection of 3D points defined in the world frame onto the image plane of the cameras and expressed in pixel points w.r.t. the top-left pixel which is defined as $(0,0)$.
# 
# The projection consists of three parts:
# 
# 1. 3D points will first be transformed to the camera frame of the used camera. This is done with the inverse of the pose as derived before.
# 2. After that, the projection matrix will transform the 3D points from 3D space to the 2D image plane coordinates (also known as normalized image coordinates). This is done with:
# $$ \tilde{\mathbf{p}} = \begin{pmatrix} f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix} \tilde{\mathbf{P}} $$ 
# where $\mathbf{p}$ is our 2D point in normalized image coordinates, $\mathbf{P}$ is our 3D point, and the tilde ($\tilde{}$) notation indicates that we expres the point in its homogeneous form.
# 3. Finally another transformation will scale the normalized image coordinates to the pixel coordinate system and translate the origin to the top left pixel which was our $(0,0)$. This is done by the following transformation (also known as a homography):
# $$ \tilde{\mathbf{p}}_{\mathrm{pixel}} = \begin{pmatrix} \frac{1}{\rho_w} & 0 & u_0 \\ 0 & \frac{1}{\rho_h} & v_0 \\ 0 & 0 & 1 \end{pmatrix} $$
# 
# This results in the following camera matrix:
# $$ \mathbf{C} = \begin{pmatrix} \frac{f}{\rho_w} & 0 & u_0 & 0 \\ 0 & \frac{f}{\rho_h} & v_0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix} \mathbf{T}^{-1} $$
# 
# See below for a definition of frames. (source: *Robotics, Vision and Control - Peter Corke*, DOI [10.1007/978-3-642-20144-8](https://link.springer.com/book/10.1007%2F978-3-642-20144-8), figure 11.5, p. 256).
# 
# <img src="img/pixel_cs.png" width="60%"/>
# 
# For the first camera, where $\mathbf{T}$ is the identity matrix, that is:

# In[8]:


T_pix = np.array([[1/rho_w, 0, u_0],
                  [0, 1/rho_h, v_0],
                  [0, 0, 1]])  # Technically not a projection because it also scales

T_proj = np.array([[f, 0, 0, 0],
                   [0, f, 0, 0],
                   [0, 0, 1, 0]])  # Technically not a transformation either but a projection

C1 = np.dot(T_pix, T_proj)
C1


# For the second camera, the first part is the same, but we have to multiply with the inverse of its pose:

# In[9]:


C2 = np.dot(C1, T_inv)
C2


# Next the point of interest in the 3D space is defined (I like to use a simple class for this to make some things easier in calculating). The coordinates of the point can be chosen freely, but for now only the $u$-coordinate of the projection is taken into account, which means that only the $x$- and $z$-coordinate of the 3D point are important.

# In[10]:


class Position(object):

    # To make numpy aware of array multiplication
    __array_priority__ = 100
    
    def __init__ (self, p):
        assert len(p) in [1, 2, 3], 'p should be a tuple with length 1, 2, or 3'
        if len(p) == 3:
            self._p = np.array([[p[0]], [p[1]], [p[2]]]).reshape((3, 1))
        elif len(p) == 2:
            self._p = np.array([[p[0]], [p[1]]]).reshape((2, 1))
        elif len(p) == 1:
            self._p = np.array([[p[0]]]).reshape((1, 1))

    @property
    def x(self):
        return self._p[0].item()
    
    @x.setter
    def x(self, val):
        self._p[0] = val
        
    @property
    def y(self):
        return self._p[1].item()
    
    @y.setter
    def y(self, val):
        self._p[1] = val
    
    @property
    def z(self):
        return self._p[2].item()
    
    @z.setter
    def z(self, val):
        self._p[2] = val
        
    def as_homogeneous(self):
        return np.append(self._p, [[1]], axis=0)
        
    def __rmul__(self, other):
        # Get homogeneous multiplication
        res = np.dot(other, self.as_homogeneous())
        # Return only the first coordinates, divided by last to switch back to non-homogeneous
        return res[:-1] / res[-1]
    
    def __repr__(self):
        return self._p.__repr__()
        
P = Position([3, 0, 10])


# Now to check which pixel will light up for the chosen point, we only have to multiply with the camera matrix of both cameras. We are only interested in the column, the $u$ coordinate, so we take the first entry of the result (use `.item()` to get the numeric value instead of a 1x1 matrix containing the value).

# In[11]:


u1 = (C1*P)[0].item()
u2 = (C2*P)[0].item()


# Now to get the limits of the pixel, we need to round the sub-pixel value up and down. We define the upper and lower limit as `uill` and `uiul` respectively, where `i` indicates the camera.

# In[12]:


u1ll, u1ul = np.floor(u1), np.ceil(u1)
u2ll, u2ul = np.floor(u2), np.ceil(u2)


# From the pixel values we need to back to the spatial domain. We can do this with the inverse of the `T_pix` matrix we defined earlier. This is not technically a Euclidian transformation because it also involves a scaling (and luckily no rotation). Therefore the previously defined function `inverse_transformation()` is not applicable and we have to determine the inverse manually.
# 
# We know that:
# $$ T_{\mathrm{pix}} = \begin{pmatrix} \frac{1}{\rho_w} & 0 & u_0 \\ 0 & \frac{1}{\rho_h} & v_0 \\ 0 & 0 & 1 \end{pmatrix} $$
# So intuitively its inverse would be:
# $$ T_{\mathrm{pix}}^{-1} = \begin{pmatrix} \rho_w & 0 & -\rho_wu_0 \\ 0 & \rho_h & -\rho_hv_0 \\ 0 & 0 & 1 \end{pmatrix} $$

# In[13]:


T_pix_inv = np.array([[rho_w, 0, -rho_w*u_0],
                      [0, rho_h, -rho_h*v_0],
                      [0, 0, 1]])
T_pix_inv


# To check if it really is the inverse, let's see if the product equals identity:

# In[14]:


np.array_equal(np.dot(T_pix_inv, T_pix), np.eye(3))


# Now to determine the upper and lower limit in terms of the image plane (so in meters), we multiply the limits (in homogeneous form where we set the $v$-coordinate to equal $v_0$ because we are simply not interested in that now) with the inverse of `T_pix`.

# In[15]:


p1ll = T_pix_inv * Position([u1ll, v_0])
p1ul = T_pix_inv * Position([u1ul, v_0])
p2ll = T_pix_inv * Position([u2ll, v_0])
p2ul = T_pix_inv * Position([u2ul, v_0])


# What we have now, is four 2D coordinates. If we refer to the geometry of the image plane again:
# 
# (source: *Robotics, Vision and Control - Peter Corke*, DOI [10.1007/978-3-642-20144-8](https://link.springer.com/book/10.1007%2F978-3-642-20144-8), figure 11.5, p. 256).
# 
# <img src="img/pixel_cs.png" width="60%"/>
# 
# We can see that the 2D point can be seen in 3D as a vector pointing from the camera frame origin, to the 2D projection. This means that if we plug in the focal length $f$ as $z$-coordinate, we get the direction vector of the ray that goes through the actual 3D point $\mathbf{P}$ that we detected with our cameras.
# 
# In the 2D stereo setup we use the upper and lower limits of the pixels translate to the purple bundles:
# 
# <img src="img/stereo_setup.svg" width="80%"/>
# 
# Now we only have to find the point where the top two rays (`p1ll` and `p2ul`) cross and where the bottom two rays (`p1ul` and `p2ll`) cross. Because we are working the $xz$-plane, we can throw away the $y$ coordinate for now.

# In[16]:


p1ll = p1ll[0].item()
p1ul = p1ul[0].item()
p2ll = p2ll[0].item()
p2ul = p2ul[0].item()


# To clarify the following text we number the rays from left to right at $z=f$, so $l_1$ crosses the origin and point $(p_{1ll}, f)$, and $l_2$ crosses the origin and the point $(p_{1ul}, f)$, et cetera.

# We know that all lines cross the the image plane at $z=f$. For $l_1$ and $l_2$ we also know that they cross the origin. These lines can thus be expressed in $z=\alpha x+\beta$ (standard line equation but with $z$ instead of $y$) where $\beta$ is 0 because they cross the origin:
# 
# $$ l_1 := \frac{f}{p_{1ll}}x $$
# $$ l_2 := \frac{f}{p_{1ul}}x $$
# 
# For $l_3$ and $l_4$ we have to take the displacement of camera 2 into account so the two points that define $l_3$ are $(p_{2ll} + b, f)$ and $(b, 0)$. Substituting these into the standard line equation gives:
# 
# $$ l_3 := \frac{f}{p_{2ll}} x + \beta $$
# 
# Solve for $\beta$ by inserting one of the points:
# 
# $$ 0 = \frac{f}{p_{2ll}} b + \beta $$
# $$ \beta = -\frac{bf}{p_{2ll}} $$
# 
# So:
# 
# $$ l_3 := \frac{f}{p_{2ll}} x -\frac{bf}{p_{2ll}} $$
# 
# For $l_4$ these points are $(p_{2ul} + b, f)$ and $(b, 0)$. This gives:
# 
# $$ l_4 := \frac{f}{p_{2ul}} x - \frac{bf}{p_{2ul}} $$
# 
# Now we would like to get the intersection of $l_1$ and $l_4$, and the intersection of $l_2$ and $l_3$. Let's start with the first (note that we have chosen $\mathbf{P}$ to lie in the middle of the two cameras, intuition tells us already that the intersections we're looking for have the same $x$-coordinate, but to make this script also applicable for other points we solve for $x$ anyway):
# 
# $$ \frac{f}{p_{1ll}}x = \frac{f}{p_{2ul}}x - \frac{bf}{p_{2ul}} $$
# 
# $$ x\left(\frac{f}{p_{1ll}} - \frac{f}{p_{2ul}}\right) = -\frac{bf}{p_{2ul}} $$
# 
# $$ x = -\frac{bf}{p_{2ul}\left( \frac{f}{p_{1ll}} - \frac{f}{p_{2ul}}\right)} $$

# In[17]:


x_upper = - b * f / ( p2ul * (f/p1ll - f/p2ul ) )
x_lower = - b * f / ( p2ll * (f/p1ul - f/p2ll ) )


# Now simply plugin the values for $x$ we found into either of the line equations:

# In[18]:


z_upper = f / p1ll * x_upper
z_upper


# In[19]:


z_lower = f / p1ul * x_lower
z_lower


# The depth resolution can now be defined as the difference between the $z$ coordinates:

# In[20]:


R_z = z_upper - z_lower
print("That's a lot of code just to find depth resolution of {}m".format(R_z))

