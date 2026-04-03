"""
    Author: Turhan Can KARGIN
    Python Version: 3.9.7

This forward_velocity_kinematics.py file contains all necessary kinematics functions for the project
"""

import numpy as np

KAPPA_EPS = 1e-6


def _sin_delta_over_kappa(base_angle, kappa, arc_length):
    """Compute (sin(base + kappa*L) - sin(base)) / kappa with a stable zero-curvature limit."""
    if abs(kappa) > KAPPA_EPS:
        return (np.sin(base_angle + (kappa * arc_length)) - np.sin(base_angle)) / kappa
    return np.cos(base_angle) * arc_length


def _cos_delta_over_kappa(base_angle, kappa, arc_length):
    """Compute (cos(base + kappa*L) - cos(base)) / kappa with a stable zero-curvature limit."""
    if abs(kappa) > KAPPA_EPS:
        return (np.cos(base_angle + (kappa * arc_length)) - np.cos(base_angle)) / kappa
    return -np.sin(base_angle) * arc_length


# %% Function three_section_planar_robot
def three_section_planar_robot(kappa1, kappa2, kappa3, l):  # noqa: E741  # TODO -> Add if else to figure out when any kappa = 0
    """
    * Homogeneous transformation matrix 
    * Mapping from configuration parameters to task space for the tip of the continuum robot
    
    Parameters
    ----------
    kappa1 : float
        curvature value for section 1.
    kappa2 : float
        curvature value for section 2.
    kappa3 : float
        curvature value for section 3.
    l : list
        trunk length contains all sections

    Returns
    -------
    T: numpy array
        Transformation matrices containing orientation and position

    """

    theta1 = kappa1 * l[0]
    theta2 = theta1 + (kappa2 * l[1])
    theta3 = theta2 + (kappa3 * l[2])

    c_ks = np.cos(theta3)
    s_ks = np.sin(theta3)

    A_14 = (
        _cos_delta_over_kappa(0.0, kappa1, l[0])
        + _cos_delta_over_kappa(theta1, kappa2, l[1])
        + _cos_delta_over_kappa(theta2, kappa3, l[2])
    )
    A_24 = (
        _sin_delta_over_kappa(0.0, kappa1, l[0])
        + _sin_delta_over_kappa(theta1, kappa2, l[1])
        + _sin_delta_over_kappa(theta2, kappa3, l[2])
    )

    T = np.array([c_ks, s_ks, 0, 0, -s_ks, c_ks, 0, 0, 0, 0, 1, 0, A_14, A_24, 0, 1])
    T = np.reshape(T, (4, 4), order="F")
    return T

# %% Function three_section_planar_robot
def jacobian_matrix(delta_kappa, kappa1, kappa2, kappa3, l):  # noqa: E741  # TODO -> figure out singularity
    """
    * Calculation of jacobian matrix by numerical differentation    

    Parameters
    ----------
    delta_kappa : float
        parameter for numerical differentation. Commonly 0.1
    kappa1 : float
        curvature value for section 1.
    kappa2 : float
        curvature value for section 2.
    kappa3 : float
        curvature value for section 3.
    l : list
        trunk length contains all sections

    Returns
    -------
    J : Numpy array with the shape of (2,3)
        Jacobian Matrix

    """
    
    J11 = (three_section_planar_robot(kappa1+delta_kappa,kappa2,kappa3,l)[0,3] - three_section_planar_robot(kappa1-delta_kappa,kappa2,kappa3,l))[0,3] / (2*delta_kappa)
    J12 = (three_section_planar_robot(kappa1,kappa2+delta_kappa,kappa3,l)[0,3] - three_section_planar_robot(kappa1,kappa2-delta_kappa,kappa3,l))[0,3] / (2*delta_kappa)
    J13 = (three_section_planar_robot(kappa1,kappa2,kappa3+delta_kappa,l)[0,3] - three_section_planar_robot(kappa1,kappa2,kappa3-delta_kappa,l))[0,3] / (2*delta_kappa)
    J21 = (three_section_planar_robot(kappa1+delta_kappa,kappa2,kappa3,l)[1,3] - three_section_planar_robot(kappa1-delta_kappa,kappa2,kappa3,l))[1,3] / (2*delta_kappa)
    J22 = (three_section_planar_robot(kappa1,kappa2+delta_kappa,kappa3,l)[1,3] - three_section_planar_robot(kappa1,kappa2-delta_kappa,kappa3,l))[1,3] / (2*delta_kappa)
    J23 = (three_section_planar_robot(kappa1,kappa2,kappa3+delta_kappa,l)[1,3] - three_section_planar_robot(kappa1,kappa2,kappa3-delta_kappa,l))[1,3] / (2*delta_kappa)

    J = np.array([J11, J12, J13, J21, J22, J23])
    J = np.reshape(J, (2, 3))
    
    return J

# %% Function three_section_planar_robot

# Planar Robot Kinematics Functions
def trans_mat_cc(kappa, l):  # noqa: E741
    """
    *  Homogeneous transformation matrix
    *  Mapping from configuration parameters to task space
    * tip frame is aligned so that the x-axis points toward the center of the circle

    Parameters
    ----------
    kappa : list
        curvature value for all sections
    l : list
        trunk length contains all sections

    Returns
    -------
    T: numpy array
        Transformation matrices containing orientation and position

    """

    # num = sect_points: points per section
    si = np.linspace(0, l, num=50)
    T = np.zeros((len(si), 16))

    for i in range(len(si)):
        s = si[i]
        c_ks = np.cos(kappa * s)
        s_ks = np.sin(kappa * s)
        T[i, :] = np.array(
            [
                c_ks,
                s_ks,
                0,
                0,
                -s_ks,
                c_ks,
                0,
                0,
                0,
                0,
                1,
                0,
                _cos_delta_over_kappa(0.0, kappa, s),
                _sin_delta_over_kappa(0.0, kappa, s),
                0,
                1,
            ]
        )

    return T

# %% Function three_section_planar_robot
def coupletransformations(T,T_tip):
    """
    * The forward kinematics for an n section manipulator can then be generated by the product of 
    n matrices. The forward kinematics for our elephant trunk robot with its n sections can be calculated with this function
    * Find orientation and position of distal section (Multiply T of current section with T at tip of previous section)
    

    Parameters
    ----------
    T : np array
        Transformation matrices of current section.
    T_tip : np array
        Transformation at tip of previous section

    Returns
    -------
    Tc : np array
        coupled Transformation matrix

    """

    Tc=np.zeros((len(T[:,0]),len(T[0,:])))
    for k in range(len(T[:,0])):
        #Tc[k,:].reshape(-1,1)
        p = np.matmul(T_tip,(np.reshape(T[k,:],(4,4),order='F')))
        Tc[k,:] = np.reshape(p,(16,),order='F')
    return Tc
