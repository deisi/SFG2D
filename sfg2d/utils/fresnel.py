"""Module to calculate fresnelcorrection with 2 interfaces."""
import numpy as np
import yaml

from .static import nm_to_wavenumbers


def read_yml(fpath):
    """Function to read the yaml data base export files from
    /refractiveindex.info

    Returns a dictionary with the data stored at:
      ret["DATA"][0]['data']
    """
    with open(fpath, 'r') as stream:
        try:
            ret = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
        # -1 because last entry is empty
        ds = ret["DATA"][0]['data']
        ret["DATA"][0]['data'] = np.array([[float(entry) for entry in elm.split(' ')] for elm in ds.split('\n')[:-1]]).T
        return ret


def read_complexwave(fpath):
    """Read complex waves as they are exported from Igor"""
    ret = np.genfromtxt(fpath, skip_header=1)
    ret = ret[:, 0]+1j*ret[:, 1]
    return ret


def r_p(n, costheta):
    """Fresnelreflectivity for two interfaces.

    **Arguments:**
      - **n**: iterable with two entries for (n_0, n_1)
      - **costheta:** iterable with two entries for (costheta_0, costheta_1)
        costheta is used, because it can be complex.
    """
    i, j = 0, 1
    a = n[j] * costheta[i] - n[i] * costheta[j]
    b = n[j] * costheta[i] + n[i] * costheta[j]
    return a/b


def r_s(n, costheta):
    """Fresnelreflectivity for two interfaces.

    **Arguments:**
      - **n**: iterable with two entries for (n_0, n_1)
      - **theta:** iterable with two entries for (theta_0, theta_1)
    """
    i, j = 0, 1
    a = n[i] * costheta[i] - n[j] * costheta[j]
    b = n[i] * costheta[i] + n[j] * costheta[j]
    return a/b


def t_p(n, costheta):
    """Fresneltransmittance for two interfaces.

    **Arguments:**
      - **n**: iterable with two entries for (n_0, n_1)
      - **theta:** iterable with two entries for (theta_0, theta_1)
    """
    i, j = 0, 1
    a = 2 * n[i] * costheta[i]
    b = n[j] * costheta[i] + n[i] * costheta[j]
    return a/b


def t_s(n, costheta):
    """Fresneltransmittance for two interfaces.

    **Arguments:**
      - **n**: iterable with two entries for (n_0, n_1)
      - **theta:** iterable with two entries for (theta_0, theta_1)
    """
    i, j = 0, 1
    a = 2 * n[i] * costheta[i]
    b = n[i] * costheta[i] + n[j] * costheta[j]
    return a/b

def calc_prisim_angle(gamma, n1, n0=1, prisim=60):
    """Calc theta1 of gamma angle for prism.
    **Arguments**:
      - **gamma**: The incidence angle of the beam as defined
        in https://pubs.acs.org/doi/abs/10.1021/jp306273d
      - **n1**: refractive index of the prisim
      - **n0**: refractive index of the surounding medium 1 for air
      - **prisim**: Angle of the prisim

    **Returns:**
      the angle of the beam with respect to the surface normal
    """
    g0 = (prisim - gamma) * np.pi/180
    gamma1 = np.arcsin(n0*np.sin(g0)/n1) * 180/np.pi
    theta1 = prisim - gamma1
    return theta1

class RefractiveIndex:
    def __init__(self, wl, n):
        """Refractive index with wavelength in microns and refractive index n.
        This class automatically interpolates for wanted wavelengths.
        """
        self.wl = wl
        self.n = n

    def __call__(self, wl):
        ret = np.interp(wl, self.wl, self.n)
        return ret


class TwoInterfaces:
    def __init__(self, wl, n0, n1, n2, theta0, d, ninterface0, ninterface1):
        """Class to calculate fresnellcorrection factors for a 3 layer aka two
        interfaces system.
        **Arguments:**
          - **wl**: Wavelength values of interest. n0, n1 and n2, can all be
            arrays of the same length as wl or constant numbers.
          - **n0**: Refactive index of the first medium
          - **n1**: Refractive index of the second medium.
          - **n2**: Refractive index of the third medium.
          - **theta1**: Incidence Angle of the beam with respect to surface
            normal
          - **d**: Thickness of material with n1 in the same units as wl
          - **ninterface0**: Dominant field interface at the first interfacial
            layer
          - **ninterface1**: Dominant filed interface at the second
            interfactial layer

        """
        self.n0 = n0  # Refractive index of first medium, either glass of air
        self.n1 = n1  # Refractive index of intermediate medium.
        self.n2 = n2  # Refractive index of last medium usually h2o or d2o
        # A helping list of indeces
        self.n = (n0, n1, n2)
        # Incidencea angle of promary beam
        self.theta0 = theta0 * np.pi/180  # transforms theta1 to rad
        # Calculate angles with snellius. This doesnt work with the complex
        # ns, atleas it doesnt give the samre sult if I take real part
        # of self.sintheta or self.costheta
        #self.theta1 = np.arcsin(self.n0.real/self.n1.real*np.sin(self.theta0))
        #self.theta2 = np.arcsin(self.n1.real/self.n2.real*np.sin(self.theta1))
        # Because n_j is complex we use costheta or sintheta during
        # calculations. costheta and sintheta can be complex numbers.
        self.sintheta0 = np.sin(self.theta0)
        self.sintheta1 = self.calc_sintheta(1)
        self.sintheta2 = self.calc_sintheta(2)
        self.sintheta = (self.sintheta0, self.sintheta1, self.sintheta2)
        self.costheta0 = np.cos(self.theta0)
        self.costheta1 = self.calc_costheta(1)
        self.costheta2 = self.calc_costheta(2)
        # Helpful list of angles
        #self.theta = (self.theta0, self.theta1, self.theta2)
        self.costheta = (self.costheta0, self.costheta1, self.costheta2)
        # Array of wavelengths
        self.wl = wl
        # Thickness of intermediate material
        self.d = d
        # Phase difference due to thickness of intermediate material
        self.beta = 2 * np.pi/self.wl * self.n1 * self.d * self.costheta1

        # This is not fully correct because I would need a self.theta1 for VIS
        # and one for IR but I dont see why this is the case?
        self.delta = (2*np.pi*self.n2*d/self.wl) * \
            (1/self.costheta1-(self.sintheta1/self.costheta1))

        # Has todo with where sfg is generated
        self.ninterface0 = ninterface0
        self.ninterface1 = ninterface1

    def calc_costheta(self, j):
        """Calculate complex costheta values for given interface."""
        # Here no abs because its not defined like that.
        a = np.sqrt(self.n[j]**2 - self.n[0]**2 * self.sintheta0**2, dtype=np.complex)
        return a / self.n[j]

    def calc_sintheta(self, j):
        """Calculate complex sintheta values for given interface."""
        return self.n0/self.n[j] * self.sintheta0

    def r_p(self, i, j):
        """P-Reflectivity  at layer i -> j."""
        n, theta = (self.n[i], self.n[j]), (self.costheta[i], self.costheta[j])
        return r_p(n, theta)

    def r_s(self, i, j):
        """S-Reflectivity  at layer i -> j."""
        n, theta = (self.n[i], self.n[j]), (self.costheta[i], self.costheta[j])
        return r_s(n, theta)

    def t_p(self, i, j):
        """P-Transmittance  at layer i -> j."""
        n, theta = (self.n[i], self.n[j]), (self.costheta[i], self.costheta[j])
        return t_p(n, theta)

    def t_s(self, i, j):
        """S-Transmittance  at layer i -> j."""
        n, theta = (self.n[i], self.n[j]), (self.costheta[i], self.costheta[j])
        return t_s(n, theta)

    @property
    def L_xx_1(self):
        r_p_12 = self.r_p(1, 2)
        eb = np.exp(2*1j*self.beta)

        a = self.t_p(0, 1) / (1+self.r_p(0, 1)*r_p_12*eb)
        b = 1 - r_p_12 * eb
        c = self.costheta1 / self.costheta0
        return a * b * c

    @property
    def L_xx_2(self):
        raise ValueError("Delta not implemented.")
        r_p_12 = self.r_p(1, 2)
        a0 = np.exp(1j * self.delta) * self.t_p(0, 1)
        a1 = 1 + self.r_p(0, 1)*r_p_12*np.exp(2*1j*self.beta)
        b = 1 - r_p_12
        c = self.costheta1/self.costheta0
        return a0/a1 * b * c

    @property
    def L_yy_1(self):
        r_s_12 = self.r_s(1, 2)
        a = self.t_s(0, 1) / (1+self.r_s(0, 1)*r_s_12*np.exp(2*1j*self.beta))
        b = 1 + r_s_12 * np.exp(2 * 1j * self.beta)
        return a * b

    @property
    def L_yy_2(self):
        raise ValueError("Delta not implemented")
        r_s_12 = self.r_s(1, 2)
        a0 = np.exp(1j*self.delta)*self.t_s(0, 1)
        a1 = 1+self.r_s(0, 1)*r_s_12*np.exp(2*1j*self.beta)
        b = 1 + r_s_12
        return a0/a1 * b

    @property
    def L_zz_1(self):
        r_p_12 = self.r_p(1, 2)
        a = self.t_p(0, 1) / (1+self.r_p(0, 1)*r_p_12*np.exp(2*1j*self.beta))
        b = 1 + r_p_12 * np.exp(2 * 1j * self.beta)
        c = self.n0 * self.n1/(self.ninterface0)**2
        return a*b*c

    @property
    def L_zz_2(self):
        raise ValueError("Delta not implemented")
        r_p_12 = self.r_p(1, 2)
        a0 = np.exp(1j*self.delta)*self.t_p(0, 1)
        a1 = 1 + self.r_p(0, 1)*r_p_12*np.exp(2*1j*self.beta)
        b = 1 + r_p_12
        c = self.n0 * self.n1/(self.ninterface1)**2
        return a0/a1*b*c

    @property
    def I_xx_1(self):
        """Square if L_xx_1."""
        return abs(self.L_xx_1)**2

    @property
    def I_xx_2(self):
        "Square of L_xx_2"

        # exp(J*Delta) becomes 1 on square
        r_p_12 = self.r_p(1, 2)
        a0 = self.t_p(0, 1)
        a1 = 1+self.r_p(0, 1)*r_p_12*np.exp(2*1j*self.beta)
        b = 1 - r_p_12
        c = self.costheta1/self.costheta0
        i_xx = a0/a1 * b * c
        return np.abs(i_xx)**2

    @property
    def I_yy_1(self):
        """Square of L_yy_1"""
        return np.abs(self.L_yy_1)**2

    @property
    def I_yy_2(self):
        """Square of L_yy_2"""
        # exp(J*Delta) becomes 1 on square
        r_s_12 = self.r_s(1, 2)
        a0 = self.t_s(0, 1)
        a1 = 1+self.r_s(0, 1)*r_s_12*np.exp(2*1j*self.beta)
        b = 1 + r_s_12
        i_yy = a0/a1 * b
        return np.abs(i_yy)**2

    @property
    def I_zz_1(self):
        """Square of L_zz_1"""
        return np.abs(self.L_zz_1)**2

    @property
    def I_zz_2(self):
        """Square of L_zz_2"""
        r_p_12 = self.r_p(1, 2)
        a0 = self.t_p(0, 1)
        a1 = 1 + self.r_p(0, 1)*r_p_12*np.exp(2*1j*self.beta)
        b = 1 + r_p_12
        c = self.n0 * self.n1/(self.ninterface1)**2
        i_zz = a0/a1*b*c
        return np.abs(i_zz)**2
