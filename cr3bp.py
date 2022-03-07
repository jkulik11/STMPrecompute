#code adapted from Gabe Soto by Jackson Kulik
import astropy.units as u
import astropy.constants as const
import numpy as np



#initialize global variables
mu = const.M_earth/ (const.M_earth + const.M_sun)
mu = mu.value # mass fraction for Earth-Sun system

# solving for Lagrange points
coeff = [1., 3.-mu, 3.-2.*mu, -mu, -2.*mu, -mu]
roots = np.roots(coeff) 
g2 = np.real(roots[np.isreal(roots)])[0] # L2 location rel Earth
L2 = g2 + (1.-mu) # L2 location rel rotating frame

G = 1.
m1 = float(1.-mu) # bigger primary (Sun)
m2 = mu # smaller primary (Earth)



#methods for to call from wolfram script

def getMu():
	return mu

def getL2():
	return L2

def setMoon():
	mu = const.M_moon/ (const.M_moon + const.M_earth)
	mu = mu.value # mass fraction for Earth-Sun system

	# solving for Lagrange points
	coeff = [1., 3.-mu, 3.-2.*mu, -mu, -2.*mu, -mu]
	roots = np.roots(coeff) 
	g2 = np.real(roots[np.isreal(roots)])[0] # L2 location rel Earth
	L2 = g2 + (1.-mu) # L2 location rel rotating frame

	G = 1.
	m1 = float(1.-mu) # bigger primary (Sun)
	m2 = mu # smaller primary (Earth)

#flag is true if km, false if AU
#m (integer 1x1): defines halo orbit direction. m=1 is a Northern Halo and m = 3 is a Southern Halo.
def getThirdOrderHaloApprox(Az,m,flag):
	if (flag):
		return thirdOrderHaloApprox(Az*u.km, m, 0.)
	return thirdOrderHaloApprox(Az*u.au, m, 0.)

def getLissajousLinearApprox(Ax,Az,flag):
	if (flag):
		return linearApprox(Ax*u.km,Az*u.km, 0.)
	return linearApprox(Ax*u.au,Az*u.au, 0.)


#original code due to Gabe Soto


def c_fun(n):
        """ Method to generate coefficients for CR3BP approximation
        
        Generates c_n coefficients for the nth order approximation of the
        Legendre polynomial expansion of the CR3BP equations of motion. 
        Based on Richardson (1980)
        """
        
        g  = g2
        cN = (1/g**3) * ( ((-1)**n)*mu + \
                ((-1)**n)*(1-mu)*g**(n+1)/(1+g)**(n+1))
        
        return cN
    
def linearApprox(Ax,Az,t,convert=True):
    """Linear approximation of a Lissajous
    
    First order approximation of a periodic orbit about the Lagrangian 
    point. Resembles a Lissajous orbit.
    
    Args:
        Ax (astropy Quantity 1x1):
            Amplitude of orbit in the ecliptic plane. Must be in astropy
            units (km, for instance).
        Az (astropy Quantity 1x1):
            Amplitude of orbit out of the ecliptic plane. Must be in 
            astropy units (km, for instance).
        t (float nx1):
            Times in normalized units
    Returns:
        x,y,z,dx,dy,dz,T  (float nx1 each, T is 1x1):
            Resultant states x,y,z,dx,dy,dz of the linear periodic 
            approximation about the Lagrangian point. T is the period of
            the orbit.
    """
    
    # Calculating frequencies and other constants
    c2 = c_fun(2)
    
    wp =  0.5*(c2-2-np.sqrt(9*c2**2-8*c2))
    wp = np.sqrt(-wp)
    
    wv = np.sqrt(c2)
    
    k= (wp**2 + 1 + 2*c2)/(2*wp)
    
    # conversion factor
    Ax = Ax.to('au').value/g2
    Az = Az.to('au').value/g2
    
    # Resultant states
    x = -Ax*np.cos(wp*t)
    y = k*Ax*np.sin(wp*t)
    z = Az*np.sin(wv*t)
    dx = wp*Ax*np.sin(wp*t)
    dy = wp*k*Ax*np.cos(wp*t)
    dz = wv*Az*np.cos(wv*t)
    
    # period of orbit
    T = 2*np.pi*(1/wv)
    
    if convert:
        # convert back to rotating frame with origin at center of mass
        x = g2*x + 1 - mu + g2
        y = g2*y
        z = g2*z
        
        dx = g2*dx
        dy = g2*dy
        dz = g2*dz
    
    return x,y,z,dx,dy,dz,T


def thirdOrderHaloApprox(Az,m,t):
        """Richardson Third Order Halo orbit
        
        Third order analytical approximation of a halo orbit. 
        
        Args:
            Az (astropy Quantity 1x1):
                Amplitude of orbit out of the ecliptic plane. Must be in 
                astropy units (km, for instance).
            m (integer 1x1):
                Defines halo orbit direction. m=1 is a Northern Halo and 
                m = 3 is a Southern Halo.
            t (float nx1):
                Times in normalized units
        Returns:
            x,y,z,dx,dy,dz,T  (float nx1 each, T is 1x1):
                Resultant states x,y,z,dx,dy,dz of the linear periodic 
                approximation about the Lagrangian point. T is the period of
                the orbit.
        """
        
        # c-constants used in the expansion
        c2 = c_fun(2)
        c3 = c_fun(3)
        c4 = c_fun(4)
        
        # frequencies 
        wp =  0.5*(c2-2-np.sqrt(9*c2**2-8*c2))
        wp = np.sqrt(-wp)
        
        k= (wp**2 + 1 + 2*c2)/(2*wp)
        
        # some annoying coefficients
        d1 = (3*wp**2/k)*(k*(6*wp**2- 1)-2*wp)
        d2 = (8*wp**2/k)*(k*(11*wp**2- 1)-2*wp)
        
        a21 = 3*c3*(k**2 - 2)/(4*(1 + 2*c2))
        a22 = 3*c3/(4*(1 + 2*c2))
        a23 = -(3*c3*wp)/(4*k*d1) * (3*k**3*wp - 6*k*(k-wp)+4)
        a24 = -(3*c3*wp)/(4*k*d1) * (2 + 3*k*wp)    
        b21 = -(3*c3*wp)/(2*d1)   * (3*k*wp - 4)
        b22 =  (3*c3*wp)/(d1)
        d21 = - c3/(2*wp**2)
        a31 = -(9*wp)/(4*d2) * (4*c3*(k*a23-b21)+k*c4*(4+k**2)) + \
               (9*wp**2 + 1 - c2)/(2*d2) * (3*c3*(2*a23-k*b21)+c4*(2 +3*k**2))
        a32 = (-1/d2)*( (9*wp/4.)*(4*c3*(k*a24-b22)+k*c4) + \
              (3/2.)*(9*wp**2 + 1 - c2)*(c3*(k*b22+d21 - 2*a24) - c4))
        
        b31 = (3)/(8*d2) *( (8*wp)*(3*c3*(k*b21- 2*a23) - c4*(2 +3*k**2)) + \
              (9*wp**2 + 1 + 2*c2)*(4*c3*(k*a23-b21)+k*c4*(4+k**2)))
        b32 = (1)/(d2) *( (9*wp)*(c3*(k*b22+d21- 2*a24) - c4) + \
              (3/8.)*(9*wp**2 + 1 + 2*c2)*(4*c3*(k*a24-b22)+k*c4))
        d31 = (3)/(64*wp**2) * (4*c3*a24 + c4)
        d32 = (3)/(64*wp**2) * (4*c3*(a23-d21) + c4*(4+k**2))
        s1  = (2*wp*(wp*(1+k**2)-2*k))**(-1) * ((1.5*c3*(2*a21*(k**2- 2) \
               - a23*(k**2+ 2)- 2*k*b21)-(3/8.)*c4*(3*k**4- 8*k**2+ 8)))
        s2  = (2*wp*(wp*(1+k**2)-2*k))**(-1) * ((1.5*c3*(2*a22*(k**2- 2) \
               + a24*(k**2+ 2)+ 2*k*b22 + 5*d21)+(3/8.)*c4*(12- k**2)))
           
        l1  = - (1.5)*c3*(2*a21 + a23 + 5*d21) - (3/8.)*c4*(12-k**2) + 2*wp**2*s1
        l2  =   (1.5)*c3*(a24- 2*a22) + (9/8.)*c4 + 2*wp**2*s2
        
        # conversion factor
        Az = Az.to('au').value/g2
        
        # even more constants
        Del = wp**2 - c2
        Ax = np.sqrt((-l2*Az**2-Del)/l1)  # from triangle equality, obtain x-amplitude
        v = 1 + s1*Ax**2 + s2*Az**2       # correction factor for time
        gm = 2 - m                        # halo direction
        
        # third order states
        x = a21*Ax**2 + a22*Az**2 - Ax*np.cos(t) + (a23*Ax**2-a24*Az**2)*np.cos(2*t) + (a31*Ax**3 - a32*Ax*Az**2)*np.cos(3*t)
        y = k*Ax*np.sin(t) + (b21*Ax**2-b22*Az**2)*np.sin(2*t) + (b31*Ax**3 - b32*Ax*Az**2)*np.sin(3*t)
        z = gm*Az*np.cos(t) + gm*d21*Ax*Az*(np.cos(2*t)-3) + gm*(d32*Az*Ax**2-d31*Az**3)*np.cos(3*t)
        
        dx = Ax*np.sin(t) - 2*(a23*Ax**2-a24*Az**2)*np.sin(t) - 3*(a31*Ax**3 - a32*Ax*Az**2)*np.sin(3*t)
        dy = k*Ax*np.cos(t) + 2*(b21*Ax**2-b22*Az**2)*np.cos(2*t) + 3*(b31*Ax**3-b32*Ax*Az**2)*np.cos(3*t)
        dz = -gm*Az*np.sin(t) - 2*gm*d21*Ax*Az*np.sin(2*t) + 3*gm*(d32*Az*Ax**2-d31*Az**3)*np.sin(3*t)

        T = 2*np.pi/(wp*v)
        
        # convert back to rotating frame with origin at center of mass
        x = g2*x + 1 - mu + g2
        y = g2*y
        z = g2*z
        
        dx = g2*wp*v*dx
        dy = g2*wp*v*dy
        dz = g2*wp*v*dz
        
        return x,y,z,dx,dy,dz,T
