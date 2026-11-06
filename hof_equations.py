"""Auto-generated from outputs/20261021_115947_0ebgKa/hall_of_fame.csv -- do not edit by hand.

The distinct Hall-of-Fame entries with complexity >= 3, as a
drop-in replacement for f_rect (same signature and units):

    f(a, b, h, m, n, E, nu, rho)

CALL CONVENTION (matters!): b is the TRUE side length, same unit as a. The
slot is fed the ratio a/b internally, so you pass the width, not the ratio.

Constants rounded to 4 significant digits.
E in Pa, rho in kg/m^3, frequencies in the unit the search was fitted in.
"""
import numpy as np

import numpy as np


# complexity 4, loss 0.01328652
# HoF slot: f = inv((a/b) + 1.822253332938699)
f_cx4 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + (1.0/(1.0*a/b + 1.822)))/a)**2
       + ((n + (1.0/(1.822 + 1.0*b/a)))/b)**2)
)

# complexity 5, loss 0.008718533
# HoF slot: f = ((a/b) * -0.28191083172659026) + 0.6330989285187787
f_cx5 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + (-0.2819*a/b + 0.6331))/a)**2
       + ((n + (0.6331 - 0.2819*b/a))/b)**2)
)

# complexity 8, loss 0.005089214
# HoF slot: f = inv((n / (m / (a/b))) + 1.857413302803714)
f_cx8 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + (1.0*m/(1.0*a*n/b + 1.857*m)))/a)**2
       + ((n + (1.0*n/(1.857*n + 1.0*b*m/a)))/b)**2)
)

# complexity 10, loss 0.004013893
# HoF slot: f = inv((((n / m) - -0.14965691200027206) * (a/b)) + 1.7253557869309337)
f_cx10 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + (1.0*m/(0.1497*a*m/b + 1.0*a*n/b + 1.725*m)))/a)**2
       + ((n + (1.0*n/(1.725*n + 1.0*b*m/a + 0.1497*b*n/a)))/b)**2)
)

# complexity 11, loss 0.003660755
# HoF slot: f = 0.8903786682667717 / ((((n * (a/b)) + -0.14921594460921261) / m) + 1.6621702759384949)
f_cx11 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + (0.8904*m/(1.0*a*n/b + 1.662*m - 0.1492)))/a)**2
       + ((n + (0.8904*n/(1.662*n - 0.1492 + 1.0*b*m/a)))/b)**2)
)

# complexity 12, loss 0.002834247
# HoF slot: f = inv(((((a/b) * n) - 0.2863284674558266) / (m + -0.22387394019658288)) + 1.9226710848342843)
f_cx12 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + ((1.0*m - 0.2239)/(1.0*a*n/b + 1.923*m - 0.7168)))/a)**2
       + ((n + ((1.0*n - 0.2239)/(1.923*n - 0.7168 + 1.0*b*m/a)))/b)**2)
)

# complexity 13, loss 0.002786063
# HoF slot: f = inv(((n + -0.5955823135338705) * (square((a/b)) / (m + -0.54526536156881))) + 1.9450416414338547)
f_cx13 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + ((1.0*m - 0.5453)/(1.0*a**2*n/b**2 - 0.5956*a**2/b**2 + 1.945*m - 1.061)))/a)**2
       + ((n + ((1.0*n - 0.5453)/(1.945*n - 1.061 + 1.0*b**2*m/a**2 - 0.5956*b**2/a**2)))/b)**2)
)

# complexity 15, loss 0.002559345
# HoF slot: f = 0.7871902774677655 / ((0.14661148114513833 / n) + ((((n * (a/b)) + -0.24392247402887307) / m) + 1.331836733788762))
f_cx15 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + (0.7872*m*n/(1.0*a*n**2/b + 1.332*m*n + 0.1466*m - 0.2439*n)))/a)**2
       + ((n + (0.7872*m*n/(1.332*m*n - 0.2439*m + 0.1466*n + 1.0*b*m**2/a)))/b)**2)
)

# complexity 17, loss 0.002536367
# HoF slot: f = (0.6133208269056245 / ((((((a/b) * n) + -0.26459789744227047) / m) - (-0.1645427084987661 / n)) + 1.1088081249779862)) + 0.04656455205170639
f_cx17 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + ((0.04656*a*n**2/b + 0.665*m*n + 0.007662*m - 0.01232*n)/(1.0*a*n**2/b + 1.109*m*n + 0.1645*m - 0.2646*n)))/a)**2
       + ((n + ((0.665*m*n - 0.01232*m + 0.007662*n + 0.04656*b*m**2/a)/(1.109*m*n - 0.2646*m + 0.1645*n + 1.0*b*m**2/a)))/b)**2)
)

# complexity 19, loss 0.0025056
# HoF slot: f = ((a/b) * 0.019592063294251637) + (0.672702387173554 / ((((((a/b) * n) + -0.27747723085716125) / m) + 1.119000199287073) - (-0.1833788937747947 / n)))
f_cx19 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + ((0.01959*a**2*n**2/b**2 + 0.02192*a*m*n/b + 0.003593*a*m/b - 0.005436*a*n/b + 0.6727*m*n)/(1.0*a*n**2/b + 1.119*m*n + 0.1834*m - 0.2775*n)))/a)**2
       + ((n + ((0.6727*m*n + 0.02192*b*m*n/a - 0.005436*b*m/a + 0.003593*b*n/a + 0.01959*b**2*m**2/a**2)/(1.119*m*n - 0.2775*m + 0.1834*n + 1.0*b*m**2/a)))/b)**2)
)

# complexity 20, loss 0.002300046
# HoF slot: f = ((inv(n) * ((-0.03498646935257077 / (a/b)) - 0.05105283613951161)) + 0.7436224396421663) / ((((n * (a/b)) + -0.33829068688123104) / m) + 1.2070749551565392)
f_cx20 = lambda a, b, h, m, n, E, nu, rho: (
    np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))
    * (((m + ((0.7436*a*m*n/b - 0.05105*a*m/b - 0.03499*m)/(1.0*a**2*n**2/b**2 + 1.207*a*m*n/b - 0.3383*a*n/b)))/a)**2
       + ((n + ((-0.03499*n + 0.7436*b*m*n/a - 0.05105*b*n/a)/(1.207*b*m*n/a - 0.3383*b*m/a + 1.0*b**2*m**2/a**2)))/b)**2)
)

HOF_MODELS = {
    "f_cx4": f_cx4,
    "f_cx5": f_cx5,
    "f_cx8": f_cx8,
    "f_cx10": f_cx10,
    "f_cx11": f_cx11,
    "f_cx12": f_cx12,
    "f_cx13": f_cx13,
    "f_cx15": f_cx15,
    "f_cx17": f_cx17,
    "f_cx19": f_cx19,
    "f_cx20": f_cx20,
}
