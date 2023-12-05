import matplotlib.pyplot as plt 
import math
import numpy as np 
import pandas as pd
from sympy import symbols, Eq, solve
from scipy.interpolate import interp1d
import sys

# User Input Initial Temperature & Constant Pressure
phi =1
T1 = 298 # Reactants Temperature
P =1 # atm of Chemical Reaction

# Equilibrium Constant Table 
df = pd.read_excel('Kp.xlsx')
Temperatures = df["Temperatures"]
lnKp1 = df['lnKp1'] # H2 <=> 2H
lnKp2 = df['lnKp2'] # O2 <=> 2O
lnKp3 = df['lnKp3'] # N2 <=> 2N
lnKp4 = df['lnKp4'] # H2O <=> H2 + 1/2 O2
lnKp5 = df['lnKp5'] # H2O <=> 1/2 H2 + OH
lnKp6 = df['lnKp6'] # CO2 <=> CO + 1/2 O2
lnKp7 = df['lnKp7'] # 1/2 N2 + 1/2 O2 <=> NO

# Reactants, or products < 1000K :
C8H18_coef_l = [-4.20868893E+00, 1.11440581E-01, -7.91346582E-05, 2.92406242E-08, -4.43743191E-12, -2.99446875E+04]
O2_coef_l  = [3.78245636E+00, -2.99673416E-03, 9.84730201E-06, -9.68129509E-09, 3.24372837E-12, -1.06394356E+03]
N2_coef_l  = [0.03298677E+02, 0.14082404E-02, -0.03963222E-04, 0.05641515E-07, -0.02444854E-10, -0.10208999E+04]

H2O_coef_l = [4.19864056E+00, -2.03643410E-03, 6.52040211E-06, -5.48797062E-09, 1.77197817E-12, -3.02937267E+04]
CO_coef_l  = [3.57953347E+00, -6.10353680E-04, 1.01681433E-06, 9.07005884E-10, -9.04424499E-13, -1.43440860E+04]
CO2_coef_l = [2.35677352E+00, 8.98459677E-03, -7.12356269E-06, 2.45919022E-09, -1.43699548E-13, -4.83719697E+04]

OH_coef_l  = [4.12530561E+00, -3.22544939E-03, 6.52764691E-06, -5.79853643E-09, 2.06237379E-12, 3.38153812E+03]
H_coef_l   = [2.50000000E+00, 7.05332819E-13, -1.99591964E-15, 2.30081632E-18, -9.27732332E-22, 2.54736599E+04]
O_coef_l   = [3.16826710E+00, -3.27931884E-03, 6.64306396E-06, -6.12806624E-09, 2.11265971E-12, 2.91222592E+04]
N_coef_l   = [0.25000000E+01, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.56104637E+05]

H2_coef_l  = [2.34433112E+00, 7.98052075E-03, -1.94781510E-05, 2.01572094E-08, -7.37611761E-12, -9.17935173E+02]
NO_coef_l  = [0.42184763E+01, -0.46389760E-02, 0.11041022E-04, -0.93361354E-08, 0.28035770E-11, 0.98446230E+04]
Ar_coef_l  = [2.50000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, -7.45375000E+02]

# Products
CO2_coef_h = [3.85746029E+00, 4.41437026E-03, -2.21481404E-06, 5.23490188E-10, -4.72084164E-14, -4.87591660E+04]
CO_coef_h  = [2.71518561E+00, 2.06252743E-03, -9.98825771E-07, 2.30053008E-10, -2.03647716E-14, -1.41518724E+04]
O2_coef_h  = [3.28253784E+00, 1.48308754E-03, -7.57966669E-07, 2.09470555E-10, -2.16717794E-14, -1.08845772E+03]
H2O_coef_h = [3.03399249E+00, 2.17691804E-03, -1.64072518E-07, -9.70419870E-11, 1.68200992E-14, -3.00042971E+04]
N2_coef_h  = [0.02926640E+02, 0.14879768E-02, -0.05684760E-05, 0.10097038E-09, -0.06753351E-13, -0.09227977E+04]

Ar_coef_h  = [2.50000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, 0.00000000E+00, -7.45375000E+02]
H2_coef_h  = [3.33727920E+00, -4.94024731E-05, 4.99456778E-07, -1.79566394E-10, 2.00255376E-14, -9.50158922E+02]
OH_coef_h  = [2.86472886E+00, 1.05650448E-03, -2.59082758E-07, 3.05218674E-11, -1.33195876E-15, 3.71885774E+03]
NO_coef_h  = [0.32606056E+01, 0.11911043E-02, -0.42917048E-06, 0.69457669E-10, -0.40336099E-14, 0.99209746E+04]
H_coef_h   = [2.50000001E+00, -2.30842973E-11, 1.61561948E-14, -4.73515235E-18, 4.98197357E-22, 2.54736599E+04]
O_coef_h   = [2.56942078E+00, -8.59741137E-05, 4.19484589E-08, -1.00177799E-11, 1.22833691E-15, 2.92175791E+04]
N_coef_h   = [0.24159429E+01, 0.17489065E-03, -0.11902369E-06, 0.30226245E-10, -0.20360982E-14, 0.56133773E+05]



R  = 8.314 #J/mol-k

# To evaluate enthalpy
def fun_1(T, coef):
	
	a1 = coef[0]
	a2 = coef[1]
	a3 = coef[2]
	a4 = coef[3]
	a5 = coef[4]
	a6 = coef[5]

	# NASA polynomial form for calculation of enthalpy
	# H/RT = a1 + a2 T /2 + a3 T^2 /3 + a4 T^3 /4 + a5 T^4 /5 + a6/T
	return (a1 + a2*(T/2) + a3*pow(T,2)/3 + a4*pow(T,3)/4 + a5*pow(T,4)/5 + (a6)/T)*(R*T)


# ***************************************************************** #
# *************************** Main Loop *************************** #
# ***************************************************************** #

a = 0.5/ phi  # Calculate 'a' based on the user input

# Initialize lists to store data
H_values = []


no_solution_found = True  # Flag to track if no solution is found for any temperature

for temperature_index in range(len(Temperatures)):
    temperature_value = Temperatures.iloc[temperature_index]
    Kp1 = math.exp(lnKp1.iloc[temperature_index])
    Kp2 = math.exp(lnKp2.iloc[temperature_index])
    Kp3 = math.exp(lnKp3.iloc[temperature_index])
    Kp4 = math.exp(lnKp4.iloc[temperature_index])
    Kp5 = math.exp(lnKp5.iloc[temperature_index])
    Kp7 = math.exp(lnKp7.iloc[temperature_index])

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9')
    # From Atom Balance
    eq1 = Eq(2, 2*x1 + 2*x2 + x5 + x7)
    eq2 = Eq(2 * a, x1 + 2*x3 + x5 + x6 + x8)
    eq3 = Eq(3.76*a*2, 2*x4 + x6 + x9)

    # From Equilibrium Constants 
    eq4 = Eq(Kp1, x7^2/x2*(P/(x1+x2+x3+x4+x5+x6+x7+x8+x9)))
    eq5 = Eq(Kp2, x8^2/x3*(P/(x1+x2+x3+x4+x5+x6+x7+x8+x9)))
    eq6 = Eq(Kp3, x9^2/x4*(P/(x1+x2+x3+x4+x5+x6+x7+x8+x9)))
    eq7 = Eq(Kp4, x3^0.5*x2/x1*(P/(x1+x2+x3+x4+x5+x6+x7+x8+x9))^0.5)
    eq8 = Eq(Kp5, x2^0.5*x5/x1*(P/(x1+x2+x3+x4+x5+x6+x7+x8+x9))^0.5)
    eq9 = Eq(Kp7, x6/(x4^0.5*x3^0.5))

    solutions = solve((eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9), (x1, x2, x3, x4, x5, x6, x7, x8, x9))

    if not solutions or any(not all(sol.is_real for sol in solution_set) for solution_set in solutions):
        print(f"No real/positive solutions at T = {temperature_value}K")
        continue

    no_solution_found = False  # Set the flag to False since a solution was found for this temperature

    for solution_set in solutions:
        x1_value, x2_value, x3_value, x4_value, x5_value, x6_value, x7_value, x8_value, x9_value= solution_set
        if x1_value > 0 and x2_value > 0 and x3_value > 0 and x4_value > 0 and x5_value > 0 and x6_value > 0 and x7_value > 0 and x8_value > 0 and x9_value > 0:
            H_reactants = 1 * fun_1(T1, H2_coef_l) + a * fun_1(T1, O2_coef_l) + 3.76 * a * fun_1(T1, N2_coef_l)

            if temperature_value < 1000:
                H_total = x1_value*fun_1(temperature_value, H2O_coef_l) + x2_value*fun_1(temperature_value, H2_coef_l) +\
                x3_value*fun_1(temperature_value, O2_coef_l) + x4_value*fun_1(temperature_value, N2_coef_l) + \
                x5_value*fun_1(temperature_value, OH_coef_l) + x6_value*fun_1(temperature_value, NO_coef_l) + \
                x7_value*fun_1(temperature_value, H_coef_l) + x8_value*fun_1(temperature_value, O_coef_l) + \
                x9_value*fun_1(temperature_value, N_coef_l) - H_reactants
               
            else:
                H_total = x1_value*fun_1(temperature_value, H2O_coef_h) + x2_value*fun_1(temperature_value, H2_coef_h) +\
                x3_value*fun_1(temperature_value, O2_coef_h) + x4_value*fun_1(temperature_value, N2_coef_h) + \
                x5_value*fun_1(temperature_value, OH_coef_h) + x6_value*fun_1(temperature_value, NO_coef_h) + \
                x7_value*fun_1(temperature_value, H_coef_h) + x8_value*fun_1(temperature_value, O_coef_h) + \
                x9_value*fun_1(temperature_value, N_coef_h) - H_reactants

            print(f"At temperature {temperature_value}: x1 = {x1_value}, x2 = {x2_value}, x3 = {x3_value},  H = {H_total}")
            print(f"x4 = {x4_value}, x5 = {x5_value}, x6 = {x6_value}, x7 = {x7_value}, x8 = {x8_value}, x9 = {x9_value},")

            # Store data
            H_values.append(H_total)


# Check if no solution was found for any temperature
if no_solution_found:
    print("No real/positive solutions found for any temperature. Choose another value for phi. Exiting script.")
    sys.exit()


# Convert Temperatures and H_total_values to numpy arrays
temperatures_array = np.array(Temperatures)
h_total_values_array = np.array(H_values)

temperatures_array = temperatures_array.astype(float)
h_total_values_array = h_total_values_array.astype(float)


print(len(h_total_values_array))
print(len(temperatures_array))


# Create an interpolation function
interp_function  = interp1d(h_total_values_array, temperatures_array, kind='linear', fill_value='extrapolate')


# Find the temperature where H_total is approximately zero
temperature_at_zero_enthalpy = interp_function(0)



print(f"The adiabatic temperature a the given equivalence ratio {phi} is {temperature_at_zero_enthalpy}K ")

