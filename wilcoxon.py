from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

acc1 = [0.63, 0.76, 0.76, 0.86, 0.99, 0.96, 0.97, 0.82, 0.79, 0.74, 0.77, 0.99, 0.85, 0.96, 0.96]
acc2 = [0.63, 0.75, 0.77, 0.86, 0.99, 0.96, 0.97, 0.81, 0.77, 0.70, 0.75, 0.99, 0.83, 0.91, 0.97]
acc3 = [0.86, 0.89, 1, 0.9, 1, 0.89, 0.91, 0.87, 0.87, 0.74, 0.78, 0.99, 0.61, 0.83, 0.88]
acc3p = [0.92, 0.92, 1, 0.91, 1, 0.98, 0.99, 0.92, 0.89, 0.76, 0.79, 0.99, 0.76, 0.91, 0.92]
acc4 = [0.88, 0.85, 1, 0.9, 1, 0.91, 0.92, 0.91, 0.91, 0.75, 0.77, 1, 0.8, 0.9, 0.93]
acc4p = [0.93, 0.91, 1, 0.9, 1, 0.98, 0.99, 0.91, 0.93, 0.77, 0.78, 0.99, 0.83, 0.81, 0.95]
gmean1 =[0.62, 0.76, 0.75, 0.85, 0.99, 0.95, 0.97, 0.74, 0.61, 0.36, 0.73, 0.98, 0.14, 0.69, 0.79]
gmean2 = [0.62, 0.75, 0.78, 0.85, 0.99, 0.95, 0.97, 0.78, 0.76, 0.72, 0.72, 1.0, 0.54, 0.94, 0.9]
gmean3 =[0.86, 0.88, 1, 0.89, 1, 0.88, 0.89, 0.85, 0.85, 0.67, 0.75, 0.99, 0.14, 0.79, 0.87]
gmean3p =[0.92, 0.92, 1, 0.9, 1, 0.98, 0.97, 0.92, 0.87, 0.62, 0.76, 0.99, 0.27, 0.58, 0.72]
gmean4 =[0.88, 0.84, 1, 0.9, 1, 0.91, 0.91, 0.91, 0.92, 0.75, 0.72, 1, 0.72, 0.91, 0.93]
gmean4p =[0.94, 0.9, 1, 0.9, 1, 0.98, 0.99, 0.91, 0.93, 0.69, 0.73, 1, 0.74, 0.78, 0.87]
f1 = [0.58, 0.77, 0.8, 0.87, 0.99, 0.95, 0.96, 0.69, 0.53, 0.84, 0.75, 0.98, 0.11, 0.63, 0.69]
f2 = [0.58, 0.74, 0.78, 0.87, 0.99, 0.95, 0.96, 0.72, 0.66, 0.76, 0.69, 0.98, 0.38, 0.63, 0.77]
f3 = [0.85, 0.89, 1, 0.9, 1, 0.87, 0.87, 0.83, 0.84, 0.8, 0.75, 0.99, 0.11, 0.77, 0.85]
f3p =[0.91, 0.92, 1, 0.92, 1, 0.99, 0.96, 0.91, 0.87, 0.82, 0.74, 0.99, 0.25, 0.57, 0.65]
f4 = [0.89, 0.85, 1, 0.91, 1, 0.91, 0.89, 0.89, 0.9, 0.75, 0.76, 1, 0.67, 0.9, 0.94]
f4p = [0.94, 0.9, 1, 0.91, 1, 0.98, 0.99, 0.91, 0.92, 0.76, 0.73, 0.97, 0.68, 0.6, 0.76]
prec1 = [0.66, 0.8, 0.74, 0.85, 0.98, 0.97, 0.96, 0.83, 0.81, 0.74, 0.78, 1, 0.3, 0.7, 0.71]
prec2 = [0.66, 0.85, 0.83, 0.85, 0.98, 0.95, 0.95, 0.69, 0.61, 0.88, 0.72, 0.97, 0.48, 0.53, 0.75]
prec3 = [0.87, 0.92, 1, 0.92, 1, 0.88, 0.92, 0.94, 0.92, 0.74, 0.81, 1, 0.29, 0.92, 0.88]
prec3p = [0.97, 0.92, 1, 0.92, 1, 0.99, 0.99, 0.97, 0.91, 0.72, 0.84, 1, 0.4, 0.67, 0.67]
prec4 = [0.88, 0.9, 1, 0.9, 1, 0.88, 0.91, 0.9, 0.88, 0.84, 0.77, 1, 0.84, 0.84, 0.89]
prec4p = [0.97, 0.96, 1, 0.91, 1, 0.97, 0.99, 0.92, 0.89, 0.87, 0.81, 0.95, 0.86, 0.52, 0.69]
rec1 = [0.52, 0.74, 0.87, 0.91, 1, 0.93, 0.97, 0.6, 0.4, 0.97, 0.78, 0.97, 0.07, 0.62, 0.74]
rec2 =[0.51, 0.65, 0.73, 0.9, 1, 0.95, 0.97, 0.75, 0.73, 0.68, 0.69, 1, 0.34, 0.97, 0.85]
rec3 = [0.86, 0.88, 1, 0.89, 1, 0.87, 0.84, 0.77, 0.79, 0.89, 0.71, 0.98, 0.07, 0.69, 0.83]
rec3p = [0.87, 0.93, 1, 0.93, 1, 0.98, 0.95, 0.87, 0.84, 0.95, 0.68, 0.99, 0.2, 0.52, 0.68]
rec4 = [0.9, 0.83, 1, 0.92, 1, 0.95, 0.89, 0.9, 0.94, 0.69, 0.8, 1, 0.62, 1, 1]
rec4p = [0.92, 0.86, 1, 0.92, 1, 0.99, 1, 0.91, 0.96, 0.7, 0.73, 1, 0.62, 0.87, 0.9]
opcja1 = [4.10, 5.13, 4.47, 4.43, 4.47]
opcja2 = [4.50, 3.97, 4.63, 4.23, 4.63]
opcja3 = [4.13, 3.90, 3.00, 4.23, 3.57]
opcja3p = [2.37, 3.00, 2.63, 3.40, 2.80]
opcja4 = [3.27, 2.80, 3.40, 2.57, 2.83]
opcja4p = [2.37, 2.20, 2.87, 2.13, 2.77]
sr1 = [0.85, 0.73, 0.79, 0.74, 0.74]
sr2 = [0.84, 0.82, 0.78, 0.78, 0.76]
sr3 = [0.87, 0.82, 0.87, 0.8, 0.82]
sr3p = [0.91, 0.83, 0.86, 0.83, 0.83]
sr4 = [0.9, 0.89, 0.9, 0.9, 0.88]
sr4p = [0.91, 0.89, 0.89, 0.89, 0.87]
#w, p = wilcoxon(sr4p, sr3p)
chi, p = friedmanchisquare(acc1, acc2, acc3, acc3p, acc4, acc4p)
print(chi, p)
alpha = 0.05
#if p > alpha:
    #print('Same distribution (fail to reject H0)')
#else:
   # print('Different distribution (reject H0)')
    
if p>0.1:
    print('H0 failed to reject')
elif p <= 0.01:
    print('H0 rejected at alpha = 0.01')
elif p <= 0.05:
    print('H0 rejected at alpha = 0.05')
else:
    print('H0 rejected at alpha = 0.1')