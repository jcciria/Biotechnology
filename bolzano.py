# Fichero de prueba, con funciones genéricas

def bolzano (f, interval, epsilon, **paramF):
    a,b= [x for x in interval]
    print (a,b)
    fa = f(a, **paramF)
    fb = f(b, **paramF)
    if fa * fb >=0:
        sol = np.nan
    else:
        while (b - a > epsilon):
            med = (a+b)/2
            if f(med, **paramF) * fa > 0:
                a = med
            else:
                b = med
        sol = (a+b)/2
    return sol
