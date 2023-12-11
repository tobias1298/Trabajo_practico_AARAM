import numpy as np
import matplotlib.pyplot as plt


def graficos():
    # grafico parte b) del ejercicio ekf

    # defino el vector para el eje X
    x_ekf = [0.125,0.25,0.5,2,4,8]

    # defino el vector para el eje Y
    y_b_ekf = [2.343146239870264,2.8380821742342985,3.243113944056096,4.744501139568933,7.2243924150362275,11.496087236170984]
    plt.plot(x_ekf, y_b_ekf)
    plt.xlabel('r')
    plt.ylabel('Error de posicion medio')
    plt.title('Gráfico del error de posición medio en función de r para el filtro de Kalman')
    plt.show()

    # grafico parte c)
    # error de posicion medio
    y_c_ekf_epm = [4.3561794286447375,3.669512902143256,3.2162947302028035,4.776433233502772,5.962007473852059,6.9308244192852895] 
    # ANEES
    y_c_ekf_ANEES = [0.8659920836116716,0.627483947929258,0.5081977400787687,0.43675417107236475,0.41598154704846113,0.39191476053353047]

    # grafico para el error de posicion medio
    plt.plot(x_ekf, y_c_ekf_epm)
    plt.xlabel('r')
    plt.ylabel('Error de posicion medio')
    plt.title('Gráfico del error de posición medio en función de r para el filtro de Kalman modificando el filter-factor')
    plt.show()

    # grafico para el ANEES
    plt.plot(x_ekf, y_c_ekf_ANEES)
    plt.xlabel('r')
    plt.ylabel('ANEES')
    plt.title('Gráfico del ANEES en función de r para el filtro de Kalman modificando el filter-factor')
    
    
    plt.show()

graficos()