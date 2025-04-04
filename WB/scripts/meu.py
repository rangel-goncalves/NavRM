from controller import Robot, Supervisor
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import signal

# =============================================
# CONSTANTES GLOBAIS
# =============================================

# Deslocamento do sensor LiDAR em relação ao centro do robô (em metros)
DESLOCAMENTO_LIDAR_X = 0.202  
DESLOCAMENTO_LIDAR_Y = 0.0  

# Dimensões do display para visualização
LARGURA_DISPLAY = 450
ALTURA_DISPLAY = 567

# Parâmetros do sensor LiDAR
ALCANCE_MAXIMO_LIDAR = 5.0  # Alcance máximo em metros
INDICE_MINIMO_LIDAR = 80    # Primeiras 80 leituras devem ser ignoradas
INDICE_MAXIMO_LIDAR = -80   # Últimas 80 leituras devem ser ignoradas

# =============================================
# FUNÇÕES AUXILIARES
# =============================================

def converter_graus_para_radianos(graus):
    """Converte ângulo de graus para radianos"""
    return graus * (math.pi / 180)

def mapear_mundo_para_tela(x_mundo, y_mundo, largura_mapa=400, altura_mapa=504, 
                          x_min_mundo=-2.25, x_max_mundo=2.25, 
                          y_min_mundo=-3.92, y_max_mundo=1.75):
    """
    Converte coordenadas do mundo real para coordenadas de tela/pixels
    
    Args:
        x_mundo, y_mundo: Posição no mundo real
        outros parâmetros definem os limites do mundo e dimensões do mapa
    
    Returns:
        Coordenadas (x,y) em pixels
    """
    escala_x = largura_mapa / (x_max_mundo - x_min_mundo)
    escala_y = altura_mapa / (y_max_mundo - y_min_mundo)
    
    px = int((x_mundo - x_min_mundo) * escala_x)
    py = int((y_mundo - y_min_mundo) * escala_y)
    py = altura_mapa - 1 - py  # Inverte eixo Y
    
    return max(0, min(largura_mapa - 1, px)), max(0, min(altura_mapa - 1, py))

def mapear_mundo_para_tela_vetorial(x_mundo, y_mundo, largura_mapa=400, altura_mapa=504, 
                                   x_min_mundo=-2.25, x_max_mundo=2.25, 
                                   y_min_mundo=-3.92, y_max_mundo=1.75):
    """
    Versão vetorizada da função de mapeamento para lidar com arrays numpy
    """
    escala_x = largura_mapa / (x_max_mundo - x_min_mundo)
    escala_y = altura_mapa / (y_max_mundo - y_min_mundo)
    
    px = ((x_mundo - x_min_mundo) * escala_x).astype(int)
    py = ((y_mundo - y_min_mundo) * escala_y).astype(int)
    py = altura_mapa - 1 - py
    
    px = np.clip(px, 0, largura_mapa - 1)
    py = np.clip(py, 0, altura_mapa - 1)

    return px, py

def convolucao_fft_2d(imagem, kernel):
    """
    Realiza convolução 2D usando FFT para maior eficiência
    
    Args:
        imagem: Matriz representando o mapa
        kernel: Kernel de convolução
        
    Returns:
        Matriz resultante da convolução
    """
    # Ajusta tamanhos com padding
    pad_x = imagem.shape[0] + kernel.shape[0] - 1
    pad_y = imagem.shape[1] + kernel.shape[1] - 1
    
    imagem_padded = np.pad(imagem, ((0, pad_x - imagem.shape[0]), (0, pad_y - imagem.shape[1])), mode='constant')
    kernel_padded = np.pad(kernel, ((0, pad_x - kernel.shape[0]), (0, pad_y - kernel.shape[1])), mode='constant')

    # Convolução no domínio da frequência
    fft_imagem = np.fft.fft2(imagem_padded)
    fft_kernel = np.fft.fft2(kernel_padded)
    resultado_fft = fft_imagem * fft_kernel
    resultado = np.fft.ifft2(resultado_fft)
    
    return np.real(resultado)

def calcular_menor_angulo(angulo_alvo, angulo_atual):
    """
    Calcula o menor ângulo para virar entre duas orientações
    
    Args:
        angulo_alvo: Ângulo desejado (radianos)
        angulo_atual: Ângulo atual (radianos)
        
    Returns:
        Diferença angular no intervalo [-π, π]
    """
    diferenca = angulo_alvo - angulo_atual
    return (diferenca + np.pi) % (2 * np.pi) - np.pi

# =============================================
# CONFIGURAÇÃO INICIAL DO ROBÔ
# =============================================

# Cria instância do robô e supervisor
robo = Supervisor()

# Configura timestep da simulação
INTERVALO_TEMPO = int(robo.getBasicTimeStep())

# Parâmetros físicos do robô
VELOCIDADE_MAXIMA = 6.28  # Velocidade máxima das rodas (rad/s)
RAIO_RODA = 0.075         # Raio das rodas (metros)
DISTANCIA_ENTRE_RODAS = 0.4  # Distância entre as rodas (metros)

# Variáveis de estado do robô
posicao_x = 0
posicao_y = 0.028  # Posição inicial
orientacao = 1.57  # Orientação inicial (radianos)

# Inicializa motores
motor_esquerdo = robo.getDevice("wheel_left_joint")
motor_direito = robo.getDevice("wheel_right_joint")
motor_esquerdo.setPosition(float('inf'))
motor_direito.setPosition(float('inf'))
motor_esquerdo.setVelocity(0.0)
motor_direito.setVelocity(0.0)

# =============================================
# CONFIGURAÇÃO DE SENSORES
# =============================================

# Sensor LiDAR
lidar = robo.getDevice("Hokuyo URG-04LX-UG01") 
lidar.enable(INTERVALO_TEMPO)
lidar.enablePointCloud()
RESOLUCAO_LIDAR = 0.36  # graus
CAMPO_VISAO_LIDAR = 240  # graus

# GPS e bússola para localização
gps = robo.getDevice("gps")
gps.enable(INTERVALO_TEMPO)
bussola = robo.getDevice("compass")
bussola.enable(INTERVALO_TEMPO)

# =============================================
# PONTOS DE REFERÊNCIA (WAYPOINTS)
# =============================================

pontos_referencia = [
    (0, 0), (0.47, -0.256),  
    (0.53, -3),
    (-1.67, -3), (-1.7, -1.38), 
    (-1.72, -0.574), (-1.28, 0.247)
]

indice_ponto_atual = 0
movimento_para_frente = 1  # Flag para controle de direção

# =============================================
# CONFIGURAÇÃO DE VISUALIZAÇÃO
# =============================================

display = robo.getDevice('display')
display.setColor(0xFF0000)  # Vermelho para a trajetória

# Inicializa mapa de probabilidades
mapa_probabilidades = np.zeros((450, 567))
tamanho_kernel = 55
kernel = np.ones((tamanho_kernel, tamanho_kernel))

# =============================================
# FUNÇÃO PRINCIPAL DE CONTROLE
# =============================================

def atualizar_pose_robo():
    """Atualiza a posição e orientação do robô com dados dos sensores"""
    global posicao_x, posicao_y, orientacao
    posicao_x = gps.getValues()[0]
    posicao_y = gps.getValues()[1]
    orientacao = np.arctan2(bussola.getValues()[0], bussola.getValues()[1])

while robo.step(INTERVALO_TEMPO) != -1:
    # Atualiza posição/orientação
    atualizar_pose_robo()
    
    # Calcula distância para o ponto atual
    distancia = np.sqrt((posicao_x - pontos_referencia[indice_ponto_atual][0])**2 + 
                       (posicao_y - pontos_referencia[indice_ponto_atual][1])**2)

    # Calcula ângulo desejado
    angulo_desejado = np.arctan2(pontos_referencia[indice_ponto_atual][1] - posicao_y, 
                                pontos_referencia[indice_ponto_atual][0] - posicao_x)

    # Calcula ângulo de virada mínimo
    angulo_virada = calcular_menor_angulo(angulo_desejado, orientacao)

    # Lógica para mudar de ponto de referência
    if distancia < 0.1 and indice_ponto_atual < len(pontos_referencia) and indice_ponto_atual >= 0:
        indice_anterior = indice_ponto_atual
        if movimento_para_frente == 1:
            indice_ponto_atual += 1
        else:
            indice_ponto_atual -= 1
            if indice_ponto_atual == 0:
                break
        if indice_ponto_atual >= len(pontos_referencia) - 1:
            movimento_para_frente = 0
        print(f"Ponto {indice_anterior} alcançado, indo para ponto {indice_ponto_atual}")

    # Controle dos motores (leis de controle)
    ganho_angular = 8.0
    ganho_linear = 3.0
    velocidade_esquerda = -angulo_virada * ganho_angular + distancia * ganho_linear
    velocidade_direita = angulo_virada * ganho_angular + distancia * ganho_linear
    
    # Aplica velocidades limitadas
    motor_esquerdo.setVelocity(max(min(velocidade_esquerda, VELOCIDADE_MAXIMA), -VELOCIDADE_MAXIMA))
    motor_direito.setVelocity(max(min(velocidade_direita, VELOCIDADE_MAXIMA), -VELOCIDADE_MAXIMA))

    # =============================================
    # DESENHO DA TRAJETÓRIA NO DISPLAY
    # =============================================
    
    # Converte posição atual para coordenadas de tela
    px_trajetoria, py_trajetoria = mapear_mundo_para_tela(posicao_x, posicao_y)
    
    # Desenha ponto vermelho na trajetória
    display.setColor(0xFF0000)  # Vermelho
    display.drawPixel(px_trajetoria, py_trajetoria)

    # =============================================
    # PROCESSAMENTO DO LIDAR E MAPEAMENTO
    # =============================================
    
    # Obtém dados do LiDAR
    distancias_lidar = lidar.getRangeImage()
    distancias_lidar = distancias_lidar[INDICE_MINIMO_LIDAR:INDICE_MAXIMO_LIDAR]
    distancias_lidar = np.where(np.isinf(distancias_lidar), 100, distancias_lidar)
    
    # Calcula ângulos correspondentes às leituras
    angulos_lidar = np.linspace(converter_graus_para_radianos(120), 
                               converter_graus_para_radianos(-120), 667)
    angulos_lidar = angulos_lidar[INDICE_MINIMO_LIDAR:INDICE_MAXIMO_LIDAR]
   
    # Transforma coordenadas polares para cartesianas
    x_robo = distancias_lidar * np.cos(angulos_lidar)
    y_robo = distancias_lidar * np.sin(angulos_lidar)

    # Aplica deslocamento do sensor
    x_robo += DESLOCAMENTO_LIDAR_X
    y_robo += DESLOCAMENTO_LIDAR_Y

    # Transforma para coordenadas globais
    cos_orient = np.cos(orientacao)
    sin_orient = np.sin(orientacao)
    x_mundo = x_robo * cos_orient - y_robo * sin_orient + posicao_x
    y_mundo = x_robo * sin_orient + y_robo * cos_orient + posicao_y

    # Converte para coordenadas de tela
    px, py = mapear_mundo_para_tela_vetorial(x_mundo, y_mundo)

    # Atualiza mapa de probabilidades
    mapa_probabilidades[px, py] = np.minimum(1, mapa_probabilidades[px, py] + 0.01)

    # Converte probabilidades para cores (escala de cinza)
    intensidade = (mapa_probabilidades[px, py] * 255).astype(int)
    cor = (intensidade * 256**2 + intensidade * 256 + intensidade).astype(int)

    # Desenha pontos do LiDAR no display
    for i in range(len(px)):
        display.setColor(int(cor[i]))
        display.drawPixel(int(px[i]), int(py[i]))

# =============================================
# PÓS-PROCESSAMENTO (ESPAÇO DE CONFIGURAÇÃO)
# =============================================

# Calcula espaço de configuração com convolução
mapa_configuracao = convolucao_fft_2d(mapa_probabilidades, kernel)
espaco_configuracao = mapa_configuracao > 0.9  # Limiarização

# Visualização final
plt.clf()
plt.imshow(espaco_configuracao, cmap='gray')
plt.title("Espaço de Configuração")
plt.pause(1000)