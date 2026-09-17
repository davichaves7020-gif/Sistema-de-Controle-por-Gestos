import cv2
import pyautogui
import numpy as np
import time
import os

# Importações oficiais da Nova API do MediaPipe Tasks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configurações de segurança do PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# Verifica se os modelos obrigatórios existem na pasta local
if not os.path.exists("hand_landmarker.task") or not os.path.exists("face_landmarker.task"):
    print("\n[ERRO]: Arquivos de modelo '.task' nao encontrados na pasta atual.")
    print("Baixe os arquivos 'hand_landmarker.task' e 'face_landmarker.task' antes de rodar.")
    exit()

class ControladorNovaAPI:
    def __init__(self):
        # Configurações de Mapeamento de Tela
        self.largura_tela, self.altura_tela = pyautogui.size()
        
        # Filtro de Média Móvel para Suavização do Mouse
        self.historico_x = []
        self.historico_y = []
        self.suavizacao = 7

        # Índices dos pontos de referência da Nova API (Mãos e Olhos)
        self.DEDO_INDICADOR = 8
        self.DEDO_POLEGAR = 4
        
        # Pontos do Olho Direito para cálculo do EAR matemático
        self.PALPEBRA_SUP_DIREITA = 159
        self.PALPEBRA_INF_DIREITA = 145
        self.CANTO_ESQ_DIREITO = 33
        self.CANTO_DIR_DIREITO = 133

        # Controle de tempo (Cooldown rápido para resposta imediata)
        self.ultimo_clique_piscada = 0
        self.cooldown_piscada = 0.25

        # Estado global dos detectores (Threads)
        self.ultimos_resultados_mao = None
        self.ultimos_resultados_rosto = None

        self.inicializar_detectores()

    def inicializar_detectores(self):
        """Inicializa os Landmarkers usando o modo LIVE_STREAM obrigatório para webcams."""
        def callback_mao(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
            self.ultimos_resultados_mao = result

        def callback_rosto(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # pyright: ignore[reportInvalidTypeForm]
            self.ultimos_resultados_rosto = result

        # Configuração de Mão
        opcoes_mao = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=callback_mao
        )
        self.detector_mao = vision.HandLandmarker.create_from_options(opcoes_mao)

        # Configuração de Rosto
        opcoes_rosto = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=callback_rosto
        )
        self.detector_rosto = vision.FaceLandmarker.create_from_options(opcoes_rosto)

    def processar_mouse(self, img_w, img_h):
        """Move o mouse e detecta o clique ao dobrar o indicador (Finger Click)."""
        if not self.ultimos_resultados_mao or not self.ultimos_resultados_mao.hand_landmarks:
            return False, None
        
        landmarks_lista = self.ultimos_resultados_mao.hand_landmarks
        if len(landmarks_lista) == 0:
            return False, None
            
        landmarks = landmarks_lista[0] # Acessa a primeira mão detectada
        
        p_indicador_ponta = landmarks[self.DEDO_INDICADOR] # Ponto 8 (Ponta)
        p_indicador_meio = landmarks[6]                   # Ponto 6 (Articulação do meio)

        # 1. Movimentação do Cursor (Mapeamento com margem adaptativa)
        margem = 0.15
        na_tela_x = np.interp(p_indicador_ponta.x, [margem, 1.0 - margem], [0, self.largura_tela])
        na_tela_y = np.interp(p_indicador_ponta.y, [margem, 1.0 - margem], [0, self.altura_tela])

        # Filtro de Média Móvel para Suavização
        self.historico_x.append(na_tela_x)
        self.historico_y.append(na_tela_y)
        if len(self.historico_x) > self.suavizacao:
            self.historico_x.pop(0)
            self.historico_y.pop(0)
        
        mouse_x = int(np.mean(self.historico_x))
        mouse_y = int(np.mean(self.historico_y))
        pyautogui.moveTo(mouse_x, mouse_y)

        # Coordenadas em pixels para retorno visual
        x1, y1 = int(p_indicador_ponta.x * img_w), int(p_indicador_ponta.y * img_h)

        # 2. Nova Lógica de Clique: Se a ponta do indicador ficar abaixo da articulação do meio
        if p_indicador_ponta.y > p_indicador_meio.y:
            pyautogui.click()
            return True, (x1, y1)
        
        return False, (x1, y1)

    def processar_piscada(self, frame, img_w, img_h):
        """Detecta a proporção geométrica (EAR) da piscada do olho direito."""
        if not self.ultimos_resultados_rosto or not self.ultimos_resultados_rosto.face_landmarks:
            return False

        rosto_lista = self.ultimos_resultados_rosto.face_landmarks
        if len(rosto_lista) == 0:
            return False
            
        landmarks = rosto_lista[0] # Acessa os pontos do primeiro rosto detectado

        p_sup = landmarks[self.PALPEBRA_SUP_DIREITA]
        p_inf = landmarks[self.PALPEBRA_INF_DIREITA]
        p_esq = landmarks[self.CANTO_ESQ_DIREITO]
        p_dir = landmarks[self.CANTO_DIR_DIREITO]

        y_sup = p_sup.y * img_h
        y_inf = p_inf.y * img_h
        x_esq = p_esq.x * img_w
        x_dir = p_dir.x * img_w

        # Desenha os pontos do olho na tela para feedback visual na feira (Ciano)
        cv2.circle(frame, (int(x_esq), int(y_sup)), 2, (255, 255, 0), -1)
        cv2.circle(frame, (int(x_dir), int(y_inf)), 2, (255, 255, 0), -1)

        altura_olho = y_inf - y_sup
        largura_olho = np.abs(x_dir - x_esq)

        if largura_olho == 0:
            return False

        # Proporção de Abertura do Olho (EAR)
        ear = altura_olho / largura_olho
        tempo_atual = time.time()
        
        # Limiar adaptado para 0.22 (piscada confortável e veloz)
        if ear < 0.22 and (tempo_atual - self.ultimo_clique_piscada) > self.cooldown_piscada:
            self.ultimo_clique_piscada = tempo_atual
            pyautogui.press('space') 
            return True
            
        return False

    def iniciar(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n=== SISTEMA TOTALMENTE CORRIGIDO INICIADO ===")
        print("-> Aponte o DEDO INDICADOR na camera para guiar o mouse.")
        print("-> DOBRE O INDICADOR rapidamente para dar CLIQUE ESQUERDO.")
        print("-> PISQUE O OLHO DIREITO para pressionar ESPAÇO.")
        print("-> Pressione 'q' na janela gráfica para fechar o software.")
        print("================================================\n")

# Criar uma variável de controle no seu __init__ (adicione lá se quiser, ou use direto aqui)
        # self.sistema_pausado = False (coloque essa linha dentro do seu __init__)

        # Adicione esta linha logo ANTES do loop while do método iniciar() se não quiser mexer no init:
        sistema_pausado = False

        while cap.isOpened():
            sucesso, frame = cap.read()
            if not sucesso:
                break

            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp = int(time.time() * 1000)
            self.detector_mao.detect_async(mp_image, timestamp)
            self.detector_rosto.detect_async(mp_image, timestamp)

            # Verifica comandos de teclado para pausa ou fechamento
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q'):
                break
            elif tecla == ord('p'): # Tecla P inverte o estado de pausa do mouse/piscada
                sistema_pausado = not sistema_pausado
                print(f"=== SISTEMA {'PAUSADO' if sistema_pausado else 'RETOMADO'} ===")

            if sistema_pausado:
                cv2.putText(frame, "SISTEMA PAUSADO [P para Retomar]", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Só processa se NÃO estiver pausado
                clicou, pos_dedo = self.processar_mouse(img_w, img_h)
                if pos_dedo:
                    cor = (0, 0, 255) if clicou else (0, 255, 0)
                    cv2.circle(frame, pos_dedo, 10, cor, -1)

                if self.processar_piscada(frame, img_w, img_h):
                    cv2.putText(frame, "PISCADA DETECTADA (ESPACO)", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Nova API MediaPipe Tasks - Engenharia de Software", frame)
if __name__ == "__main__":
    app = ControladorNovaAPI()
    app.iniciar()