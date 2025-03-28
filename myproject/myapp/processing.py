import joblib
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter

# Classe para análise de simetria
class LesionSymmetryAnalyzer:
    def __init__(self, epsilon=1.0, sigma=1.0, iterations=100, target_size=(256, 256)):
        self.epsilon = epsilon
        self.sigma = sigma
        self.iterations = iterations
        self.target_size = target_size
        
    def segment_and_analyze(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Imagem não encontrada.")
        image = cv2.resize(image, self.target_size)
        image_no_hair = self.remove_hair(image)
        gray = cv2.cvtColor(image_no_hair, cv2.COLOR_BGR2GRAY)
        segmentation = self.level_set_segmentation(gray)
        contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Nenhum contorno detectado.")
        contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            cx, cy = 128, 128
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        vertical_score, horizontal_score = self.calculate_symmetry_scores(mask, cx, cy)
        is_asymmetric = vertical_score > 0.3 or horizontal_score > 0.3
        return int(is_asymmetric), vertical_score, horizontal_score
    
    def remove_hair(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        kernel_dilate = np.ones((3, 3), np.uint8)
        hair_mask = cv2.dilate(thresh, kernel_dilate, iterations=1)
        result = cv2.inpaint(image, hair_mask, 5, cv2.INPAINT_TELEA)
        return result
        
    def level_set_segmentation(self, image):
        _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        phi = -self.epsilon * (0.5 - binary)
        smoothed = gaussian_filter(image, self.sigma)
        gx, gy = np.gradient(smoothed)
        g = 1.0 / (1.0 + np.sqrt(gx**2 + gy**2)**2)
        dt = 0.1
        for i in range(self.iterations):
            dx, dy = np.gradient(phi)
            grad_mag = np.sqrt(dx**2 + dy**2)
            nx = dx / (grad_mag + 1e-10)
            ny = dy / (grad_mag + 1e-10)
            curvature = np.gradient(nx)[0] + np.gradient(ny)[1]
            dphi = g * (curvature + 1.0)
            phi = phi + dt * dphi
            if i % 20 == 0:
                phi = np.sign(phi) * self.epsilon
        return (phi <= 0).astype(np.uint8) * 255
    
    def calculate_symmetry_scores(self, mask, cx, cy):
        vertical = mask[:, :cx]
        vertical_flipped = cv2.flip(mask[:, cx:], 1)
        min_width = min(vertical.shape[1], vertical_flipped.shape[1])
        vertical = vertical[:, -min_width:] if vertical.shape[1] > min_width else vertical
        vertical_flipped = vertical_flipped[:, :min_width] if vertical_flipped.shape[1] > min_width else vertical_flipped
        horizontal = mask[:cy, :]
        horizontal_flipped = cv2.flip(mask[cy:, :], 0)
        min_height = min(horizontal.shape[0], horizontal_flipped.shape[0])
        horizontal = horizontal[-min_height:, :] if horizontal.shape[0] > min_height else horizontal
        horizontal_flipped = horizontal_flipped[:min_height, :] if horizontal_flipped.shape[0] > min_height else horizontal_flipped
        vertical_diff = cv2.absdiff(vertical, vertical_flipped)
        horizontal_diff = cv2.absdiff(horizontal, horizontal_flipped)
        total_area = np.sum(mask) + 1e-5
        vertical_score = np.sum(vertical_diff) / total_area
        horizontal_score = np.sum(horizontal_diff) / total_area
        return vertical_score, horizontal_score

# Classe para análise de bordas
class LesionProcessor:
    def __init__(self, epsilon=1.0, sigma=1.0, iterations=100, target_size=(256, 256)):
        self.epsilon = epsilon
        self.sigma = sigma
        self.iterations = iterations
        self.target_size = target_size

    def remove_hair(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
        result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
        return result

    def level_set_segmentation(self, image):
        _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        phi = -self.epsilon * (0.5 - binary)
        smoothed = gaussian_filter(image, self.sigma)
        gx, gy = np.gradient(smoothed)
        g = 1.0 / (1.0 + np.sqrt(gx**2 + gy**2)**2)
        dt = 0.1
        for i in range(self.iterations):
            dx, dy = np.gradient(phi)
            grad_mag = np.sqrt(dx**2 + dy**2)
            nx = dx / (grad_mag + 1e-10)
            ny = dy / (grad_mag + 1e-10)
            curvature = np.gradient(nx)[0] + np.gradient(ny)[1]
            dphi = g * (curvature + 1.0)
            phi = phi + dt * dphi
            if i % 20 == 0:
                phi = np.sign(phi) * self.epsilon
        return (phi <= 0).astype(np.uint8) * 255

    def find_most_central_contour(self, segmentation):
        contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h, w = segmentation.shape
        center = (w // 2, h // 2)
        min_distance = float('inf')
        most_central_contour = None
        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                distance = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    most_central_contour = contour
        return most_central_contour

    def calculate_circularity(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        return circularity

    def calculate_solidity(self, contour):
        area = cv2.contourArea(contour)
        convex_hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(convex_hull)
        solidity = area / hull_area if hull_area > 0 else 0
        return solidity

    def calculate_fractal_dimension(self, contour, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, 1)
        box_sizes = np.logspace(1, 3, num=10, base=2, dtype=int)
        counts = []
        for size in box_sizes:
            grid = np.zeros((int(np.ceil(image_shape[0] / size)), int(np.ceil(image_shape[1] / size))), dtype=bool)
            for i in range(0, image_shape[0], size):
                for j in range(0, image_shape[1], size):
                    if np.any(mask[i:i+size, j:j+size]):
                        grid[i//size, j//size] = True
            counts.append(np.sum(grid))
        coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
        fractal_dimension = -coeffs[0]
        return fractal_dimension

    def process_image_for_prediction(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Imagem não encontrada: {image_path}")
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = (image_rgb - np.mean(image_rgb)) / np.std(image_rgb)
        image_for_metrics = cv2.resize(image, self.target_size)
        image_no_hair = self.remove_hair(image_for_metrics)
        gray = cv2.cvtColor(image_no_hair, cv2.COLOR_BGR2GRAY)
        segmentation = self.level_set_segmentation(gray)
        central_contour = self.find_most_central_contour(segmentation)
        if central_contour is None:
            raise ValueError("Nenhum contorno central encontrado.")
        circularity = self.calculate_circularity(central_contour)
        solidity = self.calculate_solidity(central_contour)
        fractal_dimension = self.calculate_fractal_dimension(central_contour, gray.shape)
        return image_normalized, [circularity, solidity, fractal_dimension]

# Classe para análise de coloração
class AnalisadorCor:
    def __init__(self, niveis=256, distancia=1):
        self.niveis = niveis
        self.distancia = distancia
        self.angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def preprocessar_imagem(self, imagem):
        imagem_sem_pelos = self.remover_pelos(imagem)
        if len(imagem_sem_pelos.shape) == 3:
            cinza = cv2.cvtColor(imagem_sem_pelos, cv2.COLOR_BGR2GRAY)
        else:
            cinza = imagem_sem_pelos
        normalizada = ((cinza / cinza.max()) * (self.niveis - 1)).astype(np.uint8)
        return imagem_sem_pelos, normalizada
    
    def remover_pelos(self, imagem):
        gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        imagem_sem_pelos = cv2.inpaint(imagem, hair_mask, 10, cv2.INPAINT_TELEA)
        return imagem_sem_pelos
    
    def computar_glcm(self, imagem):
        distancias = [self.distancia]
        glcm = graycomatrix(imagem, 
                           distances=distancias,
                           angles=self.angulos,
                           levels=self.niveis,
                           symmetric=True,
                           normed=True)
        return glcm
    
    def extrair_caracteristicas(self, imagem):
        imagem_sem_pelos, processada = self.preprocessar_imagem(imagem)
        gray_image = cv2.cvtColor(imagem_sem_pelos, cv2.COLOR_BGR2GRAY)
        _, hair_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
        valid_mask = cv2.bitwise_not(hair_mask)
        masked_image = cv2.bitwise_and(processada, processada, mask=valid_mask)
        glcm = self.computar_glcm(masked_image)
        caracteristicas = np.array([
            graycoprops(glcm, 'correlation').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'ASM').mean(),
            graycoprops(glcm, 'contrast').mean()
        ])
        return caracteristicas

# Funções de predição
def load_and_preprocess_image_for_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_symmetry(image_path):
    analyzer = LesionSymmetryAnalyzer(epsilon=1.0, sigma=1.0, iterations=100)
    is_asymmetric, _, _ = analyzer.segment_and_analyze(image_path)
    resultado = "Lesão Simétrica" if is_asymmetric == 0 else "Lesão Assimétrica"
    return resultado

def predict_border(image_path, model):
    processor = LesionProcessor(epsilon=1.0, sigma=1.0, iterations=100, target_size=(256, 256))
    processed_image, additional_features = processor.process_image_for_prediction(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)
    additional_features = np.array(additional_features).reshape(1, -1)
    prediction = model.predict([processed_image, additional_features])
    class_pred = (prediction > 0.5).astype(int)[0][0]
    probability = prediction[0][0]
    resultado = "Borda com características Benignas" if class_pred == 0 else "Borda com características Malignas"
    prob_benigna = (1 - probability) * 100
    prob_maligna = probability * 100
    return resultado, prob_benigna, prob_maligna

def predict_coloration(image_path, model):
    new_image = load_and_preprocess_image_for_color(image_path)
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    new_image_preprocessed = preprocess_input(new_image)
    new_image_features = base_model.predict(new_image_preprocessed.reshape(1, 224, 224, 3)).flatten()
    analisador = AnalisadorCor()
    new_csv_features = analisador.extrair_caracteristicas(new_image)
    new_combined_features = np.hstack((new_image_features, new_csv_features))
    prediction = model.predict(new_combined_features.reshape(1, -1))
    probability = model.predict_proba(new_combined_features.reshape(1, -1))
    resultado = "Coloração com características benignas" if prediction[0] == 0 else "Coloração com características malignas"
    prob_benigna = probability[0][0] * 100
    prob_maligna = probability[0][1] * 100
    return resultado, prob_benigna, prob_maligna