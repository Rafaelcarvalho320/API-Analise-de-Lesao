from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageUploadSerializer
from .processing import predict_symmetry, predict_border, predict_coloration
import os
import joblib
from tensorflow.keras.models import load_model

# Desativar GPU para TensorFlow (necessário para o Render)
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Impede o uso de GPU

# Carregar os modelos
color_model = joblib.load('/app/best_xgb_model_81.joblib')  # Modelo XGBoost
border_model = load_model('/app/resnet18_improved78.h5', compile=False)  # Modelo Keras sem recompilação

class ImageAnalysisView(APIView):
    permission_classes = [AllowAny]  # Permite acesso público
    parser_classes = (MultiPartParser, FormParser)  # Parsers para upload de arquivos
    serializer_class = ImageUploadSerializer  # Para renderizar o formulário no DRF

    def get(self, request, *args, **kwargs):
        # Resposta simples para GET (exibir mensagem ou formulário)
        return Response({"message": "Use POST para enviar uma imagem para análise."}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            image_path = self._save_image(image_file)  # Método renomeado com "_"

            try:
                # Realizar predições
                symmetry_resultado = predict_symmetry(image_path)
                border_resultado, border_prob_benigna, border_prob_maligna = predict_border(image_path, border_model)
                color_resultado, color_prob_benigna, color_prob_maligna = predict_coloration(image_path, color_model)

                # Remover a imagem temporária após o processamento
                if os.path.exists(image_path):
                    os.remove(image_path)

                # Preparar a resposta JSON
                response_data = {
                    'symmetry': symmetry_resultado,
                    'border': {
                        'resultado': border_resultado,
                        'prob_benigna': float(border_prob_benigna),  # Converte para tipo serializável
                        'prob_maligna': float(border_prob_maligna)
                    },
                    'coloration': {
                        'resultado': color_resultado,
                        'prob_benigna': float(color_prob_benigna),
                        'prob_maligna': float(color_prob_maligna)
                    }
                }
                return Response(response_data, status=status.HTTP_200_OK)

            except Exception as e:
                # Garantir que o arquivo temporário seja removido mesmo em caso de erro
                if os.path.exists(image_path):
                    os.remove(image_path)
                return Response({'error': f"Erro ao processar a imagem: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Retornar erros de validação do serializer
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def _save_image(self, image_file):
        """Salva a imagem temporariamente e retorna o caminho."""
        image_path = os.path.join('/tmp', f"{image_file.name}_{os.urandom(4).hex()}")  # Nome único
        os.makedirs('/tmp', exist_ok=True)  # Garante que o diretório existe
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        return image_path