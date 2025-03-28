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

# Carregar os modelos
color_model = joblib.load('/app/best_xgb_model_81.joblib')
border_model = load_model('/app/resnet18_improved78.h5')

class ImageAnalysisView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]
    serializer_class = ImageUploadSerializer  # Mantemos para o DRF renderizar o formulário

    def get(self, request, *args, **kwargs):
        # Retorna uma mensagem para exibir o formulário
        return Response({"message": "Use o formulário abaixo para enviar uma imagem."})

    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)  # Instancie manualmente
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            image_path = self.save_image(image_file)
            
            try:
                # Realizar predições
                symmetry_resultado = predict_symmetry(image_path)
                border_resultado, border_prob_benigna, border_prob_maligna = predict_border(image_path, border_model)
                color_resultado, color_prob_benigna, color_prob_maligna = predict_coloration(image_path, color_model)
                
                # Remover a imagem temporária
                os.remove(image_path)
                
                # Preparar a resposta
                response_data = {
                    'symmetry': symmetry_resultado,
                    'border': {
                        'resultado': border_resultado,
                        'prob_benigna': border_prob_benigna,
                        'prob_maligna': border_prob_maligna
                    },
                    'coloration': {
                        'resultado': color_resultado,
                        'prob_benigna': color_prob_benigna,
                        'prob_maligna': color_prob_maligna
                    }
                }
                return Response(response_data, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def save_image(self, image_file):
        image_path = os.path.join('/tmp', image_file.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        return image_path