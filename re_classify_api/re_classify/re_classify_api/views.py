from re_classify_api.infra.ioc.injector import DependencyInjector
from .serializers import RequirementSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Requirement
from rest_framework import status


classification_app = DependencyInjector().classification_app

class RequirementViewSet(APIView):
    """
    List all requirements, or create a new requirement.
    """
    def get(self, request, format=None):
        snippets = Requirement.objects.all()
        serializer = RequirementSerializer(snippets, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        response = classification_app.run(requirement=request.data)
        request.data['label'] = response
        print('RESULT', response)
        
        serializer = RequirementSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
