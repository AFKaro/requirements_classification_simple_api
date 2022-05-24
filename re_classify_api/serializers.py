from rest_framework import serializers
from .models import Requirement


class RequirementSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Requirement
        fields = ('text', 'label')
        