from django.db import models

class Requirement(models.Model):
    text = models.CharField(max_length=5000)
    label = models.CharField(max_length=2)    
    
    def __str__(self):
        return {'text': self.text, 'label': self.label}
