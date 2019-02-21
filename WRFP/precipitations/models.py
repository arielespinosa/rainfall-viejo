from django.db import models
from django.contrib.auth.models import User

class ManagerInvestigator(models.Manager):
    def add_investigator(self, invs_name, invs_lastname, invs_email, invs_user_name, invs_user_password):
        
        investigator = Investigator.objects.get(email = invs_email)
        
        if investigator is None:
            usuario = authenticate(username = invs_user_name, password = invs_user_password) 

            if usuario is None:
                new_user = User.objects.create_user(username = invs_user_name, password = invs_user_password)               
                inv = Investigator.objects.create(name = invs_name, lastname = invs_lastname, email = invs_email, user = new_user)
                inv.save()
            else:
                return  False
        else:
            return False

class Investigator(models.Model):
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.CASCADE)
    name = models.CharField(max_length = 30)    
    lastname = models.CharField(max_length = 30)
    email = models.EmailField()
    objects = ManagerInvestigator()

    def __str__(self):
        return self.name

    def FullName(self):
        return self.name + ' ' + self.lastname

class Place(models.Model):
    name = models.CharField(max_length = 30)
    latitud = models.FloatField()
    longitud = models.FloatField()
    
    def __str__(self):
        return self.name

class Notification(models.Model):
    title = models.CharField(max_length = 30)
    icon = models.CharField(max_length = 30)
    link = models.CharField(max_length = 30)
    
    def __str__(self):
        return self.title