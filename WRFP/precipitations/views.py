from django.shortcuts import render, render_to_response
from django.views.decorators.http import require_http_methods
from django.contrib.auth.admin import User
from django.contrib.auth.models import User, BaseUserManager
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.hashers import make_password
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.forms import AuthenticationForm
import datetime
import json
from precipitations.models import Investigator

def login_user(request):
    if request.method == 'POST':
        if 'username' in request.POST and request.POST['username'] and 'password' in request.POST and request.POST['password']:
            username = request.POST['username']
            password = request.POST['password']    

            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)
                return HttpResponseRedirect('/dashboard')
            else:
                return HttpResponseRedirect('/login')
    else:
        return HttpResponseRedirect('/login')

def logout_user(request):
    logout(request)
    return HttpResponseRedirect('/')

def index(request):
    return render(request, 'base.html')

def loggin(request):
    return render(request, 'login.html')

def dashboard(request):
    return render(request, 'dashboard.html')

def data(request):
    return render(request, 'data.html')

def rna(request):
    return render(request, 'rna.html')

def statistics(request):
    return render(request, 'statistics.html')

def reports(request):
    return render(request, 'reports.html')

""" Vistas de Configuracion """
def configeneral(request):    
    return render(request, 'config-general.html')

def configusers(request):
    i = Investigator.objects.all().exclude(pk = request.user.investigator.pk)
    return render(request, 'config-users.html', {'investigators':i})

def add_investigator(request):    
    if request.method == "POST" and (request.POST['inv_name'] and request.POST['inv_lastname'] and request.POST['inv_email'] and request.POST['inv_username'] and request.POST['inv_userpassword']):
        i_name = request.POST['inv_name']
        i_lastname = request.POST['inv_lastname'] 
        i_email = request.POST['inv_email']
        i_username = request.POST['inv_username'] 
        i_userpassword = request.POST['inv_userpassword']

        try:
            investigator = Investigator.objects.get(name = i_name)
        except Investigator.DoesNotExist:     
            usuario = authenticate(username = i_username, password = i_userpassword) 

            if usuario is None:
                new_user = User.objects.create_user(username = i_username, password = i_username)               
                inv = Investigator.objects.create(name = i_name, lastname = i_lastname, email = i_email, user = new_user)       
                inv.save()
                return JsonResponse({'result': 'pass'})
            else:
                return JsonResponse({'result': 'user exist'})
        return JsonResponse({'result': 'investigator exist'})
        """i = Investigator.objects.add_investigator(name, lastname, email, username, userpassword)
       
        if i is False:     
            return render(request, 'add-user.html', {'result': 'added'})
        else:
            return JsonResponse({'result': 'fail'})"""
    else:
        return JsonResponse({'result': 'null'})

def get_investigator(request):            
    inv = Investigator.objects.get(pk = request.GET['inv_pk'])   

    investigator = {
        'name':inv.FullName(),
    }             
    return JsonResponse({'investigator':investigator})

def del_investigator(request):
    return JsonResponse({'investigator':investigator})

def toggle_user_status(request):
    if request.is_ajax():
        userpk = request.GET['user-status']
        try:
            user = User.objects.get(pk = userpk)
        except User.DoesNotExist: 
            return JsonResponse({'status':"no-change"})

        if user.is_active:
            user.is_active = False            
        else:
            user.is_active = True
        user.save()

        return JsonResponse({'status':user.is_active})   

def config_data_sispi(request):
    return render(request, 'config-data-sispi.html')

def config_data_database(request):
    return render(request, 'config-database.html')