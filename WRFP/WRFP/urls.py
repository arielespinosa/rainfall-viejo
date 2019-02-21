"""WRFP URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from precipitations import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('', include('precipitations.urls', namespace='precipitations')),	
    path('', views.index, name="index"),
    path('login/', views.loggin, name="login"),
    path('loginuser/', views.login_user, name="login_user"),    
    path('logout/', views.logout_user, name="logout_user"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('data/', views.data, name="data"),
    path('rna/', views.rna, name="rna"),
    path('statistics/', views.statistics, name="statistics"),
    path('reports/', views.reports, name="reports"),
    path('configuration/', views.configeneral, name="config_general"),
    path('configuration/users', views.configusers, name="config_users"),
    path('configuration/data/sispi', views.config_data_sispi, name="config_sispi"),
    path('configuration/users/get_investigator', views.get_investigator, name="get_investigator"),
    path('configuration/users/add_investigator', views.add_investigator, name="add_investigator"),
    path('configuration/users/toggle_status', views.toggle_user_status, name="user_status"),

    # Database url configuration
    path('configuration/data/database', views.config_data_database, name="config_database"),
]
