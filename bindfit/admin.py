from django.contrib import admin
from .models import Fit, Data

class DataAdmin(admin.ModelAdmin):
    pass

class FitAdmin(admin.ModelAdmin):
    search_fields = ["meta_email", "meta_author", "meta_name", "meta_host", "meta_guest"]

admin.site.register(Data, DataAdmin)
admin.site.register(Fit,  FitAdmin)
