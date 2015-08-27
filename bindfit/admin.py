from django.contrib import admin
from .models import Fit, Data, Result
admin.site.register(Data)
admin.site.register(Fit)
admin.site.register(Result)
