from django.shortcuts import render

def index(request):
    context = { }
    return render(request, 'bindsim_client/index.html', context)
