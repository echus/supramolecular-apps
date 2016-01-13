from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import simulators

@api_view(['POST'])
def nmr_1to1(request):
    return sim_1to1("nmr_1to1", request)

@api_view(['POST'])
def nmr_1to2(request):
    return sim_1to2("nmr_1to2", request)

@api_view(['POST'])
def uv_1to1(request):
    return sim_1to1("uv_1to1", request)

@api_view(['POST'])
def uv_1to2(request):
    return sim_1to2("uv_1to2", request)

@api_view(['POST'])
def nmr_2to1(request):
    return sim_2to1("nmr_2to1", request)

@api_view(['POST'])
def uv_2to1(request):
    return sim_2to1("uv_2to1", request)

#
# Generalised response builders
#
def sim_1to1(simulator, request):
    """
    Runs requested simulation and returns simulated isotherm and molefractions
    as a function of equivalent [G]0/[H]0 fraction.

    Request:
        k1        : float  Binding constant Ka
        h0_init   : float  Initial [H]0
        g0h0_final: float  Max equiv. [G]0/[H]0
        dh        : float  Free host NMR resonance
        dhg       : float  Host-Guest complex NMR resonance

    Response:
        mf_h : array of [x, y] points for plotting
        mf_hg: array of [x, y] points for plotting
        dd   : array of [x, y] points for plotting
    """

    # Convert any numerical values in query dict to python int/floats
    data_parsed = { k:str_to_num(v) for k, v in request.data.items() }

    # Run simulator
    g0h0, dd, mf_h, mf_hg = getattr(simulators, simulator)(**data_parsed)

    # Build response dict
    response = {
        "mf_h" : [[x, y] for x, y in zip(g0h0, mf_h)],
        "mf_hg": [[x, y] for x, y in zip(g0h0, mf_hg)],
        "dd"   : [[x, y] for x, y in zip(g0h0, dd)],
        }

    return Response(response)

def sim_1to2(simulator, request):
    """
    Runs requested simulation and returns simulated isotherm and molefractions
    as a function of equivalent [G]0/[H]0 fraction.

    Query:
        k1         : float,  Binding constant K1
        k2         : float,  Binding constant K2
        h0_init    : float,  Initial [H]0
        g0h0_final : float,  Max equiv. [G]0/[H]0
        dh         : float,  Free host NMR resonance
        dhg        : float,  Host-Guest complex NMR resonance
        dhg2       : float  Host-Guest2 complex NMR resonance

    Response:
        (all objects are arrays of [x, y] points for plotting
         with equivalent [G]0/[H]0 on x axis)

        dd     : Isotherm
        mf_h   : Host molefraction
        mf_hg  : HG complex molefraction
        mf_hg2 : HG2 complex molefraction
    """

    # Convert any numerical values in query dict to python int/floats
    data_parsed = { k:str_to_num(v) for k, v in request.data.items() }

    # Run simulator
    g0h0, dd, mf_h, mf_hg, mf_hg2 = getattr(simulators, simulator)(**data_parsed)

    # Build response dict
    response = {
        "mf_h": [[x, y] for x, y in zip(g0h0, mf_h)],
        "mf_hg": [[x, y] for x, y in zip(g0h0, mf_hg)],
        "mf_hg2": [[x, y] for x, y in zip(g0h0, mf_hg2)],
        "dd": [[x, y] for x, y in zip(g0h0, dd)],
        }

    return Response(response)

def sim_2to1(simulator, request):
    """
    Runs requested simulation and returns simulated isotherm and molefractions
    as a function of equivalent [G]0/[H]0 fraction.

    Query:
        k1         : float,  Binding constant K1
        k2         : float,  Binding constant K2
        h0_init    : float,  Initial [H]0
        g0h0_final : float,  Max equiv. [G]0/[H]0
        dh         : float,  Free host NMR resonance
        dhg        : float,  Host-Guest complex NMR resonance
        dh2g       : float  Host-Guest2 complex NMR resonance

    Response:
        (all objects are arrays of [x, y] points for plotting
         with equivalent [G]0/[H]0 on x axis)

        dd     : Isotherm
        mf_h   : Host molefraction
        mf_hg  : HG complex molefraction
        mf_h2g : H2G complex molefraction
    """

    # Convert any numerical values in query dict to python int/floats
    data_parsed = { k:str_to_num(v) for k, v in request.data.items() }

    # Run simulator
    g0h0, dd, mf_h, mf_hg, mf_h2g = getattr(simulators, simulator)(**data_parsed)

    # Build response dict
    response = {
        "mf_h": [[x, y] for x, y in zip(g0h0, mf_h)],
        "mf_hg": [[x, y] for x, y in zip(g0h0, mf_hg)],
        "mf_h2g": [[x, y] for x, y in zip(g0h0, mf_h2g)],
        "dd": [[x, y] for x, y in zip(g0h0, dd)],
        }

    return Response(response)



#
# Convenience functions
#

def str_to_num(s):
    # Return float value of string if the string is parsable into float
    if s.isdigit():
        return int(s)
    else:
        try:
            return float(s)
        except ValueError:
            return s
